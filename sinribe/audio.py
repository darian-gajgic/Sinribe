"""Audio probing and decoding via the system ffmpeg (8.0.1 on this box).

Everything downstream consumes ONE canonical artefact: 16 kHz mono signed-16 PCM WAV. Decoding
once up front means the ASR worker and the diarization worker (which live in different venvs with
incompatible CUDA stacks) can share the same bytes, and neither has to care about the input
container. A 4-hour recording lands at ~460 MB, which is cheap next to re-decoding it twice.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SAMPLE_RATE = 16_000

# Optional conditioning applied while decoding to the canonical WAV. Whisper was trained on
# messy audio and generally prefers to be left alone, so none of this is on by default — these
# exist to be measured against "" by sinribe.eval, and only ship on a rung where they won.
#
#   clean    rumble removal plus slow level equalisation, for a recording made across a room
#            where one voice is much further from the microphone than the other.
#   denoise  the same, plus FFT noise reduction, for continuous background noise (fan, traffic).
FILTER_CHAINS = {
    "": "",
    "clean": "highpass=f=70,dynaudnorm=f=200:g=11:p=0.95:m=8",
    "denoise": "highpass=f=70,afftdn=nr=12:nf=-40,dynaudnorm=f=200:g=11:p=0.95:m=8",
}

AUDIO_SUFFIXES = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aiff", ".aif",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4b", ".amr", ".3gp", ".ts", ".mpg", ".mpeg",
}


class AudioError(RuntimeError):
    pass


@dataclass
class MediaInfo:
    path: Path
    duration: float          # seconds
    sample_rate: int
    channels: int
    codec: str

    @property
    def has_audio(self) -> bool:
        return self.duration > 0 and self.channels > 0


def probe(path: str | Path) -> MediaInfo:
    """Read duration/codec via ffprobe. Raises AudioError if there is no usable audio stream."""
    path = Path(path)
    if not path.exists():
        raise AudioError(f"File not found: {path}")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except FileNotFoundError as e:
        raise AudioError("ffprobe not found — install ffmpeg") from e
    except subprocess.TimeoutExpired as e:
        raise AudioError(f"ffprobe timed out on {path.name}") from e
    if out.returncode != 0:
        raise AudioError(f"ffprobe failed: {out.stderr.strip()[:300]}")
    try:
        data = json.loads(out.stdout or "{}")
    except json.JSONDecodeError as e:
        raise AudioError(f"ffprobe returned unparseable output for {path.name}") from e

    streams = data.get("streams") or []
    if not streams:
        raise AudioError(f"{path.name} contains no audio stream")
    st = streams[0]

    duration = 0.0
    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        duration = _duration_by_decode(path)

    return MediaInfo(
        path=path,
        duration=duration,
        sample_rate=int(st.get("sample_rate") or 0),
        channels=int(st.get("channels") or 0),
        codec=str(st.get("codec_name") or "?"),
    )


def _duration_by_decode(path: Path) -> float:
    """Fallback for containers with no duration in the header (some .ogg/.ts streams).

    Decodes to null and reads the final timestamp. Slow-ish but only ever runs when ffprobe
    could not answer, and an unknown duration would otherwise break the progress bar.
    """
    cmd = ["ffmpeg", "-nostdin", "-i", str(path), "-map", "0:a:0", "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    last = 0.0
    for m in re.finditer(r"time=(\d+):(\d\d):(\d\d(?:\.\d+)?)", out.stderr or ""):
        last = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return last


def decode_to_wav(
    src: str | Path,
    dest: str | Path,
    duration: float,
    on_progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    filter_chain: str = "",
) -> Path:
    """Decode any input to 16 kHz mono s16le WAV, reporting fractional progress 0.0-1.0.

    Progress comes from ffmpeg's `-progress pipe:1` machine-readable stream (out_time_us),
    which is exact rather than scraped from the human stderr log.

    `filter_chain` names an entry in FILTER_CHAINS; the default leaves the audio untouched.
    """
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    if filter_chain not in FILTER_CHAINS:
        raise AudioError(f"unknown audio filter {filter_chain!r}")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(src),
        "-map", "0:a:0",
        "-vn", "-sn", "-dn",
        *(("-af", FILTER_CHAINS[filter_chain]) if FILTER_CHAINS[filter_chain] else ()),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        # Explicit muxer: the temp file ends in ".part", so ffmpeg cannot infer the format
        # from the extension and would refuse to open the output.
        "-f", "wav",
        "-progress", "pipe:1", "-nostats",
        str(tmp),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_us=") and duration > 0 and on_progress:
                raw = line.split("=", 1)[1]
                if raw and raw != "N/A":
                    try:
                        on_progress(min(1.0, (int(raw) / 1e6) / duration))
                    except ValueError:
                        pass
            if should_cancel and should_cancel():
                proc.kill()
                proc.wait(timeout=10)
                tmp.unlink(missing_ok=True)
                raise AudioError("cancelled")
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    if proc.returncode != 0:
        err = (proc.stderr.read() if proc.stderr else "") or ""
        tmp.unlink(missing_ok=True)
        raise AudioError(f"ffmpeg decode failed: {err.strip()[:400]}")
    if not tmp.exists() or tmp.stat().st_size < 1024:
        tmp.unlink(missing_ok=True)
        raise AudioError("ffmpeg produced no audio — is there a decodable audio stream?")

    tmp.replace(dest)
    if on_progress:
        on_progress(1.0)
    return dest


def extract_clip(src: str | Path, dest: str | Path, start: float, length: float) -> Path:
    """Cut a short preview clip (used by the speaker-rename panel's play button)."""
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{max(0.0, start):.3f}", "-t", f"{max(0.5, length):.3f}",
        "-i", str(src), "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le", str(dest),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    if out.returncode != 0:
        raise AudioError(f"clip extraction failed: {out.stderr.strip()[:200]}")
    return dest
