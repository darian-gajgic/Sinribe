"""Download audio from a URL (YouTube podcasts and anything else yt-dlp handles).

This is the only part of Sinribe that touches the network, and only when you paste a link:
transcription, diarization and rendering stay entirely local. The downloaded audio is cached in
`~/.cache/sinribe/downloads/` keyed by the site's own video id, so re-running the same link — or
resuming a cancelled job — never downloads twice.

Nothing is re-encoded here. The best available audio stream is kept in its native container and
`audio.decode_to_wav` normalises it later, which avoids a pointless lossy generation.
"""

from __future__ import annotations

import re
import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

URL_RE = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)

# Containers yt-dlp commonly yields for `bestaudio`; used to find an already-cached download.
_AUDIO_GLOB = "*"


class FetchError(RuntimeError):
    pass


class FetchCancelled(RuntimeError):
    pass


@dataclass
class MediaMeta:
    url: str
    video_id: str
    title: str
    uploader: str
    duration: float          # seconds; 0.0 if the site did not report one
    extractor: str
    is_live: bool
    webpage_url: str
    # When the site says the media was published, as YYYY-MM-DD, or "" if it did not say. The
    # web check needs this: a claim can only be "outdated" relative to when it was made, and a
    # two-month-old AI video is a different object from a two-year-old one.
    published: str = ""
    # The uploader's own description and chapter markers. The description is where podcasts
    # list the papers, tools and people they mention, so its links are real provenance; the
    # chapters are the uploader's outline of the episode and a strong hint for the summary's.
    description: str = ""
    chapters: list[dict] = field(default_factory=list)

    @property
    def display(self) -> str:
        who = f" — {self.uploader}" if self.uploader else ""
        return f"{self.title}{who}"


def is_url(text: str | None) -> bool:
    return bool(text and URL_RE.match(text))


class _QuietLogger:
    """yt-dlp writes errors straight to stderr even with quiet=True. We raise our own
    FetchError with a friendlier message, so swallow its copy rather than printing both."""

    def debug(self, msg):  # noqa: D102
        pass

    def info(self, msg):  # noqa: D102
        pass

    def warning(self, msg):  # noqa: D102
        pass

    def error(self, msg):  # noqa: D102
        pass


def _ydl_opts(cookies_from_browser: str | None = None) -> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _QuietLogger(),
        # A podcast link is very often "watch?v=X&list=Y". Without this yt-dlp would queue the
        # entire playlist instead of the episode that was actually pasted.
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


def _published(info: dict) -> str:
    """The publication date as YYYY-MM-DD, from whichever field this site filled in."""
    for key in ("release_date", "upload_date"):
        raw = str(info.get(key) or "")
        if re.fullmatch(r"\d{8}", raw):
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    for key in ("release_timestamp", "timestamp"):
        stamp = info.get(key)
        if isinstance(stamp, (int, float)) and stamp > 0:
            return _dt.datetime.fromtimestamp(stamp, _dt.timezone.utc).date().isoformat()
    return ""


def _chapters(info: dict) -> list[dict]:
    out = []
    for chapter in info.get("chapters") or []:
        if not isinstance(chapter, dict):
            continue
        title = str(chapter.get("title") or "").strip()
        start = chapter.get("start_time")
        if title and isinstance(start, (int, float)):
            out.append({"title": title[:200], "start": float(start)})
    return out[:80]


def _meta_from_info(info: dict, url: str) -> MediaMeta:
    return MediaMeta(
        url=url,
        video_id=str(info.get("id") or "unknown"),
        title=str(info.get("title") or "Untitled"),
        uploader=str(info.get("uploader") or info.get("channel") or ""),
        duration=float(info.get("duration") or 0.0),
        extractor=str(info.get("extractor_key") or info.get("extractor") or "?"),
        is_live=bool(info.get("is_live")),
        webpage_url=str(info.get("webpage_url") or url),
        published=_published(info),
        description=str(info.get("description") or "")[:5000],
        chapters=_chapters(info),
    )


def _friendly(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "sign in to confirm your age" in low or "age-restricted" in low:
        return ("This video is age-restricted. Set \"yt_cookies_from_browser\" (e.g. \"firefox\") "
                "in ~/.config/sinribe/config.json so yt-dlp can use your browser login.")
    if "private video" in low:
        return "This video is private."
    if "members-only" in low or "join this channel" in low:
        return ("This is members-only content. Set \"yt_cookies_from_browser\" in "
                "~/.config/sinribe/config.json to use your browser login.")
    if "video unavailable" in low or "not available" in low:
        return "This video is unavailable (removed, or blocked in this region)."
    if "unsupported url" in low:
        return "yt-dlp does not recognise this link."
    if "unable to download webpage" in low or "failed to resolve" in low:
        return "Could not reach the site — check your internet connection."
    # yt-dlp prefixes its own messages with "ERROR: "; strip it so the UI reads cleanly.
    return re.sub(r"^ERROR:\s*", "", msg).strip()[:400]


def probe(url: str, cookies_from_browser: str | None = None) -> MediaMeta:
    """Read title/duration WITHOUT downloading, so the UI can show what it is about to fetch."""
    import yt_dlp

    opts = _ydl_opts(cookies_from_browser)
    opts["skip_download"] = True
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:  # noqa: BLE001 - yt_dlp raises a wide variety
        raise FetchError(_friendly(e)) from e

    if info is None:
        raise FetchError("No media found at that link.")
    # A playlist slipped through (e.g. a bare /playlist?list= URL): take the first entry.
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise FetchError("That link is an empty playlist.")
        info = entries[0]

    meta = _meta_from_info(info, url)
    if meta.is_live:
        raise FetchError("This is a live stream. Wait until it has finished, then transcribe it.")
    return meta


def cached_download(dest_dir: Path, video_id: str) -> Path | None:
    """Return an already-downloaded file for this video id, if one is present."""
    if not dest_dir.exists():
        return None
    for p in sorted(dest_dir.glob(f"{video_id}.{_AUDIO_GLOB}")):
        if p.is_file() and not p.name.endswith(".part") and p.stat().st_size > 4096:
            return p
    return None


def download_audio(
    url: str,
    dest_dir: Path,
    meta: MediaMeta | None = None,
    cookies_from_browser: str | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[Path, MediaMeta]:
    """Download the best audio-only stream. Returns (path, metadata).

    Progress is reported as a 0..1 fraction plus a human detail string (size and speed).
    """
    import yt_dlp

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if meta is None:
        meta = probe(url, cookies_from_browser)

    existing = cached_download(dest_dir, meta.video_id)
    if existing is not None:
        if on_progress:
            on_progress(1.0, f"already downloaded ({existing.stat().st_size / 1e6:.0f} MB)")
        return existing, meta

    def hook(d: dict) -> None:
        if should_cancel and should_cancel():
            # The only way out of yt-dlp's download loop is to raise from the hook.
            raise FetchCancelled("cancelled by user")
        if d.get("status") != "downloading" or not on_progress:
            return
        got = d.get("downloaded_bytes") or 0
        total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
        speed = d.get("speed") or 0
        detail = f"{got / 1e6:.0f} MB"
        if total:
            detail += f" of {total / 1e6:.0f} MB"
        if speed:
            detail += f" at {speed / 1e6:.1f} MB/s"
        on_progress((got / total) if total else 0.0, detail)

    opts = _ydl_opts(cookies_from_browser)
    opts.update({
        # Audio-only when the site offers it; fall back to a muxed stream otherwise. No
        # re-encoding — whatever container arrives is decoded to WAV in the next stage anyway.
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "progress_hooks": [hook],
        "overwrites": False,
        "continuedl": True,
    })

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except FetchCancelled:
        raise
    except Exception as e:  # noqa: BLE001
        raise FetchError(_friendly(e)) from e

    if info is None:
        raise FetchError("Download produced no media.")
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise FetchError("That link is an empty playlist.")
        info = entries[0]

    meta = _meta_from_info(info, url)
    path = cached_download(dest_dir, meta.video_id)
    if path is None:
        raise FetchError("Download finished but no audio file was found on disk.")
    if on_progress:
        on_progress(1.0, f"{path.stat().st_size / 1e6:.0f} MB")
    return path, meta


def purge_old_downloads(dest_dir: Path, days: int = 14) -> int:
    """Delete cached downloads older than `days`. Podcasts are large; this keeps disk in check."""
    import time
    if not Path(dest_dir).exists():
        return 0
    cutoff = time.time() - days * 86400
    n = 0
    for p in Path(dest_dir).iterdir():
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                n += 1
        except OSError:
            continue
    return n
