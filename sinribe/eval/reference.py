"""Load a transcript — human reference or Sinribe output — into a flat list of turns.

Handles the formats a reference actually arrives in: Word (.docx), LibreOffice (.odt), plain
text/Markdown, Sinribe's own Markdown, and SRT/VTT. Nothing here needs python-docx; both office
formats are zip archives with one XML part, which `zipfile` + `ElementTree` read in a dozen lines.

The one non-obvious case this must survive: a reference stitched together from several recordings,
whose timestamps RESTART at each part. The Feuerläufer interview is exactly that — its
`[00:10:52]` in part two is audio minute 36:51 — and a loader that assumed monotonic time would
either sort the turns into nonsense or reject the file. Parts are detected and numbered instead,
and scoring never relies on reference timestamps for anything but diagnostics (see align.py).
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ODT_TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"

# "[00:00:00] Interviewer: ..." / "[0:00] Person 1 – ..." / "00:00:00 Speaker:"
#
# The separator is a colon, or a dash with space on BOTH sides. A bare dash cannot end the name:
# "Interview-Partner:" would otherwise stop at the hyphen and report a speaker called "Interview",
# splitting one person into two and wrecking the attribution score.
_LINE = re.compile(
    r"""^\s*
        \**\[?\s*
        (?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,]\d+)?
        \s*\]?\**
        \s*
        (?:(?P<spk>[^:\n]{1,40}?)\s*(?::|\s[–—-]\s))?
        \s*(?P<text>.*)$
    """,
    re.VERBOSE,
)

# Sinribe's own heading: "**[00:00:00] Person 1**" with the text on following lines.
_MD_HEAD = re.compile(r"^\*\*\[(\d{1,2}):(\d{2}):(\d{2})\]\s+(?P<spk>[^*]+?)\*\*\s*$")

_SRT_TIME = re.compile(
    r"^(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})")

# Editorial annotations a human transcriber leaves behind. They mark audio the *reference* could
# not resolve, so they are not text the ASR was ever supposed to produce; scoring drops them
# rather than counting them as words the app missed.
INAUDIBLE = re.compile(
    r"""\(\s*(?:un)?verst[aä]ndlich\s*\)      # (unverständlich)
      | \[\s*(?:un)?verst[aä]ndlich\s*\]
      | \bnicht\s+verst[aä]ndlich\b
      | \bunverst[aä]ndlich\b
      | \bversteh(?:e)?\s+ich\s+nicht\b
      | \(\s*inaudible\s*\)
      | \[\s*inaudible\s*\]
      | \(\s*lacht\s*\)                        # stage direction
      | \(\s*laughs?\s*\)
      | \[\s*[^\]]{0,30}\s*\]                  # any short bracketed aside
    """,
    re.VERBOSE | re.IGNORECASE,
)

# A backwards jump larger than this means a new recording, not a typo in a timestamp.
PART_BREAK_S = 60.0


@dataclass
class RefWord:
    """A single word with the timing and confidence the recogniser reported for it."""
    word: str
    start: float
    end: float
    prob: float = 0.0


@dataclass
class RefTurn:
    """One labelled block of speech as written in the source document."""
    start: float          # seconds AS WRITTEN in the document (not audio time when part > 0)
    speaker: str | None   # whatever the document called them; None if unlabelled
    text: str
    part: int = 0         # 0 unless the document stitches several recordings together
    # Only populated for Sinribe's own output, where the sidecar carries per-word detail. A human
    # document has none, which is why nothing in scoring may *require* it.
    words: list[RefWord] | None = None


def _secs(h: str | None, m: str, s: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s)


def _docx_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    out = []
    for para in root.iter(f"{W_NS}p"):
        buf = []
        for node in para.iter():
            if node.tag == f"{W_NS}t":
                buf.append(node.text or "")
            elif node.tag in (f"{W_NS}br", f"{W_NS}cr"):
                buf.append("\n")
            elif node.tag == f"{W_NS}tab":
                buf.append("\t")
        out.extend("".join(buf).split("\n"))
    return out


def _odt_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("content.xml"))
    out = []
    for para in root.iter():
        if para.tag in (f"{ODT_TEXT_NS}p", f"{ODT_TEXT_NS}h"):
            out.append("".join(para.itertext()))
    return out


def read_lines(path: str | Path) -> list[str]:
    """Extract the document's lines, whatever container it arrived in."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _docx_paragraphs(path)
    if suffix == ".odt":
        return _odt_paragraphs(path)
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _parse_srt(lines: list[str]) -> list[RefTurn]:
    turns: list[RefTurn] = []
    for line in lines:
        m = _SRT_TIME.match(line.strip())
        if m:
            millis = int(m.group(4).ljust(3, "0")) / 1000
            turns.append(RefTurn(start=_secs(m.group(1), m.group(2), m.group(3)) + millis,
                                 speaker=None, text=""))
            continue
        stripped = line.strip()
        if not stripped or stripped.isdigit() or stripped.upper() == "WEBVTT":
            continue
        if turns:
            spk = None
            body = stripped
            # SRT speaker convention: "Person 1: hello" or "- Person 1: hello"
            head = re.match(r"^-?\s*([A-Za-zÄÖÜäöü][^:]{0,30}):\s+(.*)$", stripped)
            if head:
                spk, body = head.group(1).strip(), head.group(2)
            if spk and not turns[-1].speaker:
                turns[-1].speaker = spk
            turns[-1].text = (turns[-1].text + " " + body).strip()
    return [t for t in turns if t.text]


def _parse_sinribe_md(lines: list[str]) -> list[RefTurn] | None:
    """Sinribe's own Markdown, if that is what this is. None means 'not this format'."""
    turns: list[RefTurn] = []
    for raw in lines:
        m = _MD_HEAD.match(raw.strip())
        if m:
            turns.append(RefTurn(
                start=_secs(m.group(1), m.group(2), m.group(3)),
                speaker=m.group("spk").strip(), text=""))
            continue
        if not turns:
            continue  # still in the front matter
        body = raw.strip()
        if not body or body.startswith(("#", "|", "<a id=", "---", "- [**")):
            continue
        turns[-1].text = (turns[-1].text + " " + body).strip()
    return [t for t in turns if t.text] or None


def _parse_timestamped(lines: list[str]) -> list[RefTurn]:
    turns: list[RefTurn] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if m and m.group("text") is not None:
            spk = (m.group("spk") or "").strip() or None
            turns.append(RefTurn(start=_secs(m.group("h"), m.group("m"), m.group("s")),
                                 speaker=spk, text=m.group("text").strip()))
        elif turns:
            # A wrapped continuation line belongs to the turn above it.
            turns[-1].text = (turns[-1].text + " " + line).strip()
    return [t for t in turns if t.text]


def _mark_parts(turns: list[RefTurn]) -> list[RefTurn]:
    """Number the recordings a stitched document is made of, without reordering anything."""
    part = 0
    for i, t in enumerate(turns):
        if i and t.start < turns[i - 1].start - PART_BREAK_S:
            part += 1
        t.part = part
    return turns


def normalise_speakers(turns: list[RefTurn]) -> list[RefTurn]:
    """Collapse the spelling drift a human transcript accumulates.

    Real references contain "Interview-Partner", "Interview- Partner" and "Interview Partner" for
    the same person; left alone they would look like three speakers and wreck the attribution
    score.
    """
    for t in turns:
        if t.speaker:
            key = re.sub(r"[\s\-_.]+", "", t.speaker).lower()
            key = re.sub(r"\(.*?\)", "", key)
            t.speaker = key or None
    return turns


def load_reference(path: str | Path, keep_inaudible: bool = False) -> list[RefTurn]:
    """Load any supported transcript into turns, in document order.

    Editorial "(unverständlich)" markers are stripped by default: they denote audio the human
    could not resolve either, so counting them as words the app failed to produce would charge it
    for the reference's own gaps.
    """
    path = Path(path)
    lines = read_lines(path)
    suffix = path.suffix.lower()

    if suffix in (".srt", ".vtt"):
        turns = _parse_srt(lines)
    else:
        turns = _parse_sinribe_md(lines) or _parse_timestamped(lines)
    if not turns:
        # No timestamps at all — still usable, since alignment is sequence-based.
        body = " ".join(x.strip() for x in lines if x.strip())
        turns = [RefTurn(start=0.0, speaker=None, text=body)] if body else []

    if not keep_inaudible:
        for t in turns:
            t.text = re.sub(r"\s{2,}", " ", INAUDIBLE.sub(" ", t.text)).strip()
        turns = [t for t in turns if t.text]

    return _mark_parts(normalise_speakers(turns))


def load_hypothesis(path: str | Path) -> list[RefTurn]:
    """Load Sinribe's output, preferring the sidecar so per-word timings and confidences survive.

    Given `X.md` the sidecar `X.sinribe.json` is used automatically when it is there. That detail
    matters: word timings are what let the report bin errors over time, and word confidences are
    what let it answer whether the recogniser knew it was struggling.
    """
    path = Path(path)
    sidecar = path
    if path.suffix.lower() != ".json":
        candidate = path.with_suffix(".sinribe.json")
        sidecar = candidate if candidate.exists() else path

    if sidecar.suffix.lower() != ".json":
        return load_reference(path)

    import json
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    turns: list[RefTurn] = []
    for t in data.get("turns", []):
        words = [RefWord(word=w.get("word", ""), start=float(w.get("start", 0.0)),
                         end=float(w.get("end", 0.0)), prob=float(w.get("prob") or 0.0))
                 for w in t.get("words", [])]
        turns.append(RefTurn(start=float(t.get("start", 0.0)),
                             speaker=(t.get("speaker") or None),
                             text=t.get("text", ""),
                             words=words or None))
    return normalise_speakers(turns)
