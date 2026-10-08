#!/usr/bin/env python3
"""Parse the four transcripts into a common segment format."""
import os, re, json, unicodedata

# Working directory holding the inputs and receiving the intermediate files.
# Override with:  SINRIBE_BENCH_DIR=/path/to/data python analyze.py
BASE = os.environ.get("SINRIBE_BENCH_DIR", os.getcwd()).rstrip("/") + "/"
PROJ = BASE

# Expected inputs in BASE:
#   manual.txt   human reference   (docx -> text, one paragraph per line)
#   sinribe.md   Sinribe output
#   vibe.txt     Vibe output       (docx -> text)
#   claude_transcript.txt           baseline output from transcribe.py


def hms(s):
    p = [int(x) for x in s.split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return p[0] * 3600 + p[1] * 60 + p[2]


MANUAL_NOTE = re.compile(r"^Verbindung(s)?abbruch|^Interview Coach", re.I)


def parse_manual(path):
    """Handles: '[hh:mm:ss ] Spk: text', missing colon after speaker,
    bare continuation lines, and standalone production notes."""
    segs, notes = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\[(\d{1,2}:\d{2}:\d{2})\s*\]\s*(Interviewer|Interview-\s?Partner)\s*:?\s*(.*)$",
                     line, re.I)
        if m:
            segs.append({"t": hms(m.group(1)),
                         "spk": re.sub(r"\s+", "", m.group(2)).replace("Interview-Partner", "Interview-Partner"),
                         "text": m.group(3).strip()})
        elif MANUAL_NOTE.match(line):
            notes.append(line)
        elif segs:                      # bare continuation of the previous turn
            segs[-1]["text"] += " " + line
    return segs, notes


def parse_vibe(path):
    segs = []
    lines = [l.rstrip() for l in open(path, encoding="utf-8")]
    i = 0
    while i < len(lines):
        m = re.match(r"^(\d{1,3}:\d{2}(?::\d{2})?)\s*-->\s*(\d{1,3}:\d{2}(?::\d{2})?)$", lines[i].strip())
        if m:
            buf = []
            j = i + 1
            while j < len(lines) and not re.match(r"^\d{1,3}:\d{2}(?::\d{2})?\s*-->", lines[j].strip()):
                if lines[j].strip():
                    buf.append(lines[j].strip())
                j += 1
            if buf:
                segs.append({"t": hms(m.group(1)), "spk": None, "text": " ".join(buf)})
            i = j
        else:
            i += 1
    return segs


def parse_sinribe(path):
    segs = []
    lines = [l.rstrip() for l in open(path, encoding="utf-8")]
    i = 0
    while i < len(lines):
        m = re.match(r"^\*\*\[(\d{1,2}:\d{2}:\d{2})\]\s*(.+?)\*\*$", lines[i].strip())
        if m:
            buf = []
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("**["):
                if lines[j].strip() and not lines[j].strip().startswith("---"):
                    buf.append(lines[j].strip())
                j += 1
            segs.append({"t": hms(m.group(1)), "spk": m.group(2).strip(), "text": " ".join(buf)})
            i = j
        else:
            i += 1
    return segs


def parse_claude(path):
    segs = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$", line.strip())
        if m and m.group(2):
            segs.append({"t": hms(m.group(1)), "spk": None, "text": m.group(2).strip()})
    return segs


# ---------- normalisation for WER ----------
ANNOT = re.compile(r"\((?:lacht|lachen|lachend|hustet|räuspert sich|Pause|unverständlich)[^)]*\)", re.I)
UNVERST = re.compile(r"\bunverst[äa]ndlich\b", re.I)

# German number words are left as-is; we only normalise orthography/punctuation.
SUBS = [
    (r"[„“”\"»«]", " "), (r"[’‘']", "'"), (r"[–—-]", " "),
    (r"[.,;:!?…]", " "), (r"[()\[\]]", " "), (r"/", " "),
]


def normalize(text, drop_annotations=True):
    t = text
    if drop_annotations:
        t = ANNOT.sub(" ", t)
        t = UNVERST.sub(" ", t)
    t = unicodedata.normalize("NFC", t)
    t = t.lower()
    for pat, rep in SUBS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def words(text):
    return normalize(text).split()


def load_all():
    manual, notes = parse_manual(BASE + "manual.txt")
    data = {
        "manual": manual,
        "sinribe": parse_sinribe(PROJ + "sinribe.md"),
        "vibe": parse_vibe(BASE + "vibe.txt"),
    }
    import os
    if os.path.exists(BASE + "claude_transcript.txt"):
        c = parse_claude(BASE + "claude_transcript.txt")
        if c:
            data["claude"] = c
    return data, notes


if __name__ == "__main__":
    data, notes = load_all()
    print("manual production notes:", notes)
    for k, v in data.items():
        w = sum(len(words(s["text"])) for s in v)
        spk = sorted({s["spk"] for s in v if s["spk"]})
        print(f"{k:9s} segments={len(v):5d} words={w:6d} last_ts={v[-1]['t']:6d}s speakers={spk}")
    json.dump(data, open(BASE + "parsed.json", "w"), ensure_ascii=False)
