"""Merge ASR words with diarization turns into speaker-attributed transcript turns.

Why this is word-level and not segment-level: whisper segments are chosen for acoustic/decoding
convenience and routinely straddle a speaker change ("...that's my view. — Well I disagree"), so
attributing a whole segment to one speaker bakes in errors that no amount of diarization accuracy
can fix. Assigning each *word* to the maximally-overlapping diarization turn and then re-grouping
gives boundaries that land where people actually swap.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, asdict
from typing import Iterable

from .textfmt import format_sentences, is_sentence_end


@dataclass
class Word:
    start: float
    end: float
    word: str
    prob: float = 1.0
    speaker: str | None = None  # raw diarization label, filled in by assign_words


@dataclass
class DiarTurn:
    start: float
    end: float
    speaker: str  # raw label, e.g. "SPEAKER_00"


@dataclass
class SentenceScore:
    """One sentence re-scored against the speakers' voice prints by the refine stage."""
    speaker: str | None   # nearest centroid, or None if the span was too short to embed
    margin: float         # cosine gap to the runner-up; 0 when undecidable


@dataclass
class Turn:
    """One contiguous block of speech by one person, as rendered in the transcript."""
    speaker: str          # display label, e.g. "Person 1"
    raw: str              # underlying diarization label
    start: float
    end: float
    text: str             # sentence-per-line formatted
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["words"] = [asdict(w) for w in self.words]
        return d


def assign_words(
    words: list[Word],
    turns: list[DiarTurn],
    orphan_window_s: float = 2.0,
) -> list[Word]:
    """Attach a raw diarization label to every word, in place, and return the list.

    Each word goes to the turn it overlaps most. A word that overlaps nothing (it fell in a
    diarization gap — a laugh, a cough, an over-talked interjection) is attached to the nearest
    turn within `orphan_window_s`; failing that it inherits the previous word's speaker, so a
    stray gap never silently drops text out of the transcript.
    """
    if not turns:
        return words
    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    starts = [t.start for t in ordered]
    # Longest turn bounds how far back an overlapping turn can begin, so the backward scan
    # below can stop early instead of walking the whole list for every word.
    max_dur = max((t.end - t.start) for t in ordered)

    last_speaker: str | None = None
    for w in words:
        ws, we = w.start, w.end
        if we < ws:
            ws, we = we, ws

        best: str | None = None
        best_overlap = 0.0
        nearest: str | None = None
        nearest_gap = float("inf")

        # Turns starting at/after the word's end cannot overlap it, so the overlap search only
        # walks backwards from there, stopping once no earlier turn could still reach the word.
        j = bisect_right(starts, we) - 1
        jj = j
        while jj >= 0 and starts[jj] >= ws - max_dur:
            t = ordered[jj]
            overlap = min(we, t.end) - max(ws, t.start)
            if overlap > best_overlap:
                best_overlap, best = overlap, t.speaker
            jj -= 1

        if best is None:
            # The word fell in a diarization gap. Its nearest turn is one of the two immediate
            # neighbours: the last one starting before the word (j) or the first one starting
            # after it (j + 1). The forward neighbour must be checked explicitly — the backward
            # scan above can never reach it.
            for idx in (j, j + 1):
                if 0 <= idx < len(ordered):
                    t = ordered[idx]
                    if t.start > we:
                        gap = t.start - we
                    elif t.end < ws:
                        gap = ws - t.end
                    else:
                        gap = 0.0
                    if gap < nearest_gap:
                        nearest_gap, nearest = gap, t.speaker
            if nearest is not None and nearest_gap <= orphan_window_s:
                best = nearest
        w.speaker = best if best is not None else last_speaker
        if w.speaker is not None:
            last_speaker = w.speaker

    # Words before the first diarized turn have no predecessor to inherit from; back-fill them
    # from the first word that did get a label.
    first = next((w.speaker for w in words if w.speaker), None)
    if first:
        for w in words:
            if w.speaker is None:
                w.speaker = first
            else:
                break
    return words


def sentence_spans(words: list[Word]) -> list[list[int]]:
    """Group word indices into sentences using the renderer's own boundary rule.

    Speaker attribution is decided per sentence rather than per word: a diarization boundary that
    lands mid-sentence is almost always the diarizer being late or early, not two people splitting
    a clause, and honouring it tears one utterance across two speaker blocks. Deciding whole
    sentences makes each attribution rest on seconds of evidence instead of one 200 ms token.
    """
    spans: list[list[int]] = []
    cur: list[int] = []
    for i, w in enumerate(words):
        cur.append(i)
        if is_sentence_end(w.word, alone=len(cur) == 1):
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return spans


def vote_sentences(
    words: list[Word],
    spans: list[list[int]],
    scores: list[SentenceScore] | None = None,
    margin: float = 0.15,
    min_seconds: float = 0.8,
) -> set[int]:
    """Collapse each sentence onto one speaker; return the indices of confidently-scored words.

    The diarization overlap is the prior: every word in a sentence takes that sentence's
    duration-weighted majority label, so a long word counts for more than "ja". Where the refine
    stage compared the sentence's own audio against the speakers' voice prints and came back
    decisive, its verdict wins instead — that is what repairs a span the diarizer labelled
    outright wrong, which no amount of re-grouping can fix.

    Returned indices are "locked": their speaker rests on acoustic evidence rather than on a
    boundary guess, so flicker smoothing must not move them afterwards.
    """
    locked: set[int] = set()
    for n, span in enumerate(spans):
        if not span:
            continue
        weight: dict[str, float] = {}
        for i in span:
            w = words[i]
            if w.speaker:
                weight[w.speaker] = weight.get(w.speaker, 0.0) + max(w.end - w.start, 0.05)
        winner = max(weight, key=lambda k: weight[k]) if weight else None

        sc = scores[n] if scores and n < len(scores) else None
        if (sc is not None and sc.speaker is not None and sc.margin >= margin
                and (words[span[-1]].end - words[span[0]].start) >= min_seconds):
            winner = sc.speaker
            locked.update(span)
        if winner is not None:
            for i in span:
                words[i].speaker = winner
    return locked


def smooth_flicker(words: list[Word], min_run: int = 3,
                   protected: set[int] | None = None) -> list[Word]:
    """Absorb runs shorter than `min_run` words into the surrounding speaker.

    Diarization boundaries wobble around backchannels ("mhm", "ja", "right"), and without this
    the transcript ping-pongs between two people every few words, which is unreadable. Only runs
    flanked by the SAME speaker on both sides are absorbed — a genuine short reply between two
    different speakers is left alone.

    A run holding any `protected` index is never absorbed: those words were placed by matching
    their audio against the speakers' voice prints, which is stronger evidence than the
    surrounding boundaries this heuristic exists to paper over.
    """
    if min_run <= 1 or len(words) < 3:
        return words
    protected = protected or set()

    runs: list[list[int]] = []
    for i, w in enumerate(words):
        if runs and words[runs[-1][0]].speaker == w.speaker:
            runs[-1].append(i)
        else:
            runs.append([i])

    changed = True
    while changed:
        changed = False
        for r in range(1, len(runs) - 1):
            prev_sp = words[runs[r - 1][0]].speaker
            next_sp = words[runs[r + 1][0]].speaker
            if any(i in protected for i in runs[r]):
                continue
            if len(runs[r]) < min_run and prev_sp is not None and prev_sp == next_sp:
                for i in runs[r]:
                    words[i].speaker = prev_sp
                runs[r - 1:r + 2] = [runs[r - 1] + runs[r] + runs[r + 1]]
                changed = True
                break
    return words


def group_turns(
    words: list[Word],
    turn_gap_s: float = 1.5,
    max_turn_chars: int = 1200,
) -> list[Turn]:
    """Group consecutive same-speaker words into readable turns.

    Breaks on: speaker change, a silence longer than `turn_gap_s`, or a turn growing past
    `max_turn_chars`. The last rule matters for lectures, where one person can hold the floor for
    forty minutes and would otherwise become a single unreadable wall of text.
    """
    out: list[Turn] = []
    if not words:
        return out

    cur: list[Word] = []
    cur_chars = 0

    def flush() -> None:
        nonlocal cur, cur_chars
        if not cur:
            return
        raw = cur[0].speaker or "SPEAKER_00"
        text = format_sentences(" ".join(w.word.strip() for w in cur if w.word.strip()))
        if text:
            out.append(Turn(speaker="", raw=raw, start=cur[0].start, end=cur[-1].end,
                            text=text, words=list(cur)))
        cur, cur_chars = [], 0

    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            if (w.speaker != cur[-1].speaker
                    or gap > turn_gap_s
                    or cur_chars + len(w.word) > max_turn_chars):
                flush()
        cur.append(w)
        cur_chars += len(w.word) + 1
    flush()
    return out


def label_speakers(turns: list[Turn]) -> tuple[list[Turn], list[str]]:
    """Map raw diarization labels to 'Person N', numbered by order of first appearance.

    Numbering by first appearance (rather than by pyannote's arbitrary SPEAKER_xx order) means
    Person 1 is always whoever speaks first — the interviewer, the lecturer — which is what a
    reader expects.
    """
    mapping: dict[str, str] = {}
    for t in turns:
        if t.raw not in mapping:
            mapping[t.raw] = f"Person {len(mapping) + 1}"
    for t in turns:
        t.speaker = mapping[t.raw]
    return turns, list(mapping.values())


def speaker_stats(turns: Iterable[Turn]) -> dict[str, dict]:
    """Per-speaker talk time and share, computed from the rendered turn spans."""
    stats: dict[str, dict] = {}
    total = 0.0
    for t in turns:
        dur = max(0.0, t.end - t.start)
        s = stats.setdefault(t.speaker, {"seconds": 0.0, "turns": 0, "share": 0.0,
                                         "raw": t.raw, "first": t.start, "longest": (0.0, 0.0)})
        s["seconds"] += dur
        s["turns"] += 1
        if dur > s["longest"][1] - s["longest"][0]:
            s["longest"] = (t.start, t.end)
        total += dur
    for s in stats.values():
        s["share"] = (s["seconds"] / total) if total > 0 else 0.0
    return stats


def merge(
    words: list[Word],
    diar_turns: list[DiarTurn],
    turn_gap_s: float = 1.5,
    max_turn_chars: int = 1200,
    flicker_min_words: int = 3,
    orphan_window_s: float = 2.0,
    sentence_atomic: bool = True,
    sentence_scores: list[SentenceScore] | None = None,
    refine_margin: float = 0.15,
    refine_min_seconds: float = 0.8,
) -> tuple[list[Turn], dict[str, dict]]:
    """Full pipeline: assign -> vote per sentence -> smooth -> group -> label -> stats."""
    words = [w for w in words if (w.word or "").strip()]
    if not words:
        return [], {}
    if diar_turns:
        assign_words(words, diar_turns, orphan_window_s=orphan_window_s)
        locked: set[int] = set()
        if sentence_atomic:
            locked = vote_sentences(words, sentence_spans(words), scores=sentence_scores,
                                    margin=refine_margin, min_seconds=refine_min_seconds)
        smooth_flicker(words, min_run=flicker_min_words, protected=locked)
    else:
        # No diarization result (single-speaker file, or the stage was skipped): keep the text.
        for w in words:
            w.speaker = "SPEAKER_00"
    turns = group_turns(words, turn_gap_s=turn_gap_s, max_turn_chars=max_turn_chars)
    turns, _ = label_speakers(turns)
    return turns, speaker_stats(turns)
