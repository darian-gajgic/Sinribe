"""Combine several decodes of the same audio into one transcript, by vote.

This is the only mechanism found that turns more compute into fewer errors. It works because the
passes disagree in *different* places: change the voice-activity boundaries or switch from
sequential to batched decoding and whisper re-segments the audio, mishears different words, and
recovers others. Where two passes agree against a third, the majority is usually right.

Measured on the benchmark interview — three passes, voted — **21.3 % WER against 20.8 %**, at
roughly 2.5× the compute. Five passes reach 20.7 % for another 3.5× on top, which is why the
shipped rung uses three.

Contrast with simply searching harder: beam 20 costs 5.6× and returns 21.6 %. Beam search explores
one model's hypotheses, and on this audio the model's *probabilities* are what is wrong; no amount
of searching a wrong distribution fixes it. Voting needs the passes to be genuinely different,
which is why the rung varies segmentation rather than beam width.

The first pass is the pivot: the merged transcript keeps its word timings and only replaces the
word *text* where the vote overrules it. Timings are what speaker attribution is built on, and
measured attribution is unchanged by voting — 98.9 % either way.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from .merge import Word

# Deletion is a candidate like any other: a pass that produced nothing here is voting for "this
# word should not be in the transcript". Weighted below a real word's confidence so that a single
# pass dropping a word cannot outvote two passes that heard one, but high enough that a word only
# one pass invented gets removed.
DELETION_WEIGHT = 0.5

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _key(word: str) -> str:
    """Compare words ignoring case, punctuation and accents; keep the original for output."""
    text = unicodedata.normalize("NFKD", (word or "").strip().lower())
    return _PUNCT.sub("", text).strip()


def _ballots(passes: list[list[Word]]) -> list[list[tuple[str, str, float]]]:
    """One ballot box per pivot position: (comparison key, original text, weight).

    Split out so `agreement` can count the votes `combine` actually cast, rather than trying to
    infer them from the merged output — which does not line up with the pivot position for
    position, because a position the passes vote away is dropped from it.
    """
    pivot = passes[0]
    pivot_keys = [_key(w.word) for w in pivot]
    ballots: list[list[tuple[str, str, float]]] = [
        [(pivot_keys[i], w.word, max(w.prob, 0.01))] for i, w in enumerate(pivot)
    ]

    for other in passes[1:]:
        other_keys = [_key(w.word) for w in other]
        matcher = SequenceMatcher(a=pivot_keys, b=other_keys, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    w = other[j1 + k]
                    ballots[i1 + k].append((other_keys[j1 + k], w.word, max(w.prob, 0.01)))
            elif tag == "replace":
                # Spread this pass's alternatives across the pivot positions they replace, so a
                # two-for-three substitution still puts a candidate in every box.
                span = max(1, i2 - i1)
                for k in range(i1, i2):
                    j = min(j2 - 1, j1 + (k - i1) * (j2 - j1) // span)
                    ballots[k].append((other_keys[j], other[j].word, max(other[j].prob, 0.01)))
            elif tag == "delete":
                for k in range(i1, i2):
                    ballots[k].append(("", "", DELETION_WEIGHT))
            # Insertions are dropped: a word only one pass heard, in a place no other pass put
            # anything, is far more often a hallucination than a rescue.
    return ballots


def combine(passes: list[list[Word]]) -> list[Word]:
    """Merge decodes into one word list. `passes[0]` is the pivot and supplies the timings.

    Returns the pivot's words with text replaced wherever the other passes outvote it, and
    positions removed where the majority says nothing was said.
    """
    passes = [p for p in passes if p]
    if not passes:
        return []
    if len(passes) == 1:
        return list(passes[0])

    pivot = passes[0]
    ballots = _ballots(passes)
    out: list[Word] = []
    for i, cell in enumerate(ballots):
        tally: Counter = Counter()
        spelling: dict[str, str] = {}
        for key, text, weight in cell:
            tally[key] += weight
            # Keep the spelling from the most confident pass that voted for this word.
            if key not in spelling or weight > tally[key] - weight:
                spelling.setdefault(key, text)
        winner, _ = tally.most_common(1)[0]
        if not winner:
            continue                      # the passes agree nothing was said here
        src = pivot[i]
        out.append(Word(start=src.start, end=src.end,
                        word=spelling.get(winner, src.word), prob=src.prob,
                        speaker=src.speaker))
    return out


def agreement(passes: list[list[Word]]) -> float:
    """Share of pivot positions where every pass said the same thing. Diagnostic only.

    Low agreement means the passes are genuinely diverse and voting has something to work with;
    high agreement means the extra passes are costing time and changing nothing.

    Counted from the ballots. The earlier version compared the merged transcript against the
    pivot by index, which silently measured something else: `combine` drops the positions the
    passes vote away, so every position after the first drop was compared against its neighbour.
    Three passes that really agreed on 85 % of words reported 2 %, and 2 % is exactly what a
    reader would take as evidence the voting was broken.
    """
    passes = [p for p in passes if p]
    if len(passes) < 2:
        return 1.0
    ballots = _ballots(passes)
    if not ballots:
        return 1.0
    unanimous = sum(1 for cell in ballots if len({key for key, _, _ in cell}) == 1)
    return unanimous / len(ballots)
