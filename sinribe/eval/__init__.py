"""Transcript accuracy measurement.

Sinribe's decode settings are full of judgement calls (which model, how much context, how wide a
beam, whether to touch the audio at all) and every one of them is testable against a hand-made
reference. This package is the scoreboard: load a human transcript, align it with what the app
produced, and report WER — overall, per speaker, and over time.

It exists because the alternative is guessing. An earlier attempt to push whisper toward verbatim
with a filler-laden `initial_prompt` sounded obviously right and measurably made accuracy worse;
nothing else here ships without a number next to it.
"""

from .align import Alignment, align
from .normalize import Normalizer, norm_words
from .reference import RefTurn, RefWord, load_hypothesis, load_reference
from .score import Report, score

__all__ = [
    "Alignment",
    "Normalizer",
    "RefTurn",
    "RefWord",
    "Report",
    "align",
    "load_hypothesis",
    "load_reference",
    "norm_words",
    "score",
]
