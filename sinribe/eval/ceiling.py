"""How much is left to win — the question that decides whether to keep tuning.

Score every transcript of one recording against the reference and ask what a *perfect* chooser
between them would score. The gap between that and the best single configuration is the entire
prize available to any smarter combination scheme: consensus decoding, per-segment voting,
rescoring, anything. The words no configuration ever gets are the floor, and nothing built out of
these models will go below it.

This is what killed two-model consensus on the benchmark interview before a line of it was
written: perfect selection between large-v3 and the German fine-tune reached 89.5 % against
87.0 % for large-v3 alone. Two and a half points, for a perfect oracle and double the compute.
"""

from __future__ import annotations

from pathlib import Path

from .normalize import Normalizer
from .reference import load_hypothesis, load_reference
from .score import score

# A transcript this bad has not made different mistakes, it has failed. Including a collapsed
# decode would inflate the union with nothing and hide the real picture.
BROKEN_WER = 0.5


def _correct_mask(rep) -> list[bool]:
    mask = [False] * len(rep.alignment.ref)
    for op in rep.alignment.ops:
        if op.kind in ("equal", "ortho"):
            for i in range(op.i1, op.i2):
                mask[i] = True
    return mask


def run_ceiling(directory: Path, reference: Path, keep_fillers: bool = False) -> int:
    """Report the oracle ceiling over every transcript found under `directory`."""
    nz = Normalizer(drop_fillers=not keep_fillers)
    ref = load_reference(reference)
    if not ref:
        print(f"ERROR: no turns parsed from {reference}")
        return 2

    found: list[tuple[str, float, list[bool]]] = []
    skipped: list[tuple[str, float]] = []
    for path in sorted(Path(directory).rglob("*.md")):
        rep = score(ref, load_hypothesis(path), nz)
        name = path.parent.name if path.parent != Path(directory) else path.stem
        if rep.total.wer > BROKEN_WER:
            skipped.append((name, rep.total.wer))
            continue
        found.append((name, rep.total.wer, _correct_mask(rep)))

    if not found:
        print(f"ERROR: no usable transcripts under {directory}")
        return 2

    n = len(found[0][2])
    print(f"{len(found)} configurations against {n} reference words\n")
    for name, wer, mask in sorted(found, key=lambda r: r[1]):
        print(f"  {name:<30} WER {wer:>6.1%}   correct {sum(mask) / n:>6.1%}")
    for name, wer in skipped:
        print(f"  {name:<30} WER {wer:>6.1%}   excluded as broken")

    union = sum(any(mask[i] for _, _, mask in found) for i in range(n))
    best = max(sum(mask) for _, _, mask in found)
    print()
    print(f"  best single configuration      {best / n:>6.1%} correct")
    print(f"  perfect choice between them    {union / n:>6.1%} correct   "
          f"(+{(union - best) / n:.1%} available to any combination scheme)")
    print(f"  wrong in every configuration   {(n - union) / n:>6.1%}   "
          f"— the floor for these models on this recording")
    return 0
