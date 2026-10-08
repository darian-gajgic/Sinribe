"""Line two word sequences up against each other.

Alignment is deliberately SEQUENCE-based, never time-based. A reference stitched from two
recordings has timestamps that restart mid-document, and a hypothesis produced by a different
decode drifts by a second or two anyway; matching on word order sidesteps both problems, and it is
what made the benchmark reference usable without hand-repairing its timestamps first.

Two passes:

1. `difflib` finds the long anchors — fast on the ~9 000-word sequences a full interview produces,
   and its matching blocks are exactly the stretches where the two transcripts agree.
2. Every disagreeing region is then re-aligned with a real edit-distance DP that knows about
   German compounds: `soft skills` against `softskills`, or `gern habt` against `gernhabt`, is a
   spelling choice, not a recognition error, and is recorded as `ortho` and scored as correct.

The regions from step 2 are short (a handful of words), so the DP costs nothing while giving the
error counts inside them the accuracy a plain LCS walk would not.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

# How many words either side may be merged when testing a compound. Three covers everything real
# ("hand aufs herz" / "handaufsherz"); larger windows start inventing matches out of coincidence.
MAX_COMPOUND_SPAN = 3


@dataclass(frozen=True)
class Op:
    """One aligned region. `kind` is equal | ortho | sub | del | ins."""
    kind: str
    i1: int   # reference span [i1, i2)
    i2: int
    j1: int   # hypothesis span [j1, j2)
    j2: int

    @property
    def n_ref(self) -> int:
        return self.i2 - self.i1

    @property
    def n_hyp(self) -> int:
        return self.j2 - self.j1


@dataclass
class Alignment:
    ref: list[str]
    hyp: list[str]
    ops: list[Op]

    @property
    def correct(self) -> int:
        """Reference words the hypothesis got right, counting spelling variants as right."""
        return sum(o.n_ref for o in self.ops if o.kind in ("equal", "ortho"))

    @property
    def substitutions(self) -> int:
        return sum(o.n_ref for o in self.ops if o.kind == "sub")

    @property
    def deletions(self) -> int:
        return sum(o.n_ref for o in self.ops if o.kind == "del")

    @property
    def insertions(self) -> int:
        return sum(o.n_hyp for o in self.ops if o.kind == "ins")

    @property
    def wer(self) -> float:
        n = len(self.ref)
        if not n:
            return 0.0
        return (self.substitutions + self.deletions + self.insertions) / n

    def ref_op(self) -> list[Op | None]:
        """One entry per reference word: the op that consumed it (None is impossible)."""
        out: list[Op | None] = [None] * len(self.ref)
        for o in self.ops:
            for i in range(o.i1, o.i2):
                out[i] = o
        return out

    def hyp_op(self) -> list[Op | None]:
        """One entry per hypothesis word: the op that produced it."""
        out: list[Op | None] = [None] * len(self.hyp)
        for o in self.ops:
            for j in range(o.j1, o.j2):
                out[j] = o
        return out


def _compound_match(ref: list[str], hyp: list[str],
                    i: int, di: int, j: int, dj: int) -> bool:
    """Do these two runs spell the same thing once the spaces are removed?"""
    return "".join(ref[i:i + di]) == "".join(hyp[j:j + dj])


def _refine(ref: list[str], hyp: list[str],
            i1: int, i2: int, j1: int, j2: int) -> list[Op]:
    """Edit-distance alignment of one disagreeing region, with free compound merges."""
    n, m = i2 - i1, j2 - j1
    if n == 0:
        return [Op("ins", i1, i1, j1, j2)] if m else []
    if m == 0:
        return [Op("del", i1, i2, j1, j1)]

    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for a in range(n + 1):
        for b in range(m + 1):
            cur = dp[a][b]
            if cur == inf:
                continue
            if a < n and cur + 1 < dp[a + 1][b]:
                dp[a + 1][b] = cur + 1
                back[a + 1][b] = (a, b, "del")
            if b < m and cur + 1 < dp[a][b + 1]:
                dp[a][b + 1] = cur + 1
                back[a][b + 1] = (a, b, "ins")
            if a < n and b < m:
                same = ref[i1 + a] == hyp[j1 + b]
                cost = cur + (0.0 if same else 1.0)
                if cost < dp[a + 1][b + 1]:
                    dp[a + 1][b + 1] = cost
                    back[a + 1][b + 1] = (a, b, "equal" if same else "sub")
            # Free multi-word merges, but only when the letters agree exactly. Priced a hair
            # above 0 so that when a region can be explained either as a plain match or as a
            # compound, the plain match wins and the report stays readable.
            for da in range(1, min(MAX_COMPOUND_SPAN, n - a) + 1):
                for db in range(1, min(MAX_COMPOUND_SPAN, m - b) + 1):
                    if da == 1 and db == 1:
                        continue
                    if not _compound_match(ref, hyp, i1 + a, da, j1 + b, db):
                        continue
                    cost = cur + 1e-6
                    if cost < dp[a + da][b + db]:
                        dp[a + da][b + db] = cost
                        back[a + da][b + db] = (a, b, "ortho")

    ops: list[Op] = []
    a, b = n, m
    while (a, b) != (0, 0):
        step = back[a][b]
        if step is None:            # unreachable in a complete DP, but never loop forever
            ops.append(Op("sub", i1, i1 + a, j1, j1 + b))
            break
        pa, pb, kind = step
        ops.append(Op(kind, i1 + pa, i1 + a, j1 + pb, j1 + b))
        a, b = pa, pb
    ops.reverse()
    return _coalesce(ops)


def _coalesce(ops: list[Op]) -> list[Op]:
    """Merge adjacent ops of the same kind so the report reads in phrases, not single words."""
    out: list[Op] = []
    for op in ops:
        if out and out[-1].kind == op.kind and out[-1].i2 == op.i1 and out[-1].j2 == op.j1:
            prev = out[-1]
            out[-1] = Op(op.kind, prev.i1, op.i2, prev.j1, op.j2)
        else:
            out.append(op)
    return out


def align(ref: list[str], hyp: list[str]) -> Alignment:
    """Align two normalised word lists."""
    ops: list[Op] = []
    matcher = SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            ops.append(Op("equal", i1, i2, j1, j2))
        else:
            ops.extend(_refine(ref, hyp, i1, i2, j1, j2))
    return Alignment(ref=ref, hyp=hyp, ops=_coalesce(ops))
