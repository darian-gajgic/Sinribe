"""Score a transcript against a reference and say something useful about where it went wrong.

A single WER number tells you a config changed something; it does not tell you what to do next.
The breakdowns here are the ones that actually pointed at fixes on the Feuerläufer interview:

* per speaker — the Bavarian interviewee scored nine points worse than the interviewer, which is
  what identified accent and mic distance rather than the pipeline as the problem;
* over time — the error rate tripling across the hour showed the recording gets harder, not that
  anything drifts;
* by confidence — errors sitting at a median probability of 0.86 is why "just re-decode the
  uncertain parts" was rejected as the main fix;
* confusion pairs — `Fujitsu` losing to *fiuzi* / *jitze* / *future service* every single time is
  what made the hotwords case, and function-word swaps are what made the language-context case.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .align import Alignment, align
from .normalize import Normalizer
from .reference import RefTurn

BIN_SECONDS = 600.0


@dataclass
class Token:
    """One normalised word plus whatever the source knew about it."""
    word: str
    speaker: str | None = None
    start: float | None = None
    prob: float | None = None


@dataclass
class Counts:
    ref: int = 0
    correct: int = 0
    sub: int = 0
    dele: int = 0
    ins: int = 0

    @property
    def wer(self) -> float:
        return (self.sub + self.dele + self.ins) / self.ref if self.ref else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.ref if self.ref else 0.0


@dataclass
class Report:
    total: Counts
    alignment: Alignment
    by_speaker: dict[str, Counts] = field(default_factory=dict)
    by_bin: dict[int, Counts] = field(default_factory=dict)
    speaker_map: dict[str, str] = field(default_factory=dict)
    speaker_correct: int = 0
    speaker_total: int = 0
    confusions: list[tuple[str, str, int]] = field(default_factory=list)
    prob_correct: float | None = None
    prob_wrong: float | None = None
    ortho_words: int = 0
    label: str = ""

    @property
    def speaker_accuracy(self) -> float:
        return self.speaker_correct / self.speaker_total if self.speaker_total else 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "ref_words": self.total.ref,
            "hyp_words": len(self.alignment.hyp),
            "wer": round(self.total.wer, 5),
            "accuracy": round(self.total.accuracy, 5),
            "substitutions": self.total.sub,
            "deletions": self.total.dele,
            "insertions": self.total.ins,
            "orthographic": self.ortho_words,
            "speaker_accuracy": round(self.speaker_accuracy, 5),
            "by_speaker": {k: {"ref": c.ref, "wer": round(c.wer, 5),
                               "accuracy": round(c.accuracy, 5)}
                           for k, c in sorted(self.by_speaker.items())},
            "by_bin": {f"{k * int(BIN_SECONDS // 60)}-{(k + 1) * int(BIN_SECONDS // 60)}min":
                       round(c.wer, 5) for k, c in sorted(self.by_bin.items())},
            "prob_correct": self.prob_correct,
            "prob_wrong": self.prob_wrong,
        }

    def worst(self, limit: int = 20) -> list[tuple[str, str, str, int]]:
        """The biggest disagreements as (kind, reference text, transcript text, size).

        Sorted by how many words are involved, because a ten-word run that went astray tells you
        far more about what to fix than ten separate one-word swaps.
        """
        out = []
        for op in self.alignment.ops:
            if op.kind in ("equal", "ortho"):
                continue
            out.append((op.kind,
                        " ".join(self.alignment.ref[op.i1:op.i2]),
                        " ".join(self.alignment.hyp[op.j1:op.j2]),
                        max(op.n_ref, op.n_hyp)))
        out.sort(key=lambda x: -x[3])
        return out[:limit]

    def format_worst(self, limit: int = 20) -> str:
        label = {"sub": "misheard", "del": "missed", "ins": "invented"}
        lines = [f"  biggest disagreements ({limit} largest):"]
        for kind, ref, hyp, n in self.worst(limit):
            lines.append(f"    [{n:>2}w {label.get(kind, kind):<8}]")
            if ref:
                lines.append(f"       reference:  {ref[:150]}")
            if hyp:
                lines.append(f"       transcript: {hyp[:150]}")
        return "\n".join(lines)

    def format(self, top_confusions: int = 15) -> str:
        t = self.total
        share = (lambda n: f"   ({n / t.ref:.1%} of reference)") if t.ref else (lambda n: "")
        lines = [
            f"reference {t.ref} words   hypothesis {len(self.alignment.hyp)} words",
            "",
            f"  WER                 {t.wer:>7.1%}",
            f"  words correct       {t.accuracy:>7.1%}   ({t.correct})",
            f"  substitutions       {t.sub:>7d}{share(t.sub)}",
            f"  deletions           {t.dele:>7d}{share(t.dele)}",
            f"  insertions          {t.ins:>7d}{share(t.ins)}",
            f"  spelling-only       {self.ortho_words:>7d}   (counted as correct)",
        ]
        if self.speaker_total:
            lines += ["", (f"  speaker attribution {self.speaker_accuracy:>7.1%}   "
                           f"({self.speaker_correct}/{self.speaker_total} matched words)")]
        if self.by_speaker:
            # No insertion column: an invented word belongs to no reference speaker, so it can
            # only be charged to the file as a whole. These are substitutions and deletions.
            lines += ["", "  per speaker (substitutions + deletions):"]
            for name, c in sorted(self.by_speaker.items(), key=lambda kv: -kv[1].ref):
                lines.append(f"    {name:<22} {c.ref:>6} words   "
                             f"accuracy {c.accuracy:>6.1%}   errors {c.wer:>6.1%}")
        if self.by_bin:
            lines += ["", "  over time:"]
            step = int(BIN_SECONDS // 60)
            for k, c in sorted(self.by_bin.items()):
                lines.append(f"    {k * step:>3}-{(k + 1) * step:>3} min          "
                             f"{c.ref:>6} words   WER {c.wer:>6.1%}")
        if self.prob_correct is not None:
            lines += ["", "  recogniser confidence:",
                      f"    on correct words      {self.prob_correct:.3f}",
                      f"    on wrong words        {self.prob_wrong:.3f}"]
        if self.confusions and top_confusions:
            lines += ["", "  most frequent confusions (reference -> transcript):"]
            for ref, hyp, n in self.confusions[:top_confusions]:
                lines.append(f"    {n:>3}x  {ref[:34]:<34} -> {hyp[:34]}")
        return "\n".join(lines)


def tokens(turns: list[RefTurn], nz: Normalizer) -> list[Token]:
    """Flatten turns into normalised tokens, keeping per-word timing where the source has it."""
    out: list[Token] = []
    for t in turns:
        if t.words:
            for w in t.words:
                for piece in nz.words(w.word):
                    out.append(Token(piece, t.speaker, w.start, w.prob))
        else:
            for piece in nz.words(t.text):
                out.append(Token(piece, t.speaker, t.start, None))
    return out


def _best_mapping(pairs: Counter) -> dict[str, str]:
    """Match reference speakers to hypothesis labels, one to one.

    One to one is the whole point. A diarizer that gave up and called the entire recording one
    person would otherwise score a perfect 100 %, because every reference speaker would map to
    that same label — so a hypothesis label may be claimed by only one reference speaker, and
    whoever is left over has all of their words counted as misattributed.

    Few enough speakers to try every assignment is the normal case, and that is exact; the greedy
    fallback only ever runs on recordings with more than seven people in them.
    """
    ref_ids = sorted({r for r, _ in pairs})
    hyp_ids = sorted({h for _, h in pairs})
    if not ref_ids or not hyp_ids:
        return {}

    if len(ref_ids) <= 7 and len(hyp_ids) <= 7:
        from itertools import permutations
        best, best_score = {}, -1
        wide, narrow = (hyp_ids, ref_ids) if len(hyp_ids) >= len(ref_ids) else (ref_ids, hyp_ids)
        flipped = wide is ref_ids
        for pick in permutations(wide, len(narrow)):
            cand = dict(zip(narrow, pick, strict=True))
            if flipped:
                cand = {w: n for n, w in cand.items()}
            got = sum(n for (r, h), n in pairs.items() if cand.get(r) == h)
            if got > best_score:
                best, best_score = cand, got
        return best

    mapping: dict[str, str] = {}
    taken: set[str] = set()
    for (r, h), _ in pairs.most_common():
        if r not in mapping and h not in taken:
            mapping[r] = h
            taken.add(h)
    return mapping


def _speaker_pairs(al: Alignment, ref: list[Token], hyp: list[Token]) -> Counter:
    """Count which hypothesis speaker label sits opposite which reference label."""
    pairs: Counter = Counter()
    for op in al.ops:
        if op.kind not in ("equal", "ortho"):
            continue
        for k in range(op.n_ref):
            r = ref[op.i1 + k]
            # An `ortho` op can pair 2 reference words with 1 hypothesis word; scale the index
            # so every reference word still lands on a real hypothesis word.
            j = op.j1 + (k * op.n_hyp // op.n_ref if op.n_ref else 0)
            h = hyp[min(j, op.j2 - 1)] if op.n_hyp else None
            if r.speaker and h and h.speaker:
                pairs[(r.speaker, h.speaker)] += 1
    return pairs


def score(
    ref_turns: list[RefTurn],
    hyp_turns: list[RefTurn],
    normalizer: Normalizer | None = None,
    label: str = "",
) -> Report:
    """Compare a hypothesis transcript with a reference and summarise the difference."""
    nz = normalizer or Normalizer()
    ref = tokens(ref_turns, nz)
    hyp = tokens(hyp_turns, nz)
    al = align([t.word for t in ref], [t.word for t in hyp])

    total = Counts(ref=len(ref))
    by_speaker: dict[str, Counts] = defaultdict(Counts)
    by_bin: dict[int, Counts] = defaultdict(Counts)
    confusions: Counter = Counter()
    ortho_words = 0

    # Reference words that the hypothesis never produced have no timestamp of their own, so they
    # are booked against the last position we do know — which keeps deletions in the right bin
    # instead of silently vanishing from the time breakdown.
    cursor = 0.0

    for op in al.ops:
        hyp_start = next((hyp[j].start for j in range(op.j1, op.j2)
                          if hyp[j].start is not None), None)
        if hyp_start is not None:
            cursor = hyp_start
        bucket = int(cursor // BIN_SECONDS)

        if op.kind in ("equal", "ortho"):
            total.correct += op.n_ref
            if op.kind == "ortho":
                ortho_words += op.n_ref
        elif op.kind == "sub":
            total.sub += op.n_ref
            confusions[(" ".join(al.ref[op.i1:op.i2]), " ".join(al.hyp[op.j1:op.j2]))] += 1
        elif op.kind == "del":
            total.dele += op.n_ref
        elif op.kind == "ins":
            total.ins += op.n_hyp

        # Insertions belong to no reference speaker, so they are charged to the whole file only.
        for i in range(op.i1, op.i2):
            spk = ref[i].speaker or "?"
            for target in (by_speaker[spk], by_bin[bucket]):
                target.ref += 1
                if op.kind in ("equal", "ortho"):
                    target.correct += 1
                elif op.kind == "sub":
                    target.sub += 1
                elif op.kind == "del":
                    target.dele += 1
        if op.kind == "ins":
            by_bin[bucket].ins += op.n_hyp

    pairs = _speaker_pairs(al, ref, hyp)
    speaker_map = _best_mapping(pairs)
    spk_ok = sum(n for (r, h), n in pairs.items() if speaker_map.get(r) == h)
    spk_all = sum(pairs.values())

    good = [hyp[j].prob for op in al.ops if op.kind in ("equal", "ortho")
            for j in range(op.j1, op.j2) if hyp[j].prob is not None]
    bad = [hyp[j].prob for op in al.ops if op.kind in ("sub", "ins")
           for j in range(op.j1, op.j2) if hyp[j].prob is not None]

    return Report(
        total=total,
        alignment=al,
        by_speaker=dict(by_speaker),
        by_bin=dict(by_bin),
        speaker_map=speaker_map,
        speaker_correct=spk_ok,
        speaker_total=spk_all,
        confusions=[(r, h, n) for (r, h), n in confusions.most_common()],
        prob_correct=round(statistics.mean(good), 4) if good else None,
        prob_wrong=round(statistics.mean(bad), 4) if bad else None,
        ortho_words=ortho_words,
        label=label,
    )
