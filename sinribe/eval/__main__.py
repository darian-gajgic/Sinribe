"""Command line for transcript scoring.

    python -m sinribe.eval REFERENCE HYPOTHESIS        score one transcript
    python -m sinribe.eval --sweep CONFIGS AUDIO REF   decode AUDIO once per config, score each

REFERENCE may be .docx, .odt, .md, .txt, .srt or .vtt. HYPOTHESIS is Sinribe's .md (its
.sinribe.json sidecar is picked up automatically, which is what supplies word timings and
confidences) or any of the same formats.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .normalize import Normalizer
from .reference import load_hypothesis, load_reference
from .score import score


def _score_cmd(args: argparse.Namespace) -> int:
    ref_path, hyp_path = Path(args.reference), Path(args.hypothesis)
    for p in (ref_path, hyp_path):
        if not p.exists():
            print(f"ERROR: no such file: {p}", file=sys.stderr)
            return 2

    nz = Normalizer(drop_fillers=not args.keep_fillers,
                    spell_numbers=not args.no_number_spelling)
    ref = load_reference(ref_path, keep_inaudible=args.keep_inaudible)
    hyp = load_hypothesis(hyp_path)
    if not ref:
        print(f"ERROR: no turns parsed from {ref_path}", file=sys.stderr)
        return 2
    if not hyp:
        print(f"ERROR: no turns parsed from {hyp_path}", file=sys.stderr)
        return 2

    rep = score(ref, hyp, nz, label=hyp_path.name)
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
    else:
        parts = max(t.part for t in ref) + 1
        print(f"reference   {ref_path.name}   {len(ref)} turns"
              + (f", {parts} parts (timestamps restart)" if parts > 1 else ""))
        print(f"hypothesis  {hyp_path.name}   {len(hyp)} turns")
        print()
        print(rep.format(top_confusions=args.confusions))
        if args.errors:
            print()
            print(rep.format_worst(args.errors))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sinribe-eval",
                                 description="Measure transcription accuracy against a reference")
    ap.add_argument("--sweep", metavar="CONFIGS",
                    help="JSON file of named decode configs to compare")
    ap.add_argument("--ceiling", metavar="DIR",
                    help="score every transcript under DIR and report how much a perfect choice "
                         "between them would win — i.e. whether more tuning is worth it")
    ap.add_argument("reference", nargs="?", help="reference transcript, or AUDIO when --sweep")
    ap.add_argument("hypothesis", nargs="?", help="Sinribe output, or REFERENCE when --sweep")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--confusions", type=int, default=15, help="how many confusion pairs to show")
    ap.add_argument("--errors", type=int, default=0, metavar="N",
                    help="also list the N largest disagreements, reference against transcript")
    ap.add_argument("--keep-fillers", action="store_true",
                    help="score hesitation sounds instead of dropping them from both sides")
    ap.add_argument("--keep-inaudible", action="store_true",
                    help="keep the reference's (unverständlich) markers as scorable words")
    ap.add_argument("--no-number-spelling", action="store_true",
                    help="compare digits literally instead of spelling them out")
    ap.add_argument("-o", "--out", help="write sweep results here (JSON)")
    args = ap.parse_args(argv)

    if args.ceiling:
        from .ceiling import run_ceiling
        if not args.reference:
            ap.error("--ceiling needs REFERENCE")
        return run_ceiling(Path(args.ceiling), Path(args.reference),
                           keep_fillers=args.keep_fillers)

    if args.sweep:
        from .sweep import run_sweep
        if not (args.reference and args.hypothesis):
            ap.error("--sweep needs AUDIO and REFERENCE")
        return run_sweep(Path(args.sweep), Path(args.reference), Path(args.hypothesis),
                         out=Path(args.out) if args.out else None,
                         keep_fillers=args.keep_fillers)

    if not (args.reference and args.hypothesis):
        ap.error("need REFERENCE and HYPOTHESIS")
    return _score_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
