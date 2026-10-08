#!/usr/bin/env python3
"""Timestamp accuracy from the full word alignment (thousands of anchors, not a handful)."""
import json, numpy as np
from parse import load_all, words, BASE
from score import align, word_times, MATCH, SUB

data, notes = load_all()
ref_wt = word_times(data["manual"])
ref = [w for w, _, _ in ref_wt]
ref_t = np.array([t for _, t, _ in ref_wt])

# The manual has two known out-of-order timestamps near the end; restrict drift
# measurement to the monotonic prefix so we compare against a trustworthy clock.
mono_end = len(ref_t)
for i in range(1, len(ref_t)):
    if ref_t[i] < ref_t[i - 1] - 60:
        mono_end = i
        break
print(f"manual reference clock monotonic for first {mono_end}/{len(ref)} words "
      f"(up to t={ref_t[mono_end-1]:.0f}s)")

out = {}
for sys in ["sinribe", "vibe", "claude"]:
    hyp_wt = word_times(data[sys])
    hyp = [w for w, _, _ in hyp_wt]
    hyp_t = np.array([t for _, t, _ in hyp_wt])
    ops = align(ref, hyp)
    d = [hyp_t[b] - ref_t[a] for o, a, b in ops if o == MATCH and a < mono_end]
    a_ = np.abs(np.array(d))
    out[sys] = {
        "n": len(d), "median_signed": float(np.median(d)),
        "mean_abs": float(a_.mean()), "median_abs": float(np.median(a_)),
        "p90_abs": float(np.percentile(a_, 90)),
        "within_1s": float((a_ <= 1).mean()), "within_2s": float((a_ <= 2).mean()),
        "within_5s": float((a_ <= 5).mean()),
        "last_ts": data[sys][-1]["t"],
    }
    print(f"{sys:9s} n={len(d):5d} median={np.median(d):+6.1f}s  mean|Δ|={a_.mean():5.1f}s  "
          f"med|Δ|={np.median(a_):4.1f}s  p90={np.percentile(a_,90):5.1f}s  "
          f"≤1s={(a_<=1).mean():5.1%} ≤2s={(a_<=2).mean():5.1%} ≤5s={(a_<=5).mean():5.1%}")

json.dump(out, open(BASE + "timing.json", "w"), indent=1)
