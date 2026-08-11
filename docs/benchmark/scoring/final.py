#!/usr/bin/env python3
"""Consolidated benchmark numbers for the report."""
import json, numpy as np, itertools
from collections import Counter
from parse import load_all, words, BASE
from score import align, word_times, MATCH, SUB, DEL, INS

data, notes = load_all()
ref_wt = word_times(data["manual"])
ref = [w for w, _, _ in ref_wt]
ref_spk = [s for _, _, s in ref_wt]
out = {}

# ---- diarization with a fair majority baseline + turn detection ----
hyp_wt = word_times(data["sinribe"])
ops = align(ref, [w for w, _, _ in hyp_wt])
hyp_spk = [s for _, _, s in hyp_wt]
pairs = [(ref_spk[a], hyp_spk[b]) for o, a, b in ops if o in (MATCH, SUB)]
cm = Counter(pairs)
persons = sorted({h for _, h in pairs}); roles = sorted({r for r, _ in pairs})
best, bs = None, -1
for perm in itertools.permutations(roles, len(persons)):
    mp = dict(zip(persons, perm))
    sc = sum(n for (r, h), n in cm.items() if mp.get(h) == r)
    if sc > bs: best, bs = mp, sc
maj = Counter(r for r, _ in pairs).most_common(1)[0]
# per-role recall
per_role = {}
for role in roles:
    tot = sum(n for (r, _), n in cm.items() if r == role)
    cor = sum(n for (r, h), n in cm.items() if r == role and best.get(h) == role)
    per_role[role] = {"words": tot, "correct": cor, "recall": cor / tot}
# turn-change detection
ref_changes = sum(1 for i in range(1, len(ref_spk)) if ref_spk[i] != ref_spk[i - 1])
al = {a: b for o, a, b in ops if o in (MATCH, SUB)}
hit = tot_ch = 0
for i in range(1, len(ref_spk)):
    if ref_spk[i] != ref_spk[i - 1] and i in al and (i - 1) in al:
        tot_ch += 1
        if hyp_spk[al[i]] != hyp_spk[al[i - 1]]:
            hit += 1
out["diarization"] = {
    "mapping": best, "aligned_words": len(pairs), "accuracy": bs / len(pairs),
    "majority_baseline": maj[1] / len(pairs), "majority_role": maj[0],
    "per_role": per_role,
    "turn_changes_ref": ref_changes, "turn_changes_evaluable": tot_ch,
    "turn_changes_detected": hit, "turn_recall": hit / max(1, tot_ch),
}
print(f"Diarization  acc={bs/len(pairs):.2%}  (majority baseline {maj[1]/len(pairs):.2%})")
for r, v in per_role.items():
    print(f"   {r:18s} recall={v['recall']:.2%}  ({v['correct']}/{v['words']} words)")
print(f"   turn changes detected {hit}/{tot_ch} = {hit/max(1,tot_ch):.1%}")

# ---- inter-system agreement (pairwise WER, symmetric-ish) ----
streams = {s: [w for w, _, _ in word_times(data[s])] for s in ["sinribe", "vibe", "claude", "manual"]}
agree = {}
for a, b in itertools.combinations(["manual", "sinribe", "vibe", "claude"], 2):
    o = align(streams[a], streams[b])
    c = Counter(x for x, _, _ in o)
    agree[f"{a}~{b}"] = 1 - (c[SUB] + c[DEL] + c[INS]) / len(streams[a])
    print(f"agreement {a:8s} ~ {b:8s} = {agree[f'{a}~{b}']:.2%}")
out["agreement"] = agree

# ---- segmentation / readability ----
for s in ["manual", "sinribe", "vibe", "claude"]:
    segs = data[s]
    wl = [len(words(x["text"])) for x in segs if words(x["text"])]
    out.setdefault("segmentation", {})[s] = {
        "n_segments": len(segs), "median_words": float(np.median(wl)),
        "mean_words": float(np.mean(wl)),
        "pct_under_5_words": float(np.mean(np.array(wl) < 5)),
        "has_speakers": bool(segs[0].get("spk")),
    }
    print(f"{s:9s} segments={len(segs):5d} median={np.median(wl):5.1f} words  "
          f"<5w={np.mean(np.array(wl)<5):5.1%}  speakers={bool(segs[0].get('spk'))}")

json.dump(out, open(BASE + "final.json", "w"), ensure_ascii=False, indent=1)
