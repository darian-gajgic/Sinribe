#!/usr/bin/env python3
"""Full benchmark analysis: WER tiers, hallucination/omission runs, diarization,
timestamp drift, coverage over time, domain-term accuracy."""
import json, numpy as np
from collections import Counter, defaultdict
from parse import load_all, words, BASE
from score import align, word_times, SUB, DEL, INS, MATCH
from classify import classify_sub, classify_indel

SYSTEMS = ["sinribe", "vibe", "claude"]
LABEL = {"sinribe": "Sinribe", "vibe": "Vibe", "claude": "Stock large-v3"}

data, notes = load_all()
ref_wt = word_times(data["manual"])
ref = [w for w, _, _ in ref_wt]
ref_t = [t for _, t, _ in ref_wt]
ref_spk = [s for _, _, s in ref_wt]

report = {"n_ref_words": len(ref), "manual_notes": notes, "systems": {}}

for sys in SYSTEMS:
    hyp_wt = word_times(data[sys])
    hyp = [w for w, _, _ in hyp_wt]
    hyp_spk = [s for _, _, s in hyp_wt]
    ops = align(ref, hyp)

    c = Counter(o for o, _, _ in ops)
    S, D, I, M = c[SUB], c[DEL], c[INS], c[MATCH]
    N = len(ref)

    # ---- error tiers ----
    tiers = Counter()
    sub_examples, major_list = [], []
    for o, a, b in ops:
        if o == SUB:
            t = classify_sub(ref[a], hyp[b])
            tiers["sub_" + t] += 1
            if t == "substantive":
                major_list.append(("SUB", ref_t[a], ref[a], hyp[b]))
        elif o == DEL:
            t = classify_indel(ref[a])
            tiers["del_" + t] += 1
            if t == "substantive":
                major_list.append(("DEL", ref_t[a], ref[a], ""))
        elif o == INS:
            t = classify_indel(hyp[b])
            tiers["ins_" + t] += 1
            if t == "substantive":
                major_list.append(("INS", None, "", hyp[b]))

    major = tiers["sub_substantive"] + tiers["del_substantive"] + tiers["ins_substantive"]
    minor = (S + D + I) - major

    # ---- runs of consecutive insertions (hallucination) / deletions (omission) ----
    def runs(target_op):
        out, cur = [], []
        for o, a, b in ops:
            if o == target_op:
                cur.append((a, b))
            else:
                if len(cur) >= 8:
                    out.append(cur)
                cur = []
        if len(cur) >= 8:
            out.append(cur)
        return out

    ins_runs = runs(INS)
    del_runs = runs(DEL)
    ins_run_detail = [{"len": len(r),
                       "text": " ".join(hyp[b] for _, b in r)[:400]} for r in ins_runs]
    del_run_detail = [{"len": len(r), "t": ref_t[r[0][0]],
                       "text": " ".join(ref[a] for a, _ in r)[:400]} for r in del_runs]

    # ---- accuracy over time (2-minute buckets, by reference-word timestamp) ----
    BUCKET = 120
    buck_err, buck_tot = defaultdict(int), defaultdict(int)
    for o, a, b in ops:
        if a is None:
            continue
        k = int(ref_t[a] // BUCKET)
        buck_tot[k] += 1
        if o != MATCH:
            buck_err[k] += 1
    timeline = [{"min": k * BUCKET / 60,
                 "acc": 1 - buck_err[k] / buck_tot[k]} for k in sorted(buck_tot) if buck_tot[k] > 5]

    report["systems"][sys] = {
        "label": LABEL[sys], "n_hyp_words": len(hyp),
        "M": M, "S": S, "D": D, "I": I,
        "wer": (S + D + I) / N, "accuracy": M / N,
        "major_errors": major, "minor_errors": minor,
        "major_rate": major / N, "minor_rate": minor / N,
        "tiers": dict(tiers),
        "n_segments": len(data[sys]),
        "ins_runs": ins_run_detail, "del_runs": del_run_detail,
        "timeline": timeline,
        "major_examples": [m for m in major_list if m[0] == "SUB"][:40],
    }
    print(f"{LABEL[sys]:20s} WER={(S+D+I)/N:6.2%}  major={major/N:6.2%} ({major})  "
          f"minor={minor/N:6.2%}  halluc_runs={len(ins_runs)}  omit_runs={len(del_runs)}")

# ---------- diarization (Sinribe only system with speakers) ----------
hyp_wt = word_times(data["sinribe"])
hyp_spk = [s for _, _, s in hyp_wt]
ops = json.load(open(BASE + "align_sinribe.json")) if False else align(ref, [w for w, _, _ in hyp_wt])
pairs = [(ref_spk[a], hyp_spk[b]) for o, a, b in ops if o in (MATCH, SUB)]
cm = Counter(pairs)
# best mapping of Person N -> manual role
best, best_score = None, -1
import itertools
persons = sorted({h for _, h in pairs})
roles = sorted({r for r, _ in pairs})
for perm in itertools.permutations(roles, len(persons)):
    mp = dict(zip(persons, perm))
    sc = sum(n for (r, h), n in cm.items() if mp.get(h) == r)
    if sc > best_score:
        best, best_score = mp, sc
report["diarization"] = {
    "mapping": best,
    "aligned_words": len(pairs),
    "correct": best_score,
    "accuracy": best_score / len(pairs),
    "confusion": {f"{r} | {h}": n for (r, h), n in cm.most_common()},
}
print(f"\nSinribe diarization: {best_score}/{len(pairs)} = {best_score/len(pairs):.2%}  mapping={best}")

# ---------- timestamp drift ----------
def drift(sys):
    """Match hypothesis segments to manual segments by text anchor, compare timestamps."""
    segs = data[sys]
    ds = []
    for ms in data["manual"]:
        mw = words(ms["text"])
        if len(mw) < 6:
            continue
        key = " ".join(mw[:6])
        for hs in segs:
            if key in " ".join(words(hs["text"])):
                ds.append(hs["t"] - ms["t"])
                break
    return ds

for sys in SYSTEMS:
    d = drift(sys)
    if d:
        a = np.array(d)
        report["systems"][sys]["timestamp"] = {
            "n_anchors": len(a), "median_offset_s": float(np.median(a)),
            "mean_abs_s": float(np.mean(np.abs(a))),
            "within_2s": float(np.mean(np.abs(a) <= 2)),
            "within_5s": float(np.mean(np.abs(a) <= 5)),
        }
        print(f"{LABEL[sys]:20s} timestamps: n={len(a)} median={np.median(a):+.1f}s "
              f"mean|Δ|={np.mean(np.abs(a)):.1f}s  ≤2s={np.mean(np.abs(a)<=2):.0%}")

# ---------- domain terms ----------
TERMS = ["achtsamkeit", "achtsam", "meditation", "meditativ", "coaching", "coach",
         "führungskräfte", "mitarbeiter", "atmung", "nasenatmung", "bauchatmung",
         "feuerlauf", "feuerläufer", "management", "unternehmen", "resilienz",
         "burnout", "spirituell", "trance", "herzschlag", "atem"]
term_tbl = {}
ref_join = " ".join(ref)
for t in TERMS:
    row = {"manual": ref_join.count(t)}
    for sys in SYSTEMS:
        row[sys] = " ".join(w for w, _, _ in word_times(data[sys])).count(t)
    if row["manual"] or any(row[s] for s in SYSTEMS):
        term_tbl[t] = row
report["terms"] = term_tbl

json.dump(report, open(BASE + "report.json", "w"), ensure_ascii=False, indent=1)
print("\nwrote report.json")
