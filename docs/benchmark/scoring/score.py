#!/usr/bin/env python3
"""Exact word-level Levenshtein alignment + error analysis against the manual reference."""
import numpy as np, json, re
from collections import Counter
from parse import load_all, words, BASE

MATCH, SUB, DEL, INS = 0, 1, 2, 3


def align(ref, hyp):
    """Exact Levenshtein alignment. Returns list of (op, ref_idx|None, hyp_idx|None)."""
    n, m = len(ref), len(hyp)
    # index words to ints for fast comparison
    vocab = {}
    R = np.array([vocab.setdefault(w, len(vocab)) for w in ref], dtype=np.int32)
    H = np.array([vocab.setdefault(w, len(vocab)) for w in hyp], dtype=np.int32)

    prev = np.arange(m + 1, dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.uint8)
    bt[0, 1:] = INS
    cur = np.empty(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        cur[0] = i
        bt[i, 0] = DEL
        eq = (H == R[i - 1])
        diag = prev[:-1] + (~eq)                     # substitution / match
        up = prev[1:] + 1                            # deletion (ref word unmatched)
        # left (insertion) depends on cur[j-1] -> sequential; do a fast scan
        best = np.minimum(diag, up)
        choice = np.where(diag <= up, np.where(eq, MATCH, SUB), DEL).astype(np.uint8)
        # resolve insertions with a running minimum
        c = cur
        b = bt[i]
        best_l = best.tolist(); choice_l = choice.tolist()
        run = i
        for j in range(1, m + 1):
            v = best_l[j - 1]; ch = choice_l[j - 1]
            if run + 1 < v:
                v = run + 1; ch = INS
            c[j] = v; b[j] = ch; run = v
        prev, cur = cur, prev
    # backtrace
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        op = bt[i, j]
        if op == MATCH or op == SUB:
            ops.append((op, i - 1, j - 1)); i -= 1; j -= 1
        elif op == DEL:
            ops.append((DEL, i - 1, None)); i -= 1
        else:
            ops.append((INS, None, j - 1)); j -= 1
    ops.reverse()
    return ops


def word_times(segs):
    """Approximate a timestamp for every word by spreading each segment's words
    across the interval to the next segment start."""
    out = []
    for k, s in enumerate(segs):
        w = words(s["text"])
        if not w:
            continue
        t0 = s["t"]
        t1 = segs[k + 1]["t"] if k + 1 < len(segs) else t0 + max(2, len(w) * 0.4)
        if t1 <= t0:
            t1 = t0 + max(2, len(w) * 0.4)
        step = (t1 - t0) / len(w)
        for x in range(len(w)):
            out.append((w[x], t0 + x * step, s.get("spk")))
    return out


def cer(ref_s, hyp_s):
    a, b = list(ref_s), list(hyp_s)
    prev = np.arange(len(b) + 1, dtype=np.int32)
    A = np.array([ord(c) for c in a]); B = np.array([ord(c) for c in b])
    for i in range(1, len(a) + 1):
        cur = np.empty(len(b) + 1, dtype=np.int32); cur[0] = i
        diag = prev[:-1] + (B != A[i - 1]); up = prev[1:] + 1
        best = np.minimum(diag, up).tolist()
        run = i
        for j in range(1, len(b) + 1):
            v = best[j - 1]
            if run + 1 < v: v = run + 1
            cur[j] = v; run = v
        prev = cur
    return prev[-1] / max(1, len(a))


if __name__ == "__main__":
    data, notes = load_all()
    ref_wt = word_times(data["manual"])
    ref = [w for w, _, _ in ref_wt]
    results = {}
    for sys_name in ["sinribe", "vibe", "claude"]:
        if sys_name not in data:
            continue
        hyp_wt = word_times(data[sys_name])
        hyp = [w for w, _, _ in hyp_wt]
        ops = align(ref, hyp)
        c = Counter(op for op, _, _ in ops)
        S, D, I, M = c[SUB], c[DEL], c[INS], c[MATCH]
        wer = (S + D + I) / len(ref)
        results[sys_name] = dict(N_ref=len(ref), N_hyp=len(hyp), M=M, S=S, D=D, I=I,
                                 wer=wer, sub_rate=S / len(ref), del_rate=D / len(ref),
                                 ins_rate=I / len(ref), acc=M / len(ref))
        json.dump([[int(o), a, b] for o, a, b in ops],
                  open(f"{BASE}align_{sys_name}.json", "w"))
        print(f"{sys_name:8s} WER={wer:6.2%}  acc={M/len(ref):6.2%}  "
              f"S={S:5d} D={D:5d} I={I:5d}  hyp_words={len(hyp)}")
    json.dump(results, open(BASE + "wer.json", "w"), indent=1)
