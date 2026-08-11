#!/usr/bin/env python3
"""Classify each alignment error as cosmetic (orthographic/function-word/verbatim-style)
or substantive (changes the meaning a reader would take away)."""
import re

# ---- Kölner Phonetik: German phonetic code. Same code => words sound (near) identical.
def koelner(word):
    w = word.upper()
    w = (w.replace("Ä", "A").replace("Ö", "O").replace("Ü", "U").replace("ß", "SS"))
    w = re.sub(r"[^A-Z]", "", w)
    if not w:
        return ""
    code = []
    n = len(w)
    for i, ch in enumerate(w):
        nxt = w[i + 1] if i + 1 < n else ""
        prv = w[i - 1] if i > 0 else ""
        if ch in "AEIOUYJ":
            c = "0"
        elif ch == "H":
            c = ""
        elif ch == "B" or (ch == "P" and nxt != "H"):
            c = "1"
        elif ch in "DT" and nxt not in "CSZ":
            c = "2"
        elif ch in "FVW" or (ch == "P" and nxt == "H"):
            c = "3"
        elif ch in "GKQ":
            c = "4"
        elif ch == "C":
            if i == 0:
                c = "4" if nxt in "AHKOQUX" else "8"
            else:
                c = "8" if prv in "SZ" else ("4" if nxt in "AHKOQUX" else "8")
        elif ch == "X":
            c = "48" if prv not in "CKQ" else "8"
        elif ch == "L":
            c = "5"
        elif ch in "MN":
            c = "6"
        elif ch == "R":
            c = "7"
        elif ch in "SZ":
            c = "8"
        elif ch in "DT":
            c = "8"
        else:
            c = ""
        code.append(c)
    s = "".join(code)
    out = []
    for ch in s:                      # collapse repeats
        if not out or out[-1] != ch:
            out.append(ch)
    s = "".join(out)
    return s[0] + s[1:].replace("0", "") if s else ""


GERMAN_FUNCTION = set("""
der die das den dem des ein eine einen einem einer eines
ich du er sie es wir ihr man mich dich sich uns euch mir dir ihm ihn ihnen
mein dein sein unser euer ihre ihren ihrem seiner meiner
und oder aber denn sondern doch also ja nein nicht nur noch schon auch mal
so wie was wer wo wann warum dass wenn weil ob als
ist sind war waren bin bist sein hat habe haben hatte hatten hab
wird werden wurde wurden kann können konnte muss müssen möchte will wollen
in an auf aus bei mit nach von vor zu zur zum im am für über unter durch um
da dann dort hier jetzt immer wieder sehr ganz mehr viel etwas nichts
eben halt eigentlich vielleicht natürlich genau okay ähm äh hm mhm
ne nen ja
""".split())

FILLER = set("ähm äh hm mhm ja ne genau also halt eben okay mhm".split())

# spoken-form vs written-form pairs that represent the same utterance
COLLOQUIAL = {
    "hab": "habe", "hat": "hat", "sag": "sage", "geh": "gehe", "steh": "stehe",
    "gern": "gerne", "erstmal": "erstmals", "is": "ist", "nix": "nichts",
    "nen": "einen", "ne": "eine", "n": "ein", "was": "etwas", "mal": "einmal",
    "runter": "herunter", "rauf": "herauf", "rein": "herein", "raus": "heraus",
    "drauf": "darauf", "drin": "darin", "dran": "daran",
}


INFL = ("en", "em", "er", "es", "et", "st", "te", "n", "s", "t", "e")


def same_lemma_ish(a, b):
    """Same word, different inflection: identical stem plus only inflectional tails."""
    if a == b:
        return True
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    if len(lo) < 4 or not hi.startswith(lo[:max(4, len(lo) - 2)]):
        return False
    # the differing tails on both sides must be inflectional endings only
    common = 0
    while common < len(lo) and common < len(hi) and lo[common] == hi[common]:
        common += 1
    if common < 4:
        return False
    ta, tb = lo[common:], hi[common:]
    ok = lambda t: t == "" or t in INFL
    return ok(ta) and ok(tb)


def classify_sub(ref_w, hyp_w):
    """Return 'cosmetic' or 'substantive' for a substitution."""
    if ref_w == hyp_w:
        return "cosmetic"
    if COLLOQUIAL.get(ref_w) == hyp_w or COLLOQUIAL.get(hyp_w) == ref_w:
        return "cosmetic"
    if same_lemma_ish(ref_w, hyp_w):
        return "cosmetic"                        # inflection only
    if koelner(ref_w) and koelner(ref_w) == koelner(hyp_w):
        return "cosmetic"                        # sounds the same, spelled differently
    if ref_w in GERMAN_FUNCTION and hyp_w in GERMAN_FUNCTION:
        return "cosmetic"                        # function-word swap, meaning preserved
    return "substantive"


def classify_indel(w):
    """Insertion/deletion of a single word."""
    if w in FILLER:
        return "cosmetic"                        # verbatim-vs-cleaned difference
    if w in GERMAN_FUNCTION:
        return "cosmetic"
    return "substantive"


if __name__ == "__main__":
    for a, b in [("gern", "gerne"), ("Feuerläufer", "Feuerleufer"), ("Achtsamkeit", "Achtsamkeit"),
                 ("Trance", "Branche"), ("Herz", "Schmerz"), ("das", "es"), ("Mensch", "Menschen")]:
        print(f"{a:14s} {b:14s} {koelner(a):8s} {koelner(b):8s} -> {classify_sub(a.lower(), b.lower())}")
