"""Turn transcript text into the comparable word sequence WER is defined over.

The point of this module is to not charge the recogniser for things a reader would never call a
mistake. On the Feuerläufer interview roughly one substitution run in eight was purely
orthographic — `gernhabt`/`gern habt`, `soft skills`/`softskills`, `erstmal`/`erst mal` — and an
unnormalised score reports those as errors, which then hides whether a real change helped.

Everything here is applied to BOTH sides, so a transformation can never favour one transcript. The
compound join/split case cannot be settled word-by-word (it is a 1:2 or 2:1 alignment, not a
substitution) and is handled in align.py, after the sequences line up.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Folded to the ASCII digraph so "Feuerläufer" and "Feuerlaeufer" — and "weiß" and "weiss" —
# compare equal whichever spelling either transcript chose.
_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
                       "Ä": "ae", "Ö": "oe", "Ü": "ue",
                       "á": "a", "à": "a", "â": "a", "é": "e", "è": "e", "ê": "e",
                       "í": "i", "ì": "i", "ó": "o", "ò": "o", "ô": "o", "ú": "u", "ù": "u",
                       "ç": "c", "ñ": "n"})

# Kept out of the word stream: hesitation sounds that transcripts disagree about recording at all.
# Whisper drops most of them by design; a human writes down the ones they noticed. Neither
# behaviour is an accuracy difference.
#
# Deliberately conservative. "um", "eh", "ah" and "so" are hesitation noises in some transcripts
# and ordinary German words in others ("um fünf Uhr", "das ist eh klar"), and silently deleting a
# real word from both sides would hide errors rather than excuse them. Only sounds with no
# competing lexical reading are listed; both the umlaut and the folded spelling appear because
# normalisation order is not guaranteed at the call site.
FILLERS = frozenset({
    "äh", "ähm", "ähem", "öh", "öhm", "hm", "hmm", "hmmm", "mhm", "mh", "mhmm",
    "aeh", "aehm", "aehem", "oeh", "oehm", "uh", "uhm", "erm",
})

_ONES = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
         "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn",
         "achtzehn", "neunzehn"]
_TENS = {2: "zwanzig", 3: "dreißig", 4: "vierzig", 5: "fünfzig", 6: "sechzig",
         7: "siebzig", 8: "achtzig", 9: "neunzig"}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_TOKEN_NUM = re.compile(r"^\d+$")


def spell_int(n: int) -> str:
    """German cardinal for 0..999_999, written solid the way German spells numbers.

    Applied to both transcripts, so its job is only to be *consistent* — a year spelled
    "eintausendneunhundertzweiundachtzig" rather than "neunzehnhundert..." still makes 1982 on one
    side equal to 1982 on the other.
    """
    if n < 0:
        return "minus" + spell_int(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return (f"{_ONES[ones] if ones != 1 else 'ein'}und" if ones else "") + _TENS[tens]
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = ("ein" if hundreds == 1 else _ONES[hundreds]) + "hundert"
        return head + (spell_int(rest) if rest else "")
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        head = ("ein" if thousands == 1 else spell_int(thousands)) + "tausend"
        return head + (spell_int(rest) if rest else "")
    return str(n)


@dataclass
class Normalizer:
    """Text -> word list. Defaults match the transcript style Sinribe targets.

    `drop_fillers` is what "clean verbatim" means in practice: both sides lose their hesitation
    sounds, so a verbatim model and a tidied human reference can still be compared honestly.
    """
    casefold: bool = True
    fold_umlauts: bool = True
    strip_punctuation: bool = True
    spell_numbers: bool = True
    drop_fillers: bool = True
    extra_fillers: frozenset[str] = field(default_factory=frozenset)

    def words(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFC", text or "")
        if self.casefold:
            text = text.lower()
        if self.fold_umlauts:
            text = text.translate(_FOLD)
        # Hyphens and slashes join words that the other transcript may have written apart
        # ("VR-Bank" / "VR Bank"); splitting both makes the compound rule in align.py resolve it.
        text = re.sub(r"[-–—/]+", " ", text)
        if self.strip_punctuation:
            text = _PUNCT.sub(" ", text)
        out: list[str] = []
        drop = (FILLERS | self.extra_fillers) if self.drop_fillers else frozenset()
        for tok in text.split():
            if self.spell_numbers and _TOKEN_NUM.match(tok):
                tok = spell_int(int(tok))
                if self.fold_umlauts:
                    tok = tok.translate(_FOLD)
            if tok in drop:
                continue
            if tok:
                out.append(tok)
        return out


def norm_words(text: str, **kw) -> list[str]:
    """Convenience wrapper for a one-off normalisation."""
    return Normalizer(**kw).words(text)
