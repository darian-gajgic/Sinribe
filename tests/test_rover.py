"""Voting across decodes: what the merge keeps, and what the agreement figure actually counts."""

from __future__ import annotations

from sinribe import rover
from sinribe.merge import Word


def words(*texts: str, prob: float = 0.9) -> list[Word]:
    """A pass, one word every second, so the pivot's timings are predictable."""
    return [Word(start=float(i), end=float(i) + 0.9, word=t, prob=prob)
            for i, t in enumerate(texts)]


def text(ws: list[Word]) -> list[str]:
    return [w.word for w in ws]


class TestCombine:
    def test_a_single_pass_is_returned_unchanged(self):
        p = words("eins", "zwei", "drei")
        assert text(rover.combine([p])) == ["eins", "zwei", "drei"]

    def test_the_majority_overrules_the_pivot(self):
        p = [words("eins", "zwo", "drei"),
             words("eins", "zwei", "drei"),
             words("eins", "zwei", "drei")]
        assert text(rover.combine(p)) == ["eins", "zwei", "drei"]

    def test_the_pivot_keeps_its_timings_when_the_text_is_overruled(self):
        p = [words("eins", "zwo"), words("eins", "zwei"), words("eins", "zwei")]
        merged = rover.combine(p)
        assert [(w.start, w.end) for w in merged] == [(w.start, w.end) for w in p[0]]

    def test_a_word_only_the_pivot_heard_is_voted_away(self):
        # Two passes saying nothing here outweigh one pass saying something (DELETION_WEIGHT is
        # deliberately below a real word's confidence, but two of them clear it).
        p = [words("eins", "aeh", "zwei"), words("eins", "zwei"), words("eins", "zwei")]
        assert text(rover.combine(p)) == ["eins", "zwei"]

    def test_case_and_punctuation_do_not_split_a_majority(self):
        p = [words("Eins,", "zwei"), words("eins", "zwei"), words("eins.", "zwei")]
        assert len(rover.combine(p)) == 2


class TestAgreement:
    """The figure printed in the transcript header as "N% unanimous".

    It used to be computed by zipping the merged output against the pivot, which lines up only
    until the first position the vote drops — after that every comparison was off by one and the
    number collapsed. A real three-pass run that agreed on 85 % of its words reported 2 %, which
    reads as evidence that the voting is broken rather than as a bug in the diagnostic.
    """

    def test_identical_passes_are_fully_unanimous(self):
        p = words("eins", "zwei", "drei")
        assert rover.agreement([p, list(p), list(p)]) == 1.0

    def test_a_single_pass_makes_no_claim(self):
        assert rover.agreement([words("eins", "zwei")]) == 1.0

    def test_one_disagreement_in_four_positions(self):
        p = [words("eins", "zwei", "drei", "vier"),
             words("eins", "zwo", "drei", "vier"),
             words("eins", "zwei", "drei", "vier")]
        assert rover.agreement(p) == 0.75

    def test_a_dropped_position_does_not_shift_the_comparison(self):
        # The regression. The vote removes "aeh", so merged is shorter than the pivot; every
        # later position must still be compared with its own ballots, not with its neighbour's.
        p = [words("eins", "aeh", "zwei", "drei", "vier"),
             words("eins", "zwei", "drei", "vier"),
             words("eins", "zwei", "drei", "vier")]
        merged = rover.combine(p)
        assert text(merged) == ["eins", "zwei", "drei", "vier"]
        # Four of the pivot's five positions carry identical ballots; only "aeh" is contested.
        assert rover.agreement(p) == 0.8

    def test_passes_that_share_nothing_agree_on_nothing(self):
        p = [words("eins", "zwei"), words("neun", "acht"), words("neun", "acht")]
        assert rover.agreement(p) == 0.0
