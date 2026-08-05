"""Unit tests for the word -> speaker alignment, which is where transcript accuracy is won."""

from __future__ import annotations

import pytest

from sinribe.merge import (
    DiarTurn, SentenceScore, Word, assign_words, group_turns, label_speakers, merge,
    sentence_spans, smooth_flicker, vote_sentences,
)


def W(start, end, word="x"):
    return Word(start=start, end=end, word=word)


TURNS = [DiarTurn(0.0, 5.0, "SPEAKER_00"), DiarTurn(5.0, 10.0, "SPEAKER_01")]


class TestAssignWords:
    def test_word_inside_turn(self):
        w = [W(1.0, 2.0)]
        assign_words(w, TURNS)
        assert w[0].speaker == "SPEAKER_00"

    def test_word_in_second_turn(self):
        w = [W(6.0, 7.0)]
        assign_words(w, TURNS)
        assert w[0].speaker == "SPEAKER_01"

    def test_straddling_word_goes_to_max_overlap(self):
        # 4.6-5.1: 0.4s in SPEAKER_00, 0.1s in SPEAKER_01 -> SPEAKER_00 wins.
        w = [W(4.6, 5.1)]
        assign_words(w, TURNS)
        assert w[0].speaker == "SPEAKER_00"
        # 4.9-5.6: 0.1s vs 0.6s -> SPEAKER_01 wins.
        w = [W(4.9, 5.6)]
        assign_words(w, TURNS)
        assert w[0].speaker == "SPEAKER_01"

    def test_orphan_word_snaps_to_nearest_within_window(self):
        turns = [DiarTurn(0.0, 5.0, "A"), DiarTurn(9.0, 12.0, "B")]
        w = [W(8.0, 8.4)]  # in the gap, 0.6s from B, 3.0s from A
        assign_words(w, turns, orphan_window_s=2.0)
        assert w[0].speaker == "B"

    def test_orphan_beyond_window_inherits_previous(self):
        turns = [DiarTurn(0.0, 5.0, "A"), DiarTurn(30.0, 40.0, "B")]
        w = [W(1.0, 2.0), W(10.0, 11.0)]
        assign_words(w, turns, orphan_window_s=2.0)
        assert w[1].speaker == "A"

    def test_leading_words_backfill_from_first_labelled(self):
        turns = [DiarTurn(20.0, 30.0, "A")]
        w = [W(0.0, 1.0), W(21.0, 22.0)]
        assign_words(w, turns, orphan_window_s=1.0)
        assert w[0].speaker == "A"

    def test_no_turns_leaves_words_unlabelled(self):
        w = [W(1.0, 2.0)]
        assign_words(w, [])
        assert w[0].speaker is None

    def test_overlapping_turns_pick_dominant(self):
        turns = [DiarTurn(0.0, 6.0, "A"), DiarTurn(5.0, 10.0, "B")]
        assert_word = W(5.2, 5.9)  # 0.7 in A, 0.7 in B -> deterministic, must not crash
        assign_words([assert_word], turns)
        assert assert_word.speaker in {"A", "B"}


class TestSmoothFlicker:
    def test_absorbs_short_run_between_same_speaker(self):
        w = [W(i, i + 0.5) for i in range(7)]
        for i, sp in enumerate(["A", "A", "A", "B", "A", "A", "A"]):
            w[i].speaker = sp
        smooth_flicker(w, min_run=3)
        assert [x.speaker for x in w] == ["A"] * 7

    def test_keeps_long_runs(self):
        w = [W(i, i + 0.5) for i in range(9)]
        for i, sp in enumerate(["A", "A", "A", "B", "B", "B", "A", "A", "A"]):
            w[i].speaker = sp
        smooth_flicker(w, min_run=3)
        assert [x.speaker for x in w] == ["A", "A", "A", "B", "B", "B", "A", "A", "A"]

    def test_does_not_absorb_between_different_speakers(self):
        # A A A | B | C C C -- B is short but the flanks disagree, so it is genuine.
        w = [W(i, i + 0.5) for i in range(7)]
        for i, sp in enumerate(["A", "A", "A", "B", "C", "C", "C"]):
            w[i].speaker = sp
        smooth_flicker(w, min_run=3)
        assert w[3].speaker == "B"

    def test_min_run_one_is_a_noop(self):
        w = [W(i, i + 0.5) for i in range(3)]
        for i, sp in enumerate(["A", "B", "A"]):
            w[i].speaker = sp
        smooth_flicker(w, min_run=1)
        assert [x.speaker for x in w] == ["A", "B", "A"]


class TestGroupTurns:
    def _words(self, spec):
        out = []
        for start, end, sp, text in spec:
            w = W(start, end, text)
            w.speaker = sp
            out.append(w)
        return out

    def test_splits_on_speaker_change(self):
        w = self._words([(0, 1, "A", "Hello."), (1.1, 2, "B", "Hi.")])
        turns = group_turns(w)
        assert len(turns) == 2
        assert turns[0].raw == "A" and turns[1].raw == "B"

    def test_splits_on_long_gap(self):
        w = self._words([(0, 1, "A", "Hello."), (5.0, 6.0, "A", "Again.")])
        turns = group_turns(w, turn_gap_s=1.5)
        assert len(turns) == 2

    def test_keeps_short_gap_together(self):
        w = self._words([(0, 1, "A", "Hello"), (1.2, 2.0, "A", "there.")])
        turns = group_turns(w, turn_gap_s=1.5)
        assert len(turns) == 1
        assert turns[0].text == "Hello there."

    def test_splits_on_max_chars(self):
        w = self._words([(i * 0.2, i * 0.2 + 0.1, "A", "word") for i in range(60)])
        turns = group_turns(w, turn_gap_s=10.0, max_turn_chars=60)
        assert len(turns) > 1

    def test_empty_words_produce_no_turns(self):
        assert group_turns([]) == []


class TestLabelSpeakers:
    def test_numbered_by_first_appearance(self):
        w = []
        for start, sp in [(0, "SPEAKER_07"), (2, "SPEAKER_02"), (4, "SPEAKER_07")]:
            x = W(start, start + 1, "hi.")
            x.speaker = sp
            w.append(x)
        turns = group_turns(w, turn_gap_s=0.5)
        turns, names = label_speakers(turns)
        assert turns[0].speaker == "Person 1"   # SPEAKER_07 spoke first
        assert turns[1].speaker == "Person 2"
        assert turns[2].speaker == "Person 1"
        assert names == ["Person 1", "Person 2"]


class TestMergeEndToEnd:
    def test_full_merge(self):
        words = [W(0.5, 1.0, "Hello"), W(1.1, 1.6, "there."),
                 W(6.0, 6.5, "Hi"), W(6.6, 7.0, "back.")]
        turns, stats = merge(words, TURNS)
        assert [t.speaker for t in turns] == ["Person 1", "Person 2"]
        assert turns[0].text == "Hello there."
        assert set(stats) == {"Person 1", "Person 2"}
        assert pytest.approx(sum(s["share"] for s in stats.values()), abs=1e-6) == 1.0

    def test_no_diarization_still_produces_transcript(self):
        words = [W(0.0, 1.0, "Hello"), W(1.1, 2.0, "world.")]
        turns, stats = merge(words, [])
        assert len(turns) == 1
        assert turns[0].speaker == "Person 1"
        assert turns[0].text == "Hello world."

    def test_empty_input(self):
        assert merge([], TURNS) == ([], {})

    def test_blank_words_are_dropped(self):
        words = [W(0.0, 1.0, "  "), W(1.0, 2.0, "Real.")]
        turns, _ = merge(words, TURNS)
        assert turns[0].text == "Real."


class TestSentenceSpans:
    def test_splits_on_terminal_punctuation(self):
        words = [W(0, 1, "Hallo"), W(1, 2, "Welt."), W(2, 3, "Wie"), W(3, 4, "geht's?")]
        assert sentence_spans(words) == [[0, 1], [2, 3]]

    def test_abbreviation_does_not_split(self):
        words = [W(0, 1, "Frag"), W(1, 2, "Dr."), W(2, 3, "Meier."), W(3, 4, "Ja.")]
        assert sentence_spans(words) == [[0, 1, 2], [3]]

    def test_trailing_fragment_is_kept(self):
        words = [W(0, 1, "Ende."), W(1, 2, "Kein"), W(2, 3, "Punkt")]
        assert sentence_spans(words) == [[0], [1, 2]]

    def test_no_words_no_spans(self):
        assert sentence_spans([]) == []


class TestSentenceAtomicVoting:
    """A diarization boundary inside a sentence is the diarizer being late, not two people
    splitting a clause — the whole sentence must land on one speaker."""

    def test_sentence_is_not_split_by_a_mid_sentence_boundary(self):
        # One sentence spanning the 5.0 s boundary: 3 s of it is SPEAKER_00, 1 s SPEAKER_01.
        words = [W(2.0, 5.0, "Ein"), W(5.0, 6.0, "Satz.")]
        turns, _ = merge(words, TURNS)
        assert len(turns) == 1
        assert turns[0].text == "Ein Satz."

    def test_longest_overlap_wins_the_sentence(self):
        words = [W(4.5, 5.0, "Kurz"), W(5.0, 9.0, "lang.")]
        assert merge(words, TURNS)[0][0].raw == "SPEAKER_01"

    def test_word_level_mode_still_splits(self):
        words = [W(2.0, 5.0, "Ein"), W(5.0, 6.0, "Satz.")]
        turns, _ = merge(words, TURNS, sentence_atomic=False, flicker_min_words=0)
        assert len(turns) == 2


class TestVoicePrintOverride:
    """The refine stage overrules diarization only where the acoustic match is decisive."""

    def _words(self):
        # Both words sit squarely inside SPEAKER_00's turn, so diarization alone says SPEAKER_00.
        return [W(1.0, 2.0, "Das"), W(2.0, 3.0, "stimmt.")]

    def test_decisive_score_overrides_diarization(self):
        words = self._words()
        assign_words(words, TURNS)
        locked = vote_sentences(words, sentence_spans(words),
                                scores=[SentenceScore("SPEAKER_01", 0.5)],
                                margin=0.15, min_seconds=0.6)
        assert [w.speaker for w in words] == ["SPEAKER_01", "SPEAKER_01"]
        assert locked == {0, 1}

    def test_weak_margin_leaves_diarization_alone(self):
        words = self._words()
        assign_words(words, TURNS)
        locked = vote_sentences(words, sentence_spans(words),
                                scores=[SentenceScore("SPEAKER_01", 0.02)],
                                margin=0.15, min_seconds=0.6)
        assert [w.speaker for w in words] == ["SPEAKER_00", "SPEAKER_00"]
        assert locked == set()

    def test_short_span_is_not_rescored(self):
        # A 0.3 s backchannel carries too little voice to identify, however confident the score.
        words = [W(1.0, 1.3, "Okay.")]
        assign_words(words, TURNS)
        locked = vote_sentences(words, sentence_spans(words),
                                scores=[SentenceScore("SPEAKER_01", 0.9)],
                                margin=0.15, min_seconds=0.6)
        assert words[0].speaker == "SPEAKER_00"
        assert locked == set()

    def test_missing_score_falls_back_to_diarization(self):
        words = self._words()
        assign_words(words, TURNS)
        vote_sentences(words, sentence_spans(words), scores=[SentenceScore(None, 0.0)])
        assert [w.speaker for w in words] == ["SPEAKER_00", "SPEAKER_00"]

    def test_flicker_smoothing_cannot_undo_a_locked_sentence(self):
        # A short confident interjection flanked by the same speaker is exactly what flicker
        # smoothing exists to absorb — and exactly what it must not absorb once the audio says
        # otherwise.
        words = [W(0.0, 1.0, "Lange"), W(1.0, 2.0, "Frage."),
                 W(2.0, 3.0, "Ja."),
                 W(3.0, 4.0, "Weiter"), W(4.0, 4.9, "so.")]
        scores = [SentenceScore(None, 0.0), SentenceScore("SPEAKER_01", 0.6),
                  SentenceScore(None, 0.0)]
        turns, _ = merge(words, TURNS, sentence_scores=scores, flicker_min_words=3)
        assert [t.text for t in turns] == ["Lange Frage.", "Ja.", "Weiter so."]

    def test_scores_shorter_than_the_sentence_list_are_tolerated(self):
        words = self._words() + [W(6.0, 7.0, "Mehr."), W(7.0, 8.0, "Text.")]
        turns, _ = merge(words, TURNS, sentence_scores=[SentenceScore("SPEAKER_01", 0.5)])
        assert turns  # no IndexError


def test_speaker_stats_shares_sum_to_one():
    words = [W(0.0, 2.0, "a."), W(6.0, 9.0, "b.")]
    turns, stats = merge(words, TURNS)
    assert pytest.approx(sum(s["share"] for s in stats.values()), abs=1e-9) == 1.0
    for s in stats.values():
        assert s["longest"][1] >= s["longest"][0]
