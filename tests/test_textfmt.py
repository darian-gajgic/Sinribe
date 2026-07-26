"""Sentence splitting and timestamp formatting."""

from __future__ import annotations

from sinribe.textfmt import (
    format_sentences, hhmmss, hm, hms, human_duration, slugify, srt_time, vtt_time,
)


class TestFormatSentences:
    def test_splits_on_sentence_end(self):
        assert format_sentences("One. Two. Three.") == "One.\nTwo.\nThree."

    def test_keeps_abbreviations_together(self):
        assert format_sentences("Dr. Smith arrived.") == "Dr. Smith arrived."
        assert format_sentences("Das ist z.B. wichtig. Ende.") == "Das ist z.B. wichtig.\nEnde."

    def test_keeps_initials_together(self):
        assert format_sentences("J. R. R. Tolkien wrote it.") == "J. R. R. Tolkien wrote it."

    def test_standalone_list_marker_does_not_split(self):
        assert format_sentences("1. Buy milk.") == "1. Buy milk."

    def test_number_at_clause_end_still_splits(self):
        assert format_sentences("I scored 8. That is good.") == "I scored 8.\nThat is good."

    def test_question_and_exclamation(self):
        assert format_sentences("Really? Yes! Fine.") == "Really?\nYes!\nFine."

    def test_normalises_whitespace(self):
        assert format_sentences("  a   b\n\nc.  ") == "a b c."

    def test_empty(self):
        assert format_sentences("") == ""
        assert format_sentences(None) == ""

    def test_no_trailing_punctuation(self):
        assert format_sentences("no full stop here") == "no full stop here"


class TestTimestamps:
    def test_hms_includes_hours(self):
        assert hms(0) == "0:00:00"
        assert hms(59) == "0:00:59"
        assert hms(3661) == "1:01:01"
        assert hms(8073) == "2:14:33"

    def test_hhmmss_zero_padded(self):
        assert hhmmss(4) == "00:00:04"
        assert hhmmss(3661) == "01:01:01"

    def test_hm(self):
        assert hm(0) == "00:00"
        assert hm(1230) == "00:20"
        assert hm(6420) == "01:47"

    def test_negatives_clamp(self):
        assert hms(-5) == "0:00:00"
        assert srt_time(-1) == "00:00:00,000"

    def test_srt_and_vtt(self):
        assert srt_time(3661.5) == "01:01:01,500"
        assert vtt_time(3661.5) == "01:01:01.500"
        assert srt_time(0.001) == "00:00:00,001"

    def test_human_duration(self):
        assert human_duration(45) == "45s"
        assert human_duration(552) == "9m 12s"
        assert human_duration(3900) == "1h 05m"


def test_slugify():
    assert slugify("00:20 — Rückkopplung") == "0020-rückkopplung"
    assert slugify("!!!") == "section"
