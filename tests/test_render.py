"""Renderer tests: markdown structure, subtitles, sidecar round-trip."""

from __future__ import annotations

import json

from sinribe.merge import Turn, Word, merge, DiarTurn
from sinribe.render import markdown, sidecar, subtitles


def _sample():
    words = [Word(0.5, 1.0, "Hello"), Word(1.1, 1.6, "there."),
             Word(6.0, 6.5, "Hi"), Word(6.6, 7.0, "back.")]
    diar = [DiarTurn(0.0, 5.0, "SPEAKER_00"), DiarTurn(5.0, 10.0, "SPEAKER_01")]
    return merge(words, diar)


RESULT = {
    "source": "/tmp/vorlesung.m4a", "title": "vorlesung", "duration": 8073.0,
    "language": "de", "language_probability": 0.98, "model": "large-v3",
    "device": "cuda", "compute_type": "float16", "diar_pipeline":
        "pyannote/speaker-diarization-community-1", "elapsed": 552.0,
    "realtime_factor": 14.6, "finished_at": "2026-07-26 18:03",
}


class TestMarkdown:
    def test_structure(self):
        turns, stats = _sample()
        md = markdown.render(turns, stats, RESULT, title="Vorlesung Systemtheorie")
        assert md.startswith("# Vorlesung Systemtheorie\n")
        assert "**Duration** 2:14:33" in md
        assert "**Speakers** 2" in md
        assert "**Language** de (0.98)" in md
        assert "large-v3 · CUDA · float16" in md
        assert "**Diarization** speaker-diarization-community-1" in md
        assert "(14.6× realtime)" in md
        assert "| Speaker | Talk time | Share |" in md
        assert "**[00:00:00] Person 1**" in md
        assert "**[00:00:06] Person 2**" in md
        assert md.endswith("\n")

    def test_shares_are_percentages(self):
        turns, stats = _sample()
        md = markdown.render(turns, stats, RESULT)
        pcts = [ln for ln in md.splitlines() if ln.startswith("| Person")]
        assert len(pcts) == 2
        assert all("%" in p for p in pcts)

    def test_summary_and_chapters_included_when_present(self):
        turns, stats = _sample()
        enr = {"summary": "A lecture about feedback loops.",
               "chapters": [{"start": 0.0, "title": "Einleitung"},
                            {"start": 1230.0, "title": "Rückkopplung"}]}
        md = markdown.render(turns, stats, RESULT, enrichment=enr)
        assert "## Summary" in md
        assert "A lecture about feedback loops." in md
        assert "## Chapters" in md
        assert "- [**00:00** Einleitung](#0000-einleitung)" in md
        assert "- [**00:20** Rückkopplung](#0020-rückkopplung)" in md

    def test_every_chapter_link_has_a_matching_anchor(self):
        """A table of contents whose links go nowhere is worse than none."""
        import re
        turns, stats = _sample()
        enr = {"summary": "", "chapters": [{"start": 0.0, "title": "Einleitung"},
                                           {"start": 3.0, "title": "Rückkopplung"}]}
        md = markdown.render(turns, stats, RESULT, enrichment=enr)
        targets = set(re.findall(r"\]\(#([^)]+)\)", md))
        anchors = set(re.findall(r'<a id="([^"]+)"></a>', md))
        assert targets, "expected chapter links"
        assert targets <= anchors, f"dangling links: {targets - anchors}"

    def test_no_enrichment_sections_when_absent(self):
        turns, stats = _sample()
        md = markdown.render(turns, stats, RESULT, enrichment={})
        assert "## Summary" not in md
        assert "## Chapters" not in md

    def test_empty_transcript_is_still_valid(self):
        md = markdown.render([], {}, RESULT)
        assert "*No speech detected" in md
        assert "**Speakers** 0" in md

    def test_sentences_are_one_per_line(self):
        words = [Word(0.0, 1.0, "One."), Word(1.05, 2.0, "Two.")]
        turns, stats = merge(words, [DiarTurn(0.0, 5.0, "S0")])
        md = markdown.render(turns, stats, RESULT)
        assert "One.\nTwo." in md

    def test_write_is_atomic_and_creates_parents(self, tmp_path):
        p = tmp_path / "nested" / "out.md"
        markdown.write(p, "# hi\n")
        assert p.read_text() == "# hi\n"
        assert not (tmp_path / "nested" / "out.md.tmp").exists()


class TestEnrichmentLanguage:
    """gemma3:4b summarised a German interview in English and titled its chapters
    "Client Background & History". "Same language as the input" is not an instruction a small
    model reliably follows; naming the language is."""

    def test_detected_language_is_named(self):
        from sinribe.enrich import _language_rule
        assert "German" in _language_rule("de")
        assert "English" in _language_rule("en")

    def test_unknown_language_falls_back_without_naming_one(self):
        from sinribe.enrich import _language_rule
        for code in (None, "", "auto", "zz"):
            rule = _language_rule(code)
            assert "same language" in rule
            assert "German" not in rule


class TestReviewSpans:
    """The 'worth a listen' list — a proofreading aid, so it must point at real spots only."""

    @staticmethod
    def _turn(probs, start=0.0):
        words = [Word(start + i * 0.5, start + i * 0.5 + 0.4, f"w{i}", prob=p)
                 for i, p in enumerate(probs)]
        return Turn(speaker="Person 1", raw="S0", start=words[0].start, end=words[-1].end,
                    text=" ".join(w.word for w in words), words=words)

    def test_confident_transcript_has_nothing_to_review(self):
        turns = [self._turn([0.99] * 8)]
        assert markdown.review_spans(turns) == []

    def test_finds_the_uncertain_word(self):
        turns = [self._turn([0.99, 0.99, 0.2, 0.99, 0.99])]
        spans = markdown.review_spans(turns)
        assert len(spans) == 1
        assert "**w2**" in spans[0][2]

    def test_includes_surrounding_context(self):
        turns = [self._turn([0.9] * 6 + [0.1] + [0.9] * 6)]
        _, _, text = markdown.review_spans(turns, context=2)[0]
        assert "w4 w5 **w6** w7 w8" in text

    def test_nearby_uncertain_words_merge_into_one_passage(self):
        # "…, [one confident word], …" is one thing to re-listen to, not two.
        turns = [self._turn([0.9, 0.2, 0.9, 0.2, 0.9])]
        assert len(markdown.review_spans(turns)) == 1

    def test_distant_uncertain_words_stay_separate(self):
        turns = [self._turn([0.2] + [0.9] * 6 + [0.2])]
        assert len(markdown.review_spans(turns)) == 2

    def test_reports_the_start_of_the_uncertain_run(self):
        turns = [self._turn([0.99, 0.99, 0.2, 0.99], start=100.0)]
        assert markdown.review_spans(turns)[0][0] == 101.0

    def test_respects_the_cap(self):
        turns = [self._turn([0.2, 0.9, 0.9, 0.9, 0.9] * 40)]
        assert len(markdown.review_spans(turns, max_spans=5)) == 5

    def test_the_cap_keeps_the_least_confident_not_the_earliest(self):
        # Truncating in time order would hand back a review list covering only the opening
        # minutes of a long recording while the worst passages sit past the cutoff.
        probs = [0.9] * 60
        probs[2] = 0.45          # early, mildly doubtful
        probs[50] = 0.05         # late, the worst passage in the file
        spans = markdown.review_spans([self._turn(probs)], max_spans=1)
        assert "**w50**" in spans[0][2]

    def test_capped_output_is_still_in_time_order(self):
        probs = [0.9] * 60
        for i, p in ((5, 0.1), (25, 0.05), (45, 0.02)):
            probs[i] = p
        starts = [s for s, _, _ in markdown.review_spans([self._turn(probs)], max_spans=3)]
        assert starts == sorted(starts)

    def test_total_count_ignores_the_cap(self):
        turns = [self._turn([0.2, 0.9, 0.9, 0.9, 0.9] * 40)]
        assert markdown.count_review_spans(turns) == 40

    def test_section_says_when_it_is_showing_a_subset(self):
        turns = [self._turn([0.2, 0.9, 0.9, 0.9, 0.9] * 80)]
        md = markdown.render(turns, {}, RESULT, review=True, review_max=60)
        assert "of 80" in md

    def test_section_stays_quiet_when_nothing_was_dropped(self):
        turns = [self._turn([0.2, 0.9, 0.9, 0.9, 0.9] * 3)]
        md = markdown.render(turns, {}, RESULT, review=True)
        assert "## Worth a listen" in md
        assert "Showing the" not in md

    def test_nothing_is_dropped_by_default(self):
        # The section is a proofreading checklist. It used to stop at 60, so on the recording it
        # was built for it silently withheld 303 of the 363 places worth an ear.
        turns = [self._turn([0.2, 0.9, 0.9, 0.9, 0.9] * 80)]
        assert len(markdown.review_spans(turns)) == 80
        md = markdown.render(turns, {}, RESULT, review=True)
        assert "Showing the" not in md
        assert "All 80 of them are here" in md

    def test_a_cap_of_zero_means_everything(self):
        turns = [self._turn([0.2, 0.9, 0.9, 0.9, 0.9] * 40)]
        assert len(markdown.review_spans(turns, max_spans=0)) == 40

    def test_spans_are_sorted_by_time(self):
        turns = [self._turn([0.2, 0.9], start=50.0), self._turn([0.2, 0.9], start=10.0)]
        starts = [s for s, _, _ in markdown.review_spans(turns)]
        assert starts == sorted(starts)

    def test_section_is_opt_in(self):
        turns, stats = _sample()
        assert "Worth a listen" not in markdown.render(turns, stats, RESULT)

    def test_section_renders_when_asked(self):
        turns = [self._turn([0.99, 0.2, 0.99])]
        md = markdown.render(turns, {}, RESULT, review=True)
        assert "## Worth a listen" in md
        assert "**[00:00:00]**" in md

    def test_no_section_when_everything_is_confident(self):
        turns, stats = _sample()   # merge() defaults every word to prob 1.0
        assert "Worth a listen" not in markdown.render(turns, stats, RESULT, review=True)


class TestSubtitles:
    def test_srt_format(self):
        turns, _ = _sample()
        srt = subtitles.render_srt(turns)
        assert srt.startswith("1\n00:00:00,500 --> ")
        assert "[Person 1] Hello there." in srt
        assert "[Person 2] Hi back." in srt

    def test_vtt_format(self):
        turns, _ = _sample()
        vtt = subtitles.render_vtt(turns)
        assert vtt.startswith("WEBVTT\n")
        assert "<v Person 1>Hello there." in vtt
        assert " --> " in vtt

    def test_long_turn_is_split_into_multiple_cues(self):
        words = [Word(i * 0.4, i * 0.4 + 0.3, "word") for i in range(60)]
        turns, _ = merge(words, [DiarTurn(0.0, 40.0, "S0")], max_turn_chars=10_000)
        srt = subtitles.render_srt(turns)
        assert srt.count(" --> ") > 3

    def test_cue_has_minimum_duration(self):
        turns = [Turn(speaker="Person 1", raw="S0", start=1.0, end=1.0, text="Hi.",
                      words=[Word(1.0, 1.0, "Hi.")])]
        srt = subtitles.render_srt(turns)
        start, _, end = srt.splitlines()[1].partition(" --> ")
        assert end > start


class TestSidecar:
    def test_round_trip_preserves_turns_and_words(self, tmp_path):
        turns, stats = _sample()
        payload = sidecar.build(turns, stats, RESULT, diar_turns=[DiarTurn(0, 5, "S0")])
        p = tmp_path / "x.sinribe.json"
        sidecar.write(p, payload)

        loaded, lstats, meta, raw = sidecar.load(p)
        assert len(loaded) == len(turns)
        assert loaded[0].text == turns[0].text
        assert loaded[0].words[0].word == "Hello"
        assert meta["model"] == "large-v3"
        assert set(lstats) == set(stats)
        assert raw["schema_version"] == sidecar.SCHEMA_VERSION

    def test_is_valid_json_with_unicode(self, tmp_path):
        turns, stats = _sample()
        turns[0].text = "Rückkopplung über Systeme."
        p = tmp_path / "u.json"
        sidecar.write(p, sidecar.build(turns, stats, RESULT))
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["turns"][0]["text"] == "Rückkopplung über Systeme."

    def test_apply_names_renames_and_rerenders(self):
        turns, stats = _sample()
        sidecar.apply_names(turns, {"Person 1": "Prof. Müller", "Person 2": ""})
        assert turns[0].speaker == "Prof. Müller"
        assert turns[1].speaker == "Person 2"   # blank rename is ignored
        md = markdown.render(turns, {"Prof. Müller": {"seconds": 1.0, "share": 1.0,
                                                      "turns": 1, "raw": "S0", "first": 0.0,
                                                      "longest": (0.0, 1.0)}}, RESULT)
        assert "Prof. Müller" in md
