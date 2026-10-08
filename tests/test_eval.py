"""Unit tests for the accuracy harness — the thing every other accuracy claim rests on.

If the scorer is wrong, every measured improvement is a coin flip, so these check the specific
behaviours that were needed to make a real reference usable: multi-part timestamps, hyphenated
speaker names, editorial markers, and the German spelling variants that are not errors.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from sinribe.eval.align import align
from sinribe.eval.normalize import Normalizer, spell_int
from sinribe.eval.reference import RefTurn, load_reference, normalise_speakers
from sinribe.eval.score import score

DOCX_XML = """<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:body>{paras}</w:body>
</w:document>"""


def make_docx(path: Path, lines: list[str]) -> Path:
    paras = "".join(f"<w:p><w:r><w:t>{ln}</w:t></w:r></w:p>" for ln in lines)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", DOCX_XML.format(paras=paras))
    return path


class TestNormalizer:
    def test_folds_umlauts_and_sharp_s(self):
        assert Normalizer().words("Läufer weiß") == ["laeufer", "weiss"]

    def test_strips_punctuation_and_case(self):
        assert Normalizer().words("Ja, wirklich?!") == ["ja", "wirklich"]

    def test_splits_hyphenated_compounds(self):
        # Written apart on one side and joined on the other is a spelling choice, so both must
        # reach the aligner as separable pieces for the compound rule to resolve them.
        assert Normalizer().words("VR-Bank") == ["vr", "bank"]

    def test_drops_hesitation_sounds(self):
        assert Normalizer().words("also äh ja") == ["also", "ja"]

    def test_keeps_words_that_only_look_like_fillers(self):
        # "um" and "eh" are ordinary German words; deleting them would hide real errors.
        assert Normalizer().words("um fünf Uhr") == ["um", "fuenf", "uhr"]

    def test_can_keep_fillers(self):
        assert Normalizer(drop_fillers=False).words("äh ja") == ["aeh", "ja"]

    @pytest.mark.parametrize("n,expected", [
        (0, "null"), (7, "sieben"), (12, "zwölf"), (21, "einundzwanzig"),
        (44, "vierundvierzig"), (100, "einhundert"), (305, "dreihundertfünf"),
        (1982, "eintausendneunhundertzweiundachtzig"),
    ])
    def test_spells_numbers(self, n, expected):
        assert spell_int(n) == expected

    def test_digits_and_words_compare_equal(self):
        nz = Normalizer()
        assert nz.words("44 Jahre") == nz.words("vierundvierzig Jahre")


class TestAlign:
    def test_identical(self):
        al = align(["a", "b", "c"], ["a", "b", "c"])
        assert al.wer == 0.0
        assert al.correct == 3

    def test_substitution(self):
        al = align(["das", "ist", "gut"], ["es", "ist", "gut"])
        assert (al.substitutions, al.deletions, al.insertions) == (1, 0, 0)
        assert al.wer == pytest.approx(1 / 3)

    def test_deletion_and_insertion(self):
        assert align(["a", "b", "c"], ["a", "c"]).deletions == 1
        assert align(["a", "c"], ["a", "b", "c"]).insertions == 1

    def test_compound_split_is_not_an_error(self):
        al = align(["softskills"], ["soft", "skills"])
        assert al.wer == 0.0
        assert [o.kind for o in al.ops] == ["ortho"]

    def test_compound_join_is_not_an_error(self):
        al = align(["gern", "habt"], ["gernhabt"])
        assert al.wer == 0.0
        assert al.correct == 2

    def test_compound_inside_a_longer_disagreement(self):
        # The real shape: one spelling variant sitting next to genuine misrecognitions.
        al = align(["riesen", "thema", "da", "mach"], ["riesenthema", "ich", "mache"])
        assert al.correct == 2                       # riesen + thema
        assert al.substitutions + al.deletions == 2  # da mach -> ich mache

    def test_unrelated_words_do_not_merge(self):
        al = align(["hund", "katze"], ["hundkatzemaus"])
        assert al.correct == 0

    def test_empty_hypothesis(self):
        al = align(["a", "b"], [])
        assert al.deletions == 2
        assert al.wer == 1.0


class TestReference:
    def test_parses_timestamped_lines(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text("[00:00:01] Interviewer: Hallo\n[00:00:05] Gast: Guten Tag\n")
        turns = load_reference(p)
        assert [t.speaker for t in turns] == ["interviewer", "gast"]
        assert turns[1].start == 5.0

    def test_hyphenated_speaker_survives(self, tmp_path):
        # A bare dash must not end the name, or "Interview-Partner" becomes a speaker called
        # "Interview" and the attribution score silently splits one person into two.
        p = tmp_path / "r.txt"
        p.write_text("[00:00:01] Interview-Partner: Hallo\n[00:00:04] Interview- Partner: Ja\n")
        assert {t.speaker for t in load_reference(p)} == {"interviewpartner"}

    def test_wrapped_lines_join_the_turn_above(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text("[00:00:01] A: erste Zeile\nzweite Zeile\n")
        turns = load_reference(p)
        assert len(turns) == 1
        assert turns[0].text == "erste Zeile zweite Zeile"

    def test_restarting_timestamps_become_parts(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text("[00:10:00] A: eins\n[00:20:00] A: zwei\n[00:01:00] A: drei\n")
        turns = load_reference(p)
        assert [t.part for t in turns] == [0, 0, 1]
        assert [t.text for t in turns] == ["eins", "zwei", "drei"]   # order preserved

    def test_small_backwards_jump_is_not_a_part(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text("[00:10:00] A: eins\n[00:09:58] A: zwei\n")
        assert {t.part for t in load_reference(p)} == {0}

    def test_drops_inaudible_markers(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text("[00:00:01] A: das ist (unverständlich) wichtig\n")
        assert load_reference(p)[0].text == "das ist wichtig"

    def test_keeps_inaudible_markers_on_request(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text("[00:00:01] A: das ist (unverständlich) wichtig\n")
        assert "unverständlich" in load_reference(p, keep_inaudible=True)[0].text

    def test_reads_docx(self, tmp_path):
        p = make_docx(tmp_path / "r.docx", ["[00:00:01] A: Hallo", "[00:00:03] B: Tag"])
        assert [t.text for t in load_reference(p)] == ["Hallo", "Tag"]

    def test_reads_sinribe_markdown(self, tmp_path):
        p = tmp_path / "out.md"
        p.write_text("# Title\n\n---\n\n**[00:00:01] Person 1**\nHallo.\n\n"
                     "**[00:00:05] Person 2**\nGuten Tag.\n")
        turns = load_reference(p)
        assert [t.speaker for t in turns] == ["person1", "person2"]
        assert turns[0].text == "Hallo."

    def test_reads_srt(self, tmp_path):
        p = tmp_path / "s.srt"
        p.write_text("1\n00:00:01,000 --> 00:00:02,500\nPerson 1: Hallo\n\n")
        turns = load_reference(p)
        assert turns[0].start == pytest.approx(1.0)
        assert turns[0].text == "Hallo"

    def test_untimestamped_text_still_loads(self, tmp_path):
        p = tmp_path / "plain.txt"
        p.write_text("nur ein satz ohne zeitstempel\n")
        assert load_reference(p)[0].text == "nur ein satz ohne zeitstempel"

    def test_normalise_speakers_strips_stage_directions(self):
        turns = normalise_speakers([RefTurn(0.0, "Interviewer (lacht)", "x")])
        assert turns[0].speaker == "interviewer"


class TestScore:
    def test_perfect_transcript(self):
        ref = [RefTurn(0.0, "a", "hallo welt")]
        rep = score(ref, [RefTurn(0.0, "person1", "hallo welt")])
        assert rep.total.wer == 0.0
        assert rep.speaker_accuracy == 1.0

    def test_counts_each_error_kind(self):
        ref = [RefTurn(0.0, "a", "eins zwei drei vier")]
        hyp = [RefTurn(0.0, "p", "eins zwo drei vier fuenf")]
        rep = score(ref, hyp)
        assert rep.total.sub == 1
        assert rep.total.ins == 1
        assert rep.total.wer == pytest.approx(0.5)

    def test_speaker_mapping_is_learned_not_assumed(self):
        # The reference calls them "gast"/"host"; the app says "Person 1"/"Person 2". Attribution
        # is about who spoke which words, not about the labels agreeing.
        ref = [RefTurn(0.0, "gast", "eins zwei"), RefTurn(5.0, "host", "drei vier")]
        hyp = [RefTurn(0.0, "person2", "eins zwei"), RefTurn(5.0, "person1", "drei vier")]
        rep = score(ref, hyp)
        assert rep.speaker_map == {"gast": "person2", "host": "person1"}
        assert rep.speaker_accuracy == 1.0

    def test_swapped_speakers_are_caught(self):
        ref = [RefTurn(0.0, "gast", "eins zwei drei"), RefTurn(5.0, "host", "vier")]
        hyp = [RefTurn(0.0, "person1", "eins zwei drei vier")]
        rep = score(ref, hyp)
        assert rep.speaker_accuracy < 1.0

    def test_per_speaker_breakdown(self):
        ref = [RefTurn(0.0, "gast", "eins zwei"), RefTurn(5.0, "host", "drei vier")]
        hyp = [RefTurn(0.0, "p1", "eins zwei"), RefTurn(5.0, "p2", "drei falsch")]
        rep = score(ref, hyp)
        assert rep.by_speaker["gast"].wer == 0.0
        assert rep.by_speaker["host"].wer == pytest.approx(0.5)

    def test_spelling_variants_score_as_correct(self):
        ref = [RefTurn(0.0, "a", "wir haben uns gernhabt")]
        hyp = [RefTurn(0.0, "p", "wir haben uns gern habt")]
        rep = score(ref, hyp)
        assert rep.total.wer == 0.0
        assert rep.ortho_words == 1

    def test_confusion_pairs_are_reported(self):
        ref = [RefTurn(0.0, "a", "das ist gut"), RefTurn(2.0, "a", "das war gut")]
        hyp = [RefTurn(0.0, "p", "es ist gut"), RefTurn(2.0, "p", "es war gut")]
        rep = score(ref, hyp)
        assert ("das", "es", 2) in rep.confusions

    def test_confidence_split_uses_word_probabilities(self):
        from sinribe.eval.reference import RefWord
        ref = [RefTurn(0.0, "a", "eins zwei")]
        hyp = [RefTurn(0.0, "p", "eins drei",
                       words=[RefWord("eins", 0.0, 0.5, 0.99), RefWord("drei", 0.5, 1.0, 0.2)])]
        rep = score(ref, hyp)
        assert rep.prob_correct == pytest.approx(0.99)
        assert rep.prob_wrong == pytest.approx(0.2)

    def test_empty_reference_does_not_divide_by_zero(self):
        rep = score([], [RefTurn(0.0, "p", "hallo")])
        assert rep.total.wer == 0.0
        assert rep.format()


class TestCeiling:
    """How much a perfect chooser between transcripts would win — the 'keep tuning?' answer."""

    @staticmethod
    def _write(tmp_path, name, text):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "t.md").write_text(f"# t\n\n---\n\n**[00:00:00] Person 1**\n{text}\n")
        return d

    def test_reports_the_union_over_configurations(self, tmp_path, capsys):
        from sinribe.eval.ceiling import run_ceiling
        ref = tmp_path / "ref.txt"
        ref.write_text("[00:00:00] A: eins zwei drei vier\n")
        out = tmp_path / "runs"
        self._write(out, "a", "eins zwei falsch falsch")   # gets the first half
        self._write(out, "b", "falsch falsch drei vier")   # gets the second half
        assert run_ceiling(out, ref) == 0
        printed = capsys.readouterr().out
        assert "50.0% correct" in printed        # either one alone
        assert "100.0% correct" in printed       # perfect choice between them
        assert "0.0%" in printed                 # nothing is beyond both

    def test_names_the_floor_when_a_word_is_never_right(self, tmp_path, capsys):
        from sinribe.eval.ceiling import run_ceiling
        ref = tmp_path / "ref.txt"
        ref.write_text("[00:00:00] A: eins zwei\n")
        out = tmp_path / "runs"
        self._write(out, "a", "eins falsch")
        self._write(out, "b", "eins auch falsch")
        assert run_ceiling(out, ref) == 0
        assert "50.0%   — the floor" in capsys.readouterr().out

    def test_collapsed_transcripts_are_excluded(self, tmp_path, capsys):
        # A decode that returned nothing has not made different mistakes, it has failed; counting
        # it would leave the union unchanged while pretending another config was considered.
        from sinribe.eval.ceiling import run_ceiling
        ref = tmp_path / "ref.txt"
        ref.write_text("[00:00:00] A: eins zwei drei vier fuenf sechs\n")
        out = tmp_path / "runs"
        self._write(out, "good", "eins zwei drei vier fuenf sechs")
        self._write(out, "collapsed", "eins")
        assert run_ceiling(out, ref) == 0
        printed = capsys.readouterr().out
        assert "excluded as broken" in printed
        assert "1 configurations" in printed

    def test_empty_directory_is_an_error(self, tmp_path):
        from sinribe.eval.ceiling import run_ceiling
        ref = tmp_path / "ref.txt"
        ref.write_text("[00:00:00] A: eins\n")
        (tmp_path / "empty").mkdir()
        assert run_ceiling(tmp_path / "empty", ref) == 2


class TestPresets:
    """Behaviour of the ladder, not its contents — the rungs are recalibrated by measurement."""

    def test_every_rung_is_selectable_by_its_own_target(self):
        from sinribe import presets
        for rung in presets.RUNGS:
            assert presets.for_target(rung.target_rtf) is rung

    def test_rungs_are_ordered_fastest_first(self):
        # The slider maps position to index directly, so the order is part of the contract.
        from sinribe import presets
        targets = [r.target_rtf for r in presets.RUNGS]
        assert targets == sorted(targets, reverse=True)
        assert len(set(targets)) == len(targets)

    def test_between_rungs_the_slower_one_wins(self):
        # The slider promises a speed ceiling; missing it by running faster and less accurately
        # than asked would be the wrong way to be wrong.
        from sinribe import presets
        fast, slow = sorted(r.target_rtf for r in presets.RUNGS)[-2:]
        assert presets.for_target(slow - 0.5).target_rtf < slow
        assert presets.for_target(fast).target_rtf == fast

    def test_out_of_range_is_clamped(self):
        from sinribe import presets
        assert presets.for_target(999).target_rtf == presets.MAX_TARGET
        assert presets.for_target(0).target_rtf == presets.MIN_TARGET

    def test_label_admits_when_a_rung_is_unmeasured(self):
        from sinribe.presets import Rung
        assert "errors" not in Rung(target_rtf=9, model="m", note="").label()
        assert "17.4% errors" in Rung(target_rtf=9, model="m", note="",
                                      measured_rtf=9.4, measured_wer=0.174).label()

    def test_labels_distinguish_rungs_that_are_close_together(self):
        # The shipped rungs sit a third of a point apart; whole percents would print two of them
        # identically and make the slider look like it does nothing.
        from sinribe import presets
        measured = [r.label() for r in presets.RUNGS if r.measured_wer is not None]
        assert len(set(measured)) == len(measured)

    def test_exactly_one_rung_is_recommended(self):
        from sinribe import presets
        assert sum(1 for r in presets.RUNGS if r.recommended) == 1
        assert presets.recommended().recommended

    def test_the_default_is_the_recommended_rung(self):
        # Someone who never touches the slider should land on the advice, not on whichever rung
        # happens to sit at the end of the list.
        from sinribe import config, presets
        assert presets.for_target(config.DEFAULTS["speed_target"]) is presets.recommended()

    def test_the_recommendation_is_marked_in_its_label(self):
        from sinribe import presets
        assert "recommended" in presets.recommended().label()
        for rung in presets.RUNGS:
            if not rung.recommended:
                assert "recommended" not in rung.label()

    def test_duration_estimate_uses_the_measured_speed(self):
        from sinribe.presets import Rung
        rung = Rung(target_rtf=10, model="m", note="", measured_rtf=20.0)
        assert rung.duration_for(600.0) == 30.0        # measured wins over the target
        assert Rung(target_rtf=10, model="m", note="").duration_for(600.0) == 60.0

    def test_what_this_machine_actually_did_beats_the_benchmark(self):
        # The shipped figures came off one machine on one day and proved unreproducible; a real
        # observation from the user's own hardware has to win.
        from sinribe.presets import Rung
        rung = Rung(target_rtf=10, model="m", note="", measured_rtf=20.0)
        assert rung.duration_for(600.0, observed_rtf=5.0) == 120.0

    def test_observed_speeds_are_averaged_not_replaced(self):
        # One job that ran while Ollama was loading a model must not become the estimate.
        from sinribe import presets
        cfg = {}
        target = presets.RUNGS[0].target_rtf
        presets.record_rtf(cfg, target, 20.0)
        assert presets.observed_rtf(cfg, presets.RUNGS[0]) == 20.0
        presets.record_rtf(cfg, target, 10.0)
        assert presets.observed_rtf(cfg, presets.RUNGS[0]) == 15.0

    def test_nonsense_speeds_are_ignored(self):
        from sinribe import presets
        cfg = {}
        presets.record_rtf(cfg, presets.RUNGS[0].target_rtf, 0.0)
        presets.record_rtf(cfg, presets.RUNGS[0].target_rtf, -3.0)
        assert presets.observed_rtf(cfg, presets.RUNGS[0]) is None

    def test_a_corrupt_observation_does_not_crash_the_estimate(self):
        from sinribe import presets
        cfg = {"observed_rtf": {str(presets.RUNGS[0].target_rtf): "not a number"}}
        assert presets.observed_rtf(cfg, presets.RUNGS[0]) is None

    def test_the_star_marks_the_most_accurate_measured_rung(self):
        """The invariant that actually protects the user.

        The slider deliberately spans the whole range, including slow positions that measured no
        better — they exist so their labels can say so. What must never happen is the ★ pointing
        somewhere other than the best measured setting, because that is the one number a user who
        does not want to read any of this will act on.
        """
        from sinribe import presets
        measured = [r for r in presets.RUNGS if r.measured_wer is not None]
        assert presets.recommended().measured_wer == min(r.measured_wer for r in measured)

    def test_accuracy_improves_down_to_the_recommendation(self):
        # Everything from the fast end down to the ★ must be a real improvement, or the fast half
        # of the slider is selling time for nothing.
        from sinribe import presets
        upto = []
        for rung in presets.RUNGS:
            upto.append(rung)
            if rung.recommended:
                break
        wers = [r.measured_wer for r in upto if r.measured_wer is not None]
        assert wers == sorted(wers, reverse=True), wers

    def test_unverified_rungs_do_not_print_an_error_rate(self):
        # A predicted number that has not survived the shipped implementation must not be shown
        # as though it had — a simulation already over-promised by 0.4 points once.
        from sinribe import presets
        for rung in presets.RUNGS:
            if rung.measured_wer is None:
                assert "errors" not in rung.label()
                assert "NOT" in rung.note or "not" in rung.note

    def test_the_slider_spans_the_full_measured_range(self):
        from sinribe import presets
        assert presets.MIN_TARGET <= 1
        assert presets.MAX_TARGET >= 84

    def test_voting_rungs_use_genuinely_different_passes(self):
        # Voting can only correct a word the passes disagree about, so identical passes would be
        # pure wasted compute.
        from sinribe import presets
        for rung in presets.RUNGS:
            seen = [rung.decode] + list(rung.extra_passes)
            keyed = [tuple(sorted(p.items())) for p in seen]
            assert len(set(keyed)) == len(keyed), f"{rung.target_rtf}x repeats a pass"

    def test_every_rung_names_a_known_model(self):
        from sinribe import config, presets
        for rung in presets.RUNGS:
            assert rung.model in config.ASR_MODELS

    def test_no_rung_pairs_hotwords_with_a_model_that_breaks_on_them(self):
        # large-v3-german returns an empty transcript when prompted; it must never be what
        # "auto" silently selects for someone who typed names into the box.
        from sinribe import config, presets
        for rung in presets.RUNGS:
            assert config.supports_hotwords(rung.model), rung.target_rtf

    def test_decode_overrides_reach_the_settings(self):
        from sinribe import presets
        cfg = {"speed_target": 6, "decode_overrides": {"beam_size": 42}}
        assert presets.decode_settings(cfg)["beam_size"] == 42

    def test_unknown_override_keys_are_ignored(self):
        from sinribe import presets
        cfg = {"speed_target": 6, "decode_overrides": {"nonsense": True}}
        assert "nonsense" not in presets.decode_settings(cfg)

    def test_explicit_model_beats_the_rung(self):
        from sinribe import presets
        assert presets.model_for({"asr_model": "small", "speed_target": 1}) == "small"
        assert presets.model_for({"asr_model": "auto", "speed_target": 12}) == \
            presets.for_target(12).model
