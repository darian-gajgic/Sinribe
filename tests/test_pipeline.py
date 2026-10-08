"""Pipeline-level tests that do not need a GPU: VRAM tiering, job keys, cache housekeeping."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from sinribe import presets
from sinribe.merge import DiarTurn
from sinribe.pipeline import (
    ASR_TIERS, JobSpec, Runner, StageError, _asr_env, _diar_env, _job_key, pick_asr_tier,
    purge_old_jobs, speech_coverage,
)
# Safe to import from the ASR venv: the worker only pulls torch/pyannote inside main().
from sinribe.workers.refine_worker import trusted_turns


class TestVramTiers:
    def test_tier_indices_are_in_range(self):
        for free in (-1, 0, 1000, 2500, 3500, 5500, 7000, 12000, 99999):
            assert 0 <= pick_asr_tier(free) < len(ASR_TIERS)

    def test_more_vram_never_selects_a_heavier_tier(self):
        # Tiers are ordered heaviest-first, so the index must be monotonically non-increasing
        # as free VRAM grows.
        prev = pick_asr_tier(0)
        for free in range(0, 13_000, 250):
            cur = pick_asr_tier(free)
            assert cur <= prev
            prev = cur

    def test_batch16_needs_real_headroom(self):
        # Measured footprint of float16/batch16 on this box is ~9.1 GB; picking it with less
        # than that free means an OOM several minutes into a long transcription.
        assert pick_asr_tier(9_000) > 0
        assert pick_asr_tier(11_000) == 0

    def test_low_vram_falls_back_to_cpu(self):
        assert ASR_TIERS[pick_asr_tier(500)][0] == "cpu"

    def test_unknown_vram_is_conservative(self):
        # -1 means nvidia-smi failed; must not gamble on the largest tier.
        assert pick_asr_tier(-1) > 0


class TestWorkerEnvironments:
    def test_asr_env_adds_cuda12_libs(self):
        env = _asr_env()
        path = env["LD_LIBRARY_PATH"]
        assert "nvidia/cublas/lib" in path
        assert "nvidia/cudnn/lib" in path
        assert env["HF_HUB_OFFLINE"] == "1"

    def test_diar_env_strips_ld_library_path(self, monkeypatch):
        # cuDNN 12 and cuDNN 13 wheels share the SONAME libcudnn.so.9, so an inherited path
        # from the launcher would make torch load a cuDNN built for the wrong CUDA major.
        monkeypatch.setenv("LD_LIBRARY_PATH", "/some/cuda12/lib")
        env = _diar_env()
        assert "LD_LIBRARY_PATH" not in env
        assert env["HF_HUB_OFFLINE"] == "1"


class TestJobKey:
    def test_stable_for_same_file(self, tmp_path):
        p = tmp_path / "a.wav"
        p.write_bytes(b"x" * 100)
        assert _job_key(p) == _job_key(p)

    def test_changes_when_content_changes(self, tmp_path):
        p = tmp_path / "a.wav"
        p.write_bytes(b"x" * 100)
        first = _job_key(p)
        time.sleep(0.01)
        p.write_bytes(b"y" * 200)
        assert _job_key(p) != first

    def test_differs_between_files(self, tmp_path):
        a, b = tmp_path / "a.wav", tmp_path / "b.wav"
        a.write_bytes(b"x" * 10)
        b.write_bytes(b"x" * 10)
        assert _job_key(a) != _job_key(b)


class TestPurge:
    def test_removes_only_old_dirs(self, tmp_path, monkeypatch):
        import sinribe.pipeline as pl

        jobs = tmp_path / "jobs"
        jobs.mkdir()
        old, new = jobs / "old", jobs / "new"
        for d in (old, new):
            d.mkdir()
            (d / "diar.json").write_text("{}")
        past = time.time() - 40 * 86400
        import os
        os.utime(old, (past, past))

        monkeypatch.setattr(pl, "JOBS_DIR", jobs)
        removed = purge_old_jobs(days=14)
        assert removed == 1
        assert not old.exists()
        assert new.exists()

    def test_missing_dir_is_not_an_error(self, tmp_path, monkeypatch):
        import sinribe.pipeline as pl
        monkeypatch.setattr(pl, "JOBS_DIR", tmp_path / "nope")
        assert purge_old_jobs(14) == 0


class TestWorkerErrorPropagation:
    """A failing worker emits an `error` event and *then* exits non-zero.

    The exit-code check lives inside the _run_worker generator, so it runs before the caller's
    loop body sees that last event. Unless the generator captures it, the real diagnosis is
    replaced by a bare "exited with code 1" — which is how a plain "diarizer not installed"
    reached the user as an unreadable stage failure — and kind="oom" is flattened to "runtime",
    silently disabling the ASR tier ladder that dispatches on it.
    """

    def _drive(self, tmp_path, body: str):
        (tmp_path / "fake_worker.py").write_text(body)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(tmp_path)
        runner = Runner(JobSpec(input_path=None, output_dir=tmp_path))
        seen: list[dict] = []
        for ev in runner._run_worker(Path(sys.executable), "fake_worker", {"x": 1}, env):
            seen.append(ev)
        return seen

    def test_error_event_survives_a_nonzero_exit(self, tmp_path):
        with pytest.raises(StageError) as excinfo:
            self._drive(tmp_path,
                        'import json, sys\n'
                        'sys.stdin.readline()\n'
                        'print(json.dumps({"ev": "error", "kind": "load", '
                        '"msg": "pipeline not in local cache"}), flush=True)\n'
                        'sys.exit(1)\n')
        assert "pipeline not in local cache" in str(excinfo.value)
        assert excinfo.value.kind == "load"

    def test_oom_kind_reaches_the_caller(self, tmp_path):
        # This is what the ASR fallback ladder branches on; "runtime" would abort the job
        # instead of retrying at a smaller batch size.
        with pytest.raises(StageError) as excinfo:
            self._drive(tmp_path,
                        'import json, sys\n'
                        'sys.stdin.readline()\n'
                        'print(json.dumps({"ev": "error", "kind": "oom", '
                        '"msg": "CUDA out of memory"}), flush=True)\n'
                        'sys.exit(1)\n')
        assert excinfo.value.kind == "oom"

    def test_bare_nonzero_exit_still_reports_the_code(self, tmp_path):
        with pytest.raises(StageError, match="exited with code 3"):
            self._drive(tmp_path, 'import sys\nsys.stdin.readline()\nsys.exit(3)\n')

    def test_events_still_reach_the_caller(self, tmp_path):
        seen = self._drive(tmp_path,
                           'import json, sys\n'
                           'sys.stdin.readline()\n'
                           'print(json.dumps({"ev": "ready"}), flush=True)\n'
                           'print("stray library chatter", flush=True)\n'
                           'print(json.dumps({"ev": "done"}), flush=True)\n')
        assert [e.get("ev") for e in seen] == ["ready", "done"]


class TestTrustedTurns:
    """Voice prints are only as good as the turns they are built from: a turn that touches a
    differently-labelled neighbour may carry the other person's voice, and averaging that in
    blurs the two prints together until the comparison decides nothing."""

    def test_short_turns_are_rejected(self):
        assert trusted_turns([{"start": 0.0, "end": 1.0, "speaker": "A"}]) == []

    def test_isolated_long_turn_is_accepted(self):
        assert len(trusted_turns([{"start": 0.0, "end": 5.0, "speaker": "A"}])) == 1

    def test_turn_abutting_a_different_speaker_is_rejected(self):
        turns = [{"start": 0.0, "end": 5.0, "speaker": "A"},
                 {"start": 5.0, "end": 10.0, "speaker": "B"}]
        assert trusted_turns(turns) == []

    def test_a_gap_from_the_other_speaker_restores_trust(self):
        turns = [{"start": 0.0, "end": 5.0, "speaker": "A"},
                 {"start": 6.0, "end": 11.0, "speaker": "B"}]
        assert len(trusted_turns(turns)) == 2

    def test_same_speaker_neighbour_does_not_disqualify(self):
        turns = [{"start": 0.0, "end": 5.0, "speaker": "A"},
                 {"start": 5.0, "end": 10.0, "speaker": "A"}]
        assert len(trusted_turns(turns)) == 2


class TestCheckpointShape:
    """The resume path reuses a checkpoint only when its settings match exactly."""

    def test_settings_mismatch_is_detectable(self, tmp_path):
        ck = tmp_path / "diar.json"
        ck.write_text(json.dumps({"settings": {"pipeline": "a"}, "turns": []}))
        data = json.loads(ck.read_text())
        assert data["settings"] != {"pipeline": "b"}
        assert data["settings"] == {"pipeline": "a"}

    def test_every_decode_knob_reaches_the_asr_key(self):
        """A knob passed to the worker but absent from the checkpoint key silently reuses the
        previous decode — which during a sweep makes every variant score the same."""
        from sinribe import presets
        decode = presets.decode_settings({"speed_target": 6})
        src = (Path(__file__).resolve().parent.parent
               / "sinribe" / "pipeline.py").read_text()
        key_block = src.split("settings = {\n", 2)[2].split("}\n", 1)[0]
        assert "**{k: decode[k] for k in sorted(decode)}" in key_block
        assert set(presets.BASE_DECODE) == set(decode)


class TestDiarizerAvailability:
    """A pipeline that was never downloaded must not be selectable.

    Workers run with HF_HUB_OFFLINE=1, so picking an absent pipeline used to look fine right up
    until the job had decoded the audio and then died several minutes in.
    """

    def _cache(self, tmp_path, monkeypatch, *present):
        from sinribe import config
        for name in present:
            d = (tmp_path / f"models--{name.replace('/', '--')}" / "snapshots" / "abc123")
            d.mkdir(parents=True)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        return config

    def test_present_pipeline_is_available(self, tmp_path, monkeypatch):
        config = self._cache(tmp_path, monkeypatch, "pyannote/speaker-diarization-community-1")
        assert config.diar_available("pyannote/speaker-diarization-community-1")

    def test_absent_pipeline_is_not(self, tmp_path, monkeypatch):
        config = self._cache(tmp_path, monkeypatch, "pyannote/speaker-diarization-community-1")
        assert not config.diar_available("pyannote/speaker-diarization-3.1")

    def test_an_empty_snapshot_dir_does_not_count(self, tmp_path, monkeypatch):
        # An interrupted download leaves the directory behind with nothing in it.
        from sinribe import config
        (tmp_path / "models--pyannote--speaker-diarization-3.1" / "snapshots").mkdir(parents=True)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        assert not config.diar_available("pyannote/speaker-diarization-3.1")

    def test_hf_home_is_honoured(self, tmp_path, monkeypatch):
        from sinribe import config
        hub = tmp_path / "hub"
        (hub / "models--pyannote--speaker-diarization-community-1" / "snapshots" / "x").mkdir(
            parents=True)
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        assert config.diar_available("pyannote/speaker-diarization-community-1")

    def test_the_installed_pipeline_is_really_there(self):
        # Guards the real machine, not a fixture: install.sh downloads community-1.
        from sinribe import config
        assert config.diar_available("pyannote/speaker-diarization-community-1")


class TestHotwordCompatibility:
    """large-v3-german returns an EMPTY transcript when given hotwords. Measured, twice."""

    def test_the_broken_model_is_flagged(self):
        from sinribe.config import supports_hotwords
        assert not supports_hotwords("large-v3-german")

    def test_working_models_are_not(self):
        from sinribe.config import supports_hotwords
        assert supports_hotwords("large-v3")
        assert supports_hotwords("large-v3-turbo-german")
        assert supports_hotwords("small")


class TestSpeechCoverage:
    """A collapsed decode exits 0 and writes a tidy, nearly empty transcript. Catch it."""

    @staticmethod
    def _words(spans):
        from sinribe.merge import Word
        return [Word(start=a, end=b, word="x") for a, b in spans]

    def test_full_coverage(self):
        words = self._words([(0.0, 5.0), (5.0, 10.0)])
        assert speech_coverage(words, [DiarTurn(0.0, 10.0, "S0")]) == 1.0

    def test_collapsed_decode_is_caught(self):
        # The real failure: 26 seconds of words against 65 minutes of diarized speech.
        words = self._words([(0.0, 26.0)])
        assert speech_coverage(words, [DiarTurn(0.0, 3945.0, "S0")]) < 0.05

    def test_normal_gappy_transcript_passes(self):
        # Real transcripts leave gaps between words and pauses between turns; ~80 % is healthy.
        words = self._words([(i, i + 0.8) for i in range(100)])
        assert speech_coverage(words, [DiarTurn(0.0, 100.0, "S0")]) > 0.7

    def test_overlapping_diarization_is_not_double_counted(self):
        words = self._words([(0.0, 10.0)])
        turns = [DiarTurn(0.0, 10.0, "S0"), DiarTurn(0.0, 10.0, "S1")]
        assert speech_coverage(words, turns) == 1.0

    def test_overlapping_words_are_not_double_counted(self):
        words = self._words([(0.0, 10.0), (0.0, 10.0)])
        assert speech_coverage(words, [DiarTurn(0.0, 20.0, "S0")]) == 0.5

    def test_no_diarization_makes_no_claim(self):
        assert speech_coverage([], []) == 1.0

    def test_no_words_is_zero_coverage(self):
        assert speech_coverage([], [DiarTurn(0.0, 100.0, "S0")]) == 0.0


class TestConfigMigration:
    """An existing config.json predates the quality slider and must land somewhere sensible."""

    def _load(self, tmp_path, monkeypatch, saved: dict) -> dict:
        from sinribe import config
        path = tmp_path / "config.json"
        path.write_text(json.dumps(saved))
        monkeypatch.setattr(config, "CONFIG_PATH", path)
        return config.load_config()

    def test_old_accurate_mode_maps_to_a_sequential_rung(self, tmp_path, monkeypatch):
        # The upgrade should change the wording, not the behaviour: "accurate" meant decoding
        # in order, so it must land on a rung that still does.
        from sinribe import presets
        cfg = self._load(tmp_path, monkeypatch, {"asr_mode": "accurate"})
        assert presets.decode_settings(cfg)["asr_mode"] == "accurate"

    def test_old_fast_mode_maps_to_a_batched_rung(self, tmp_path, monkeypatch):
        from sinribe import presets
        cfg = self._load(tmp_path, monkeypatch, {"asr_mode": "fast"})
        assert presets.decode_settings(cfg)["asr_mode"] == "fast"

    def test_legacy_large_v3_does_not_pin_the_model(self, tmp_path, monkeypatch):
        # It was the default and the only serious option, so it records no preference — kept as
        # a pin it would override the slider and hand an upgrading user none of the improvement.
        cfg = self._load(tmp_path, monkeypatch, {"asr_model": "large-v3", "asr_mode": "accurate"})
        assert cfg["asr_model"] == "auto"

    def test_deliberate_pin_survives(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, monkeypatch, {"asr_model": "large-v3", "speed_target": 4})
        assert cfg["asr_model"] == "large-v3"
        assert cfg["speed_target"] == 4

    def test_other_models_are_never_touched(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, monkeypatch, {"asr_model": "small", "asr_mode": "accurate"})
        assert cfg["asr_model"] == "small"

    def test_unknown_keys_are_still_dropped(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, monkeypatch, {"nonsense": 1, "asr_mode": "accurate"})
        assert "nonsense" not in cfg


class TestVotingProgress:
    """A voting rung runs the same decode up to nine times, and each one reports its own 0..1.

    Before `_subspan` those went straight to the bar, so on the 1x rung it filled and snapped
    back to the start nine times while the ETA promised the job was nearly over once per pass.
    """

    def _runner(self):
        r = Runner.__new__(Runner)
        seen: list[tuple[float, str, str]] = []
        r._on_progress = lambda f, p, d: seen.append((f, p, d))
        r._on_log = lambda m: None
        r._base = r._weight = r._stage_base = r._stage_weight = 0.0
        r._phase, r._t_phase = "", 0.0
        return r, seen

    def test_a_nine_pass_decode_never_moves_the_bar_backwards(self):
        r, seen = self._runner()
        r._stage("Transcribing", 0.3, 0.6)
        costs = [1.0, 0.4, 0.4, 0.16, 1.0, 1.0, 1.6, 4.0, 1.0]
        for i in range(len(costs)):
            with r._subspan(i, costs):
                for f in (0.0, 0.25, 0.5, 1.0):
                    r._frac(f)
        vals = [f for f, _, _ in seen]
        # Tolerance because consecutive passes meet at a boundary both of them compute, and the
        # two roundings can differ in the last bit. The bar is a permille integer; this is noise.
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:], strict=False))
        assert vals[0] == pytest.approx(0.3)
        assert vals[-1] == pytest.approx(0.9)

    def test_a_pass_gets_a_share_of_the_stage_proportional_to_its_cost(self):
        r, seen = self._runner()
        r._stage("Transcribing", 0.0, 1.0)
        costs = [1.0, 3.0]
        for i in range(2):
            with r._subspan(i, costs):
                r._frac(1.0)
        assert [f for f, _, _ in seen][1:] == pytest.approx([0.25, 1.0])

    def test_the_stage_span_is_restored_after_a_subspan(self):
        r, _ = self._runner()
        r._stage("Transcribing", 0.3, 0.6)
        with r._subspan(1, [1.0, 1.0]):
            pass
        assert (r._base, r._weight) == (0.3, 0.6)

    def test_the_estimate_counts_down_once_not_once_per_pass(self, monkeypatch):
        # Ten seconds in, one of four equal passes done: a quarter of the stage, so ~30s left.
        # The pre-fix code saw "pass 1 is 100% done" and said the stage was finished.
        r, seen = self._runner()
        r._stage("Transcribing", 0.0, 1.0)
        r._t_phase = time.time() - 10.0
        with r._subspan(0, [1.0] * 4):
            r._frac(1.0)
        assert "30s left" in seen[-1][2]


class TestSummaryStageSpans:
    """The summary is an optional stage, so the bar has to span 0..1 with and without it."""

    def _spans(self, **kw):
        from sinribe.pipeline import stage_spans
        return stage_spans(**kw)

    def test_the_bar_always_spans_exactly_one(self):
        for fetch in (True, False):
            for refine in (True, False):
                for summary in (True, False):
                    spans = self._spans(with_fetch=fetch, with_refine=refine,
                                        with_summary=summary)
                    base, width = spans[max(spans, key=lambda k: spans[k][0])]
                    assert base + width == pytest.approx(1.0), (fetch, refine, summary)

    def test_the_summary_is_absent_unless_asked_for(self):
        assert "summarise" not in self._spans(with_fetch=False, with_summary=False)
        assert "summarise" in self._spans(with_fetch=False, with_summary=True)

    def test_the_summary_comes_last(self):
        spans = self._spans(with_fetch=True, with_refine=True, with_summary=True)
        assert max(spans, key=lambda k: spans[k][0]) == "summarise"
        # Everything before it is squeezed but keeps its order.
        starts = [spans[k][0] for k in ("fetch", "decode", "diarize", "transcribe",
                                        "refine", "finish", "summarise")]
        assert starts == sorted(starts)

    def test_transcription_still_dominates(self):
        """A summary that ate half the bar would make a 4 h job look stalled at 50 %."""
        spans = self._spans(with_fetch=False, with_refine=True, with_summary=True)
        assert spans["transcribe"][1] > spans["summarise"][1]


class TestPassCost:
    def test_cheap_decodes_are_ranked_below_the_baseline(self):
        from sinribe.pipeline import _pass_cost
        base = _pass_cost({"asr_mode": "accurate", "beam_size": 5}, "large-v3")
        batched = _pass_cost({"asr_mode": "fast", "beam_size": 5}, "large-v3")
        turbo = _pass_cost({"asr_mode": "accurate", "beam_size": 5}, "large-v3-turbo-german")
        wide = _pass_cost({"asr_mode": "accurate", "beam_size": 20}, "large-v3")
        assert batched < base and turbo < base and wide > base

    def test_a_cost_is_never_zero(self):
        from sinribe.pipeline import _pass_cost
        # A zero would make one pass invisible on the bar and, with every pass at zero, divide
        # the stage by nothing at all.
        assert _pass_cost({"asr_mode": "fast", "beam_size": 0}, "large-v3-turbo-german") > 0


class TestVotingPlan:
    """What the passes of a voting rung actually run.

    Two passes that resolve to the same model and the same decode settings do not check each
    other — they double one opinion's weight and outvote the passes that disagree with it.
    """

    def _runner(self, cfg, monkeypatch):
        """A Runner whose decodes are recorded instead of executed."""
        from sinribe import pipeline as pl
        monkeypatch.setattr(pl, "model_available", lambda m: True)
        r = Runner(JobSpec(input_path=Path("in.wav"), output_dir=Path("out"), cfg=cfg))
        seen: list[tuple[str, str, Path]] = []

        def fake_once(wav, duration, cache, cfg=None, tag=""):
            d = presets.decode_settings(cfg)
            seen.append((presets.model_for(cfg), f"{d['asr_mode']}/{d['audio_filter']}", wav))
            return [], {}

        monkeypatch.setattr(r, "_transcribe_once", fake_once)
        monkeypatch.setattr(r, "_decode",
                            lambda info, cache, chain=None: Path(f"decoded-{chain or 'raw'}.wav"))
        return r, seen

    def test_the_recommended_rung_runs_its_three_distinct_passes(self, monkeypatch):
        cfg = {"speed_target": 10, "asr_model": "auto"}
        r, seen = self._runner(cfg, monkeypatch)
        r._transcribe(Path("base.wav"), 60.0, Path("cache"))
        assert [(m, d) for m, d, _ in seen] == [
            ("large-v3", "accurate/"), ("large-v3", "fast/"), ("large-v3-turbo-german",
                                                               "accurate/")]

    def test_a_pinned_model_does_not_leave_a_duplicate_pass_voting_twice(self, monkeypatch):
        # Pinning large-v3 in the dropdown collapses the "different model" pass onto the pivot.
        # Running it anyway would give the pivot two of three votes and cost GPU minutes to do it.
        cfg = {"speed_target": 10, "asr_model": "large-v3"}
        r, seen = self._runner(cfg, monkeypatch)
        r._transcribe(Path("base.wav"), 60.0, Path("cache"))
        assert [(m, d) for m, d, _ in seen] == [("large-v3", "accurate/"), ("large-v3", "fast/")]

    def test_a_pass_that_asks_for_filtered_audio_is_given_filtered_audio(self, monkeypatch):
        # The 5x rung's denoise pass used to decode the untouched WAV, because the filter was read
        # off the job config where only the passes ever set one. Its output came back
        # byte-identical to the pivot's: a duplicate ballot wearing a different name.
        cfg = {"speed_target": 5, "asr_model": "auto"}
        r, seen = self._runner(cfg, monkeypatch)
        r._transcribe(Path("base.wav"), 60.0, Path("cache"), object())
        by_filter = {d.split("/")[1]: wav for _, d, wav in seen}
        assert by_filter[""] == Path("base.wav")
        assert by_filter["denoise"] == Path("decoded-denoise.wav")

    def test_an_unfilterable_pass_is_dropped_rather_than_run_on_the_wrong_audio(self,
                                                                                monkeypatch):
        # No source audio to filter (an old caller that passes no MediaInfo): skip the pass. A
        # pass that cannot honour its own settings must not vote as if it had.
        cfg = {"speed_target": 5, "asr_model": "auto"}
        r, seen = self._runner(cfg, monkeypatch)
        r._transcribe(Path("base.wav"), 60.0, Path("cache"))
        assert all(d.split("/")[1] == "" for _, d, _ in seen)


class TestSpeedHonesty:
    """A run's reported speed must describe work it did, not work it read off disk.

    The bug this pins down: a re-run at the 10.7x rung reused all three cached passes, re-ran
    only diarization, and reported "29.5x realtime" — then fed that into `observed_rtf`, so the
    slider would have gone on promising 29.5x for a rung that runs at ten.
    """

    def test_a_run_that_decoded_every_pass_is_comparable_with_its_rung(self):
        result = {"asr_passes_ran": 3, "asr_passes_total": 3, "asr_realtime_factor": 10.4,
                  "reused_stages": ["diarization"]}
        assert presets.trustworthy_rtf(result) == 10.4
        assert "10.4× realtime transcribing" in presets.speed_note(result)

    def test_a_fully_cached_transcription_reports_no_speed(self):
        result = {"asr_passes_ran": 0, "asr_passes_total": 3, "asr_realtime_factor": 0.0,
                  "reused_stages": ["transcription pass0"], "realtime_factor": 29.5}
        assert presets.trustworthy_rtf(result) == 0.0
        assert "reused from cache" in presets.speed_note(result)
        assert "29.5" not in presets.speed_note(result)

    def test_a_partly_cached_run_says_how_much_it_actually_decoded(self):
        result = {"asr_passes_ran": 1, "asr_passes_total": 3, "asr_realtime_factor": 20.1,
                  "reused_stages": ["transcription pass1", "transcription pass2"]}
        assert presets.trustworthy_rtf(result) == 0.0
        assert "1 of 3 passes decoded" in presets.speed_note(result)

    def test_a_cached_run_cannot_teach_the_slider_a_speed(self):
        cfg = {"observed_rtf": {}}
        cached = {"asr_passes_ran": 0, "asr_passes_total": 3, "asr_realtime_factor": 0.0,
                  "reused_stages": ["transcription pass0"]}
        presets.record_rtf(cfg, 10, presets.trustworthy_rtf(cached))
        assert cfg["observed_rtf"] == {}
        cold = {"asr_passes_ran": 3, "asr_passes_total": 3, "asr_realtime_factor": 10.4,
                "reused_stages": []}
        presets.record_rtf(cfg, 10, presets.trustworthy_rtf(cold))
        assert cfg["observed_rtf"] == {"10": 10.4}

    def test_an_old_result_without_the_counters_makes_no_claim(self):
        assert presets.trustworthy_rtf({"realtime_factor": 29.5}) == 0.0
        assert presets.speed_note({"realtime_factor": 29.5}) == ""


class TestObservedRtfMigration:
    """Speeds recorded before the fix are a different measurement and must not be averaged in."""

    def _load(self, tmp_path, monkeypatch, user):
        from sinribe import config as cfgmod
        path = tmp_path / "config.json"
        path.write_text(json.dumps(user))
        monkeypatch.setattr(cfgmod, "CONFIG_PATH", path)
        return cfgmod.load_config()

    def test_end_to_end_figures_are_dropped_once(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, monkeypatch,
                         {"speed_target": 10, "observed_rtf": {"1": 2.12, "10": 29.5}})
        assert cfg["observed_rtf"] == {}
        assert cfg["observed_rtf_kind"] == "transcription"

    def test_figures_recorded_since_the_fix_survive(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, monkeypatch,
                         {"speed_target": 10, "observed_rtf": {"10": 10.4},
                          "observed_rtf_kind": "transcription"})
        assert cfg["observed_rtf"] == {"10": 10.4}

    def test_a_config_that_never_recorded_one_is_untouched(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, monkeypatch, {"speed_target": 10})
        assert cfg["observed_rtf"] == {}


class TestSummaryGuard:
    """Research that cannot happen is refused at second zero, not discovered in the output."""

    def test_research_without_a_searching_provider_is_a_setup_error(self, monkeypatch, tmp_path):
        from sinribe.summary import providers
        monkeypatch.setattr(providers.OllamaProvider, "probe", lambda self: None)
        cfg = {"web_summary": True, "summary_research": True, "summary_provider": "ollama"}
        runner = Runner(JobSpec(input_path=Path("missing.wav"), output_dir=tmp_path, cfg=cfg))
        with pytest.raises(StageError) as excinfo:
            runner.run()
        assert excinfo.value.kind == "setup"
        assert "cannot search the web" in str(excinfo.value)

    def test_an_auto_fallback_is_logged(self, monkeypatch, tmp_path):
        from sinribe.summary import providers

        def missing(self):
            raise providers.Unavailable("the `claude` command was not found")

        monkeypatch.setattr(providers.ClaudeCodeProvider, "probe", missing)
        monkeypatch.setattr(providers.ClaudeProvider, "probe", missing)
        monkeypatch.setattr(providers.OllamaProvider, "probe", lambda self: None)
        logs: list[str] = []
        cfg = {"web_summary": True, "summary_research": False, "summary_provider": "auto"}
        runner = Runner(JobSpec(input_path=None, output_dir=tmp_path, cfg=cfg),
                        on_log=logs.append)
        with pytest.raises(StageError):                  # no input: stops right after the probe
            runner.run()
        assert any("skipped: claude-code" in m for m in logs)
