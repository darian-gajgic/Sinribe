"""Pipeline-level tests that do not need a GPU: VRAM tiering, job keys, cache housekeeping."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

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
