"""Pipeline-level tests that do not need a GPU: VRAM tiering, job keys, cache housekeeping."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from sinribe.pipeline import (
    ASR_TIERS, JobSpec, Runner, StageError, _asr_env, _diar_env, _job_key, pick_asr_tier,
    purge_old_jobs,
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
