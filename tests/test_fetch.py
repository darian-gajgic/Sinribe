"""URL handling: detection, filename sanitising, progress weighting, cache lookup.

Nothing here touches the network — the live download path is exercised manually against a real
link, but the pure logic around it is worth pinning down.
"""

from __future__ import annotations

from sinribe.fetch import cached_download, is_url, purge_old_downloads
from sinribe.pipeline import STAGE_SHARES, safe_stem, stage_spans


class TestIsUrl:
    def test_accepts_http_and_https(self):
        assert is_url("https://www.youtube.com/watch?v=abc")
        assert is_url("http://example.com/ep1.mp3")

    def test_tolerates_surrounding_whitespace(self):
        assert is_url("  https://youtu.be/abc  \n")

    def test_rejects_non_urls(self):
        for bad in ("", None, "not a url", "/home/user/a.mp3", "www.youtube.com/watch?v=a",
                    "ftp://example.com/x.mp3"):
            assert not is_url(bad)

    def test_rejects_url_with_spaces_inside(self):
        assert not is_url("https://example.com/a b")


class TestSafeStem:
    def test_strips_path_separators(self):
        assert "/" not in safe_stem("AC/DC interview")
        assert safe_stem("AC/DC interview") == "AC-DC interview"

    def test_strips_windows_reserved_characters(self):
        out = safe_stem('Ep 12: "Why?" <live> | part*2')
        for ch in '<>:"/\\|?*':
            assert ch not in out

    def test_collapses_whitespace_and_trims(self):
        assert safe_stem("  Ep   7 —  intro . ") == "Ep 7 — intro"

    def test_keeps_unicode(self):
        assert safe_stem("Rückkopplung über Systeme") == "Rückkopplung über Systeme"

    def test_truncates_long_titles(self):
        out = safe_stem("x" * 400)
        assert len(out) <= 120

    def test_never_returns_empty(self):
        for bad in ("", "   ", "///", "...", None):
            assert safe_stem(bad) == "transcript"


class TestStageSpans:
    def test_spans_cover_zero_to_one_without_fetch(self):
        spans = stage_spans(with_fetch=False)
        assert "fetch" not in spans
        start, width = spans["finish"]
        assert abs(start + width - 1.0) < 1e-9
        assert abs(spans["decode"][0]) < 1e-9

    def test_spans_cover_zero_to_one_with_fetch(self):
        spans = stage_spans(with_fetch=True)
        assert "fetch" in spans
        assert abs(spans["fetch"][0]) < 1e-9
        start, width = spans["finish"]
        assert abs(start + width - 1.0) < 1e-9

    def test_stages_are_contiguous_and_ordered(self):
        for with_fetch in (False, True):
            spans = stage_spans(with_fetch=with_fetch)
            acc = 0.0
            for name in STAGE_SHARES:
                if name not in spans:
                    continue
                start, width = spans[name]
                assert abs(start - acc) < 1e-9, f"{name} is not contiguous"
                assert width > 0
                acc += width
            assert abs(acc - 1.0) < 1e-9

    def test_refine_can_be_left_out(self):
        spans = stage_spans(with_fetch=False, with_refine=False)
        assert "refine" not in spans
        start, width = spans["finish"]
        assert abs(start + width - 1.0) < 1e-9

    def test_refine_is_included_by_default(self):
        assert "refine" in stage_spans(with_fetch=False)

    def test_fetch_shrinks_the_other_stages(self):
        without = stage_spans(with_fetch=False)
        with_ = stage_spans(with_fetch=True)
        assert with_["transcribe"][1] < without["transcribe"][1]


class TestDownloadCache:
    def test_finds_an_existing_download(self, tmp_path):
        f = tmp_path / "abc123.m4a"
        f.write_bytes(b"0" * 8192)
        assert cached_download(tmp_path, "abc123") == f

    def test_ignores_partial_downloads(self, tmp_path):
        (tmp_path / "abc123.m4a.part").write_bytes(b"0" * 8192)
        assert cached_download(tmp_path, "abc123") is None

    def test_ignores_truncated_stubs(self, tmp_path):
        (tmp_path / "abc123.m4a").write_bytes(b"0" * 10)
        assert cached_download(tmp_path, "abc123") is None

    def test_missing_dir_is_not_an_error(self, tmp_path):
        assert cached_download(tmp_path / "nope", "abc123") is None

    def test_does_not_match_a_different_id(self, tmp_path):
        (tmp_path / "other.m4a").write_bytes(b"0" * 8192)
        assert cached_download(tmp_path, "abc123") is None


class TestPurgeDownloads:
    def test_removes_only_old_files(self, tmp_path):
        import os
        import time
        old, new = tmp_path / "old.m4a", tmp_path / "new.m4a"
        old.write_bytes(b"0" * 8192)
        new.write_bytes(b"0" * 8192)
        past = time.time() - 40 * 86400
        os.utime(old, (past, past))
        assert purge_old_downloads(tmp_path, days=14) == 1
        assert not old.exists()
        assert new.exists()

    def test_missing_dir_is_not_an_error(self, tmp_path):
        assert purge_old_downloads(tmp_path / "nope", 14) == 0
