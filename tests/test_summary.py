"""Summary tests: schema clamping, the citation guard, both renderers, and the pass orchestration.

No network and no model. The provider interface is one method wide on purpose, which makes the
interesting half of this feature -- what survives a bad answer, and what the page does with a
good one -- testable against a fake.
"""

from __future__ import annotations

import json
import re
import subprocess

import pytest

from sinribe.merge import Turn
from sinribe.render import webpage
from sinribe.summary import build, providers, schema


def _turns():
    return [
        Turn(speaker="Person 1", raw="SPEAKER_00", start=0.0, end=12.0,
             text="So the question everyone asks is when.\nMy median is 2029."),
        Turn(speaker="Person 2", raw="SPEAKER_01", start=12.5, end=30.0,
             text="And you left OpenAI over this.\nYou gave up two million dollars."),
    ]


RESULT = {
    "source": "https://www.youtube.com/watch?v=_g4l7YkDQwA",
    "title": "He Risked Everything To Warn You",
    "uploader": "The Diary Of A CEO",
    "duration": 7250.0,
    "language": "en",
}
STATS = {"Person 1": {"seconds": 10.0, "share": 0.4},
         "Person 2": {"seconds": 15.0, "share": 0.6}}


# --------------------------------------------------------------------------- schema

class TestSchema:
    def test_plan_clamps_and_drops_junk(self):
        plan = schema.plan({
            "title": "  A   headline\nwith newlines ",
            "kicker": "Interview",
            "subtitle": None,
            "takeaways": ["one", "", "two", "three", "four", "five", "six", "seven"],
            "why_it_matters": 42,
            "stats": [
                {"value": "70%", "label": "chance of catastrophe", "tone": "RED"},
                {"value": "", "label": "no value so dropped", "tone": "red"},
                {"value": "2029", "label": "median", "tone": "banana"},
                "not a dict",
            ],
            "sections": [{"heading": "One", "subheading": "s", "focus": "f",
                          "starts_at": "[0:12:30]"},
                         {"heading": "", "subheading": "dropped", "focus": ""}],
        })
        assert plan["title"] == "A headline with newlines"
        assert plan["subtitle"] == ""
        assert plan["why_it_matters"] == "42"
        assert len(plan["takeaways"]) == schema.MAX_TAKEAWAYS
        assert plan["sections"][0]["starts_at"] == "0:12:30"
        # Tone is case-folded onto the palette; an unknown one becomes neutral rather than a
        # CSS class that does not exist.
        assert [s["tone"] for s in plan["stats"]] == ["red", "neutral"]
        assert len(plan["sections"]) == 1

    def test_a_clamped_field_never_ends_mid_word(self):
        """A real page ended with "would not press a button that s". A hard slice did that."""
        long = ("He argues the ordering is the cruel part, because mass unemployment arrives "
                "after superintelligence exists rather than before it. So the public warning "
                "shot that would trigger regulation comes far too late to matter at all.")
        for limit in range(30, len(long) + 20, 7):
            out = schema._clip(long, limit)
            assert len(out) <= limit + 1               # +1 for the ellipsis character
            assert not out.endswith(" ")
            if out != long:
                # Either a clean sentence end, or a word boundary marked as cut.
                assert out.endswith((".", "\u2026")), out
                # Whatever it ends on, it is a whole word.
                assert long.startswith(out.rstrip("\u2026")), out

    def test_a_clamp_prefers_a_late_sentence_boundary(self):
        text = "First part is quite long and complete. Then a trailing fragment that is cut"
        assert schema._clip(text, 60) == "First part is quite long and complete."

    def test_the_gist_has_room_for_a_real_paragraph(self):
        """Clamped at 900 it was losing content, not padding: the model writes more than that."""
        para = "word " * 300
        assert len(schema.plan({"gist": para})["gist"]) > 1000

    def test_claims_are_kept_with_a_valid_section_or_none(self):
        plan = schema.plan({
            "sections": [{"heading": "A"}, {"heading": "B"}],
            "claims": [{"claim": "GPT-5 is the best model", "at": "0:05:00", "section": 2,
                        "time_sensitive": True},
                       {"claim": "Refers to a section that does not exist", "section": 9},
                       {"claim": "", "section": 1}],
        })
        assert [c["section"] for c in plan["claims"]] == [2, 0]
        assert plan["claims"][0]["at"] == "0:05:00"
        assert plan["claims"][0]["time_sensitive"] is True

    def test_plan_survives_total_garbage(self):
        assert schema.plan(None)["sections"] == []
        assert schema.plan({"sections": "nope"})["sections"] == []

    def test_section_falls_back_to_the_plan(self):
        built = schema.section({"blocks": [{"kind": "caveat", "paragraphs": ["because"]}]},
                               {"heading": "Planned", "subheading": "Sub",
                                "starts_at": "0:10:00"}, 3)
        assert built["id"] == "s3"
        assert built["heading"] == "Planned"
        assert built["subheading"] == "Sub"
        assert built["starts_at"] == "0:10:00"
        # A block with no heading gets its kind's name, so the page never shows a bare dot.
        assert built["blocks"][0]["heading"] == "Caveat"
        assert built["updates"] == []

    def test_section_drops_empty_blocks_and_bad_kinds(self):
        built = schema.section({"blocks": [
            {"kind": "forecast", "paragraphs": [], "bullets": []},
            {"kind": "nonsense", "paragraphs": ["kept"]},
        ]}, {}, 1)
        assert len(built["blocks"]) == 1
        assert built["blocks"][0]["kind"] == "example"

    def test_a_section_with_only_prose_counts_as_content(self):
        assert schema.has_content(schema.section({"paragraphs": ["p"]}, {}, 1))
        assert schema.has_content(schema.section({"key_points": ["k"]}, {}, 1))
        assert not schema.has_content(schema.section({"gist": "only a gist"}, {}, 1))

    def test_ragged_table_is_squared_off(self):
        built = schema.section({
            "paragraphs": ["p"],
            "table": {"caption": "Plans", "columns": ["Plan", "What", "Odds"],
                      "rows": [["A", "regulate"], ["D", "race", "likely", "extra"], []]},
        }, {}, 1)
        assert built["table"]["rows"] == [["A", "regulate", ""], ["D", "race", "likely"]]

    def test_one_column_table_is_not_a_table(self):
        built = schema.section({"paragraphs": ["p"],
                                "table": {"columns": ["Only"], "rows": [["x"]]}}, {}, 1)
        assert built["table"] is None

    def test_close_drops_invented_citations(self):
        allowed = {"https://ai-2027.com/about", "https://metr.org/time-horizons/"}
        close = schema.close({
            "glossary": [{"term": "AGI", "meaning": "general AI"}, {"term": "", "meaning": "x"}],
            "quiz": [{"question": "Why?", "answer": "Because."}, {"question": "No answer"}],
            "worth_watching": [{"at": "1:02:03", "title": "The demo", "why": "see it"},
                               {"at": "not a time", "title": "dropped"}],
            "sources": [
                {"group": "Primary", "links": [
                    {"label": "AI 2027", "url": "https://ai-2027.com/about/"},
                    {"label": "Invented", "url": "https://example.com/made-up"},
                    {"label": "Dupe", "url": "https://ai-2027.com/about"},
                ]},
                {"group": "All fake", "links": [{"label": "x", "url": "https://nope.test"}]},
            ],
            "footer": "Made from the transcript.",
        }, allowed)
        assert len(close["glossary"]) == 1 and len(close["quiz"]) == 1
        assert close["worth_watching"] == [{"at": "1:02:03", "title": "The demo", "why": "see it"}]
        assert len(close["sources"]) == 1
        assert [link["label"] for link in close["sources"][0]["links"]] == ["AI 2027"]

    def test_a_swallowed_field_is_cut_back_to_what_was_meant(self):
        """Measured on gemma3:4b: typographic quotes inside a JSON string make it swallow every
        key that should have followed, and the answer still parses."""
        bad = ('Cochetto\u2019s Timeline\u201d, \u201csubheading\u201d: \u201cUnderstanding '
               'the Date\u201d, \u201cfocus\u201d: \u201cDetail the dates')
        assert schema._s(bad) == "Cochetto\u2019s Timeline"
        assert schema._s('Real Heading", "subheading": "more') == "Real Heading"

    def test_the_leak_guard_leaves_ordinary_prose_alone(self):
        for text in ('He said \u201chello\u201d, and then left',
                     'Risk 1: loss of control, and why it matters',
                     'She wrote \u201cx\u201d, \u201cy\u201d: no comment',
                     'The plan, the timeline: both slipped',
                     # A string that was not cut keeps its punctuation, closing quote included:
                     # tidying unconditionally ate the last character off every quotation.
                     'He said "we will pause"',
                     'ends in a curly quote \u201cso\u201d',
                     'a trailing comma,'):
            assert schema._s(text) == text

    def test_a_swallowed_plan_still_yields_usable_headings(self):
        plan = schema.plan({"sections": [
            {"heading": 'Timelines", "subheading": "When it arrives", "focus": "dates',
             "subheading": "", "focus": ""}]})
        assert plan["sections"][0]["heading"] == "Timelines"

    def test_en_and_em_dashes_are_replaced(self):
        """Asked not to use them, gemma3:4b used them anyway in its first section. An
        instruction a model can decline is not a guarantee, so the rule is enforced here."""
        assert schema.dedash("all domains \u2013 and the need") == "all domains, and the need"
        assert schema.dedash("trends \u2014 essentially") == "trends, essentially"
        # Between digits a dash is a range, so it becomes a hyphen, not a comma.
        assert schema.dedash("2028\u20132030") == "2028-2030"
        assert schema.dedash("5 \u2013 10 years") == "5-10 years"
        # Never a comma straight after punctuation that already does the job.
        assert (schema.dedash("already punctuated, \u2014 and more")
                == "already punctuated, and more")
        assert schema.dedash("entity &ndash; form") == "entity, form"
        assert schema.dedash("plain hyphen-joined stays") == "plain hyphen-joined stays"

    def test_every_generated_field_is_dedashed(self):
        plan = schema.plan({"title": "A \u2014 B", "takeaways": ["one \u2013 two"],
                            "sections": [{"heading": "C \u2013 D", "subheading": "", "focus": ""}]})
        assert plan["title"] == "A, B"
        assert plan["takeaways"] == ["one, two"]
        assert plan["sections"][0]["heading"] == "C, D"

    def test_a_link_matches_despite_www_fragment_or_case(self):
        allowed = {"https://www.Anthropic.com/news/claude-5/"}
        assert schema._url_allowed("http://anthropic.com/news/claude-5#intro", allowed)
        assert not schema._url_allowed("https://anthropic.com/news/claude-6", allowed)

    def test_inline_links_off_the_list_are_unlinked_but_keep_their_label(self):
        """The Sources list was checked; links written inside the prose were not, so a section
        could cite a URL nothing ever returned and the page would render it live."""
        allowed = {"https://real.test/page"}
        brief = {"gist": "See [the paper](https://invented.test/x) and [this](https://real.test/page).",
                 "sections": [{"paragraphs": ["[x](https://nope.test)"],
                               "updates": [{"sources": [{"url": "https://kept.test"}]}]}]}
        out = schema.restrict_links(brief, allowed)
        assert out["gist"] == "See the paper and [this](https://real.test/page)."
        assert out["sections"][0]["paragraphs"] == ["x"]
        # `url` fields were filtered upstream and are not rewritten here.
        assert out["sections"][0]["updates"][0]["sources"][0]["url"] == "https://kept.test"

    def test_find_urls_strips_sentence_punctuation(self):
        found = schema.find_urls("see https://ai-2027.com/about, and https://metr.org/x.")
        assert found == {"https://ai-2027.com/about", "https://metr.org/x"}


class TestSalvage:
    """Recovering a truncated answer. Measured failure: gemma3:4b stopping mid-string."""

    def test_complete_json_is_unchanged(self):
        assert providers.salvage_json('{"a": 1, "b": ["x"]}') == {"a": 1, "b": ["x"]}

    def test_truncated_mid_string_keeps_the_complete_entries(self):
        cut = '{"topics": ["one", "two"], "claims": ["kept", "half a cla'
        assert providers.salvage_json(cut) == {"topics": ["one", "two"], "claims": ["kept"]}

    def test_truncated_after_a_comma(self):
        assert providers.salvage_json('{"a": ["x", "y"],') == {"a": ["x", "y"]}

    def test_truncated_inside_a_nested_object_keeps_the_fields_that_arrived(self):
        cut = ('{"sections": [{"heading": "One", "focus": "f"}, {"heading": "Two", '
               '"focus": "part')
        out = providers.salvage_json(cut)
        # The half-written object survives with the fields that completed; only the field that
        # was being written is lost. schema.plan then fills the gap from its own defaults.
        assert [s["heading"] for s in out["sections"]] == ["One", "Two"]
        assert "focus" not in out["sections"][1]

    def test_a_brace_inside_a_string_is_not_structure(self):
        assert providers.salvage_json('{"a": "it said {not real} here", "b": 2}') == {
            "a": "it said {not real} here", "b": 2}

    def test_an_escaped_quote_does_not_end_the_string(self):
        assert providers.salvage_json(r'{"a": "he said \"hi\"", "b": 1}') == {
            "a": 'he said "hi"', "b": 1}

    def test_numbers_and_literals_at_the_cut(self):
        # A cut number is indistinguishable from a complete one (25 may have been 2500), so it
        # is kept. Harmless here: every field in these schemas is a string or a container.
        assert providers.salvage_json('{"a": 1, "b": true, "c": 25') == {
            "a": 1, "b": True, "c": 25}
        assert providers.salvage_json('{"a": null, "b": fal') == {"a": None}

    def test_nothing_usable_returns_none(self):
        assert providers.salvage_json("") is None
        assert providers.salvage_json("not json at all") is None
        assert providers.salvage_json("[1, 2, 3]") is None          # arrays are not answers

    def test_the_ollama_provider_recovers_instead_of_failing(self, monkeypatch):
        provider = providers.OllamaProvider()
        logs: list[str] = []
        provider.on_log = logs.append
        monkeypatch.setattr(provider, "_generate",
                            lambda payload: '{"topics": ["kept"], "claims": ["half a cl')
        assert provider.ask_json("s", "p", schema.NOTES_SCHEMA) == {"topics": ["kept"]}
        assert any("cut off at" in m for m in logs)

    def test_an_unrecoverable_answer_still_raises(self, monkeypatch):
        provider = providers.OllamaProvider()
        monkeypatch.setattr(provider, "_generate", lambda payload: "I cannot do that")
        with pytest.raises(providers.SummaryError, match="unparseable"):
            provider.ask_json("s", "p", schema.NOTES_SCHEMA)


# --------------------------------------------------------------------------- renderer

SUMMARY = {
    "title": "Kokotajlo: no one is ready",
    "kicker": "Interview · 2 h",
    "subtitle": "A **bold** claim and a [link](https://ai-2027.com/about).",
    "gist": "The labs automate AI research first, and then progress gets fast.",
    "takeaways": ["**Timelines are short.** Median 2029.", "**Alignment lags.** It is unsolved."],
    "why_it_matters": "Use the calm years.",
    "stats": [{"value": "70%", "label": "catastrophe", "tone": "red"},
              {"value": "2029", "label": "median", "tone": "neutral"}],
    "check": {
        "overall": "Mostly current, one model claim is stale.",
        "checks": [
            {"claim": "GPT-4 is the frontier model", "at": "0:10:00", "section": 1,
             "status": "outdated", "now": "Several newer frontier models exist.",
             "as_of": "2026-09", "sources": [{"label": "News", "url": "https://news.test/a"}]},
            {"claim": "AGI by 2027", "at": "0:20:00", "section": 1, "status": "open",
             "now": "Not yet resolved.", "as_of": "2026-10",
             "sources": [{"label": "Tracker", "url": "https://track.test"}]},
            {"claim": "METR exists", "at": "", "section": 0, "status": "confirmed", "now": "",
             "as_of": "", "sources": [{"label": "METR", "url": "https://metr.org"}]},
        ],
        "developments": [{"headline": "A new model shipped", "detail": "It is better.",
                          "date": "2026-09-01", "section": 1,
                          "sources": [{"label": "Blog", "url": "https://blog.test/x"}]}],
    },
    "sections": [{
        "id": "s1", "number": 1, "heading": "Timelines", "subheading": "When",
        "starts_at": "0:05:00", "gist": "The median is 2029.",
        "paragraphs": ["He walks through the forecast.", "Then the reasons."],
        "key_points": ["Median 2029", "Wide error bars"],
        "blocks": [
            {"kind": "forecast", "heading": "Median", "paragraphs": ["About 2029."],
             "bullets": ["Scaling holds", "Revenue grows"]},
            {"kind": "caveat", "heading": "If wrong", "paragraphs": ["It slips."], "bullets": []},
        ],
        "quotes": [{"text": "My median is 2029.", "who": "Person 1", "at": "0:00:06"}],
        "table": {"caption": "Plans", "columns": ["Plan", "Odds"],
                  "rows": [["A", "unlikely"], ["D", "likely"]]},
        "recommendation": {"paragraphs": ["Calibrate the shape, not the date."],
                           "bullets": ["Track METR"]},
        "updates": [{"claim": "GPT-4 is the frontier model", "at": "0:10:00", "section": 1,
                     "status": "outdated", "now": "Several newer frontier models exist.",
                     "as_of": "2026-09",
                     "sources": [{"label": "News", "url": "https://news.test/a"}]}],
    }],
    "glossary": [{"term": "AGI", "meaning": "AI as capable as people at most work."}],
    "quiz": [{"question": "Why does research automation matter?", "answer": "It compounds."}],
    "worth_watching": [{"at": "1:02:03", "title": "The scenario", "why": "Hear it told."}],
    "sources": [{"group": "Primary", "links": [
        {"label": "AI 2027", "url": "https://ai-2027.com/about"}]}],
    "footer": "Blocks marked (derived) are inference.",
    "meta": {"provider": "claude", "model": "claude-opus-5-5", "label": "Claude claude-opus-5-5",
             "research": True, "checked_on": "2026-10-08", "language": "en",
             "source": "https://www.youtube.com/watch?v=_g4l7YkDQwA",
             "media_title": "He Risked Everything", "duration": 7250.0, "uploader": "DOAC",
             "published": "2026-08-03", "age_days": 66,
             "speakers": ["Person 1", "Person 2"], "generated_at": "2026-09-16 12:00"},
}


class TestHtml:
    def test_document_shape(self):
        page = webpage.render_html(SUMMARY)
        assert page.startswith("<!doctype html>")
        assert page.rstrip().endswith("</html>")
        assert '<html lang="en">' in page
        assert "<title>Kokotajlo: no one is ready</title>" in page
        # Self-contained: no request leaves the page.
        assert "<script" not in page
        assert "https://cdn" not in page and "fonts.googleapis" not in page

    def test_the_page_reads_in_learning_order(self):
        """One-minute version, then what changed, then terms, then the detail, then the quiz."""
        page = webpage.render_html(SUMMARY)
        order = ['id="minute"', 'id="changed"', 'id="terms"', 'id="s1"', 'id="watch"',
                 'id="quiz"', 'id="sources"']
        positions = [page.index(marker) for marker in order]
        assert positions == sorted(positions)

    def test_toc_anchors_match_sections(self):
        page = webpage.render_html(SUMMARY)
        assert '<a href="#s1">' in page and '<section id="s1">' in page
        for anchor in ("minute", "changed", "terms", "watch", "quiz", "sources"):
            assert f'<a href="#{anchor}">' in page
        # A section the web check flagged is marked in the contents.
        assert 'class="flag"' in page

    def test_content_renders_with_its_classes(self):
        page = webpage.render_html(SUMMARY)
        assert 'class="block b-forecast' in page
        assert 'class="block b-caveat' in page
        assert 'class="stat red"' in page
        assert 'class="stat"' in page                # neutral carries no tone class
        assert '<div class="reco"><h4>What to do with this</h4>' in page
        assert '<div class="keys"><h4>Key points</h4>' in page
        assert "<th>Plan</th>" in page and "<td>unlikely</td>" in page
        assert "<dt>AGI</dt>" in page
        assert "<details><summary>Why does research automation matter?</summary>" in page
        assert '<a href="https://ai-2027.com/about" rel="noopener noreferrer">AI 2027</a>' in page

    def test_timestamps_jump_into_the_video(self):
        page = webpage.render_html(SUMMARY)
        assert 'href="https://www.youtube.com/watch?v=_g4l7YkDQwA&amp;t=6s"' in page     # quote
        assert 'href="https://www.youtube.com/watch?v=_g4l7YkDQwA&amp;t=300s"' in page   # section
        assert 'href="https://www.youtube.com/watch?v=_g4l7YkDQwA&amp;t=3723s"' in page  # watch

    def test_timestamps_stay_plain_text_for_a_local_file(self):
        local = {**SUMMARY, "meta": {**SUMMARY["meta"], "source": "/home/me/talk.m4a"}}
        page = webpage.render_html(local)
        assert '<span class="ts">0:00:06</span>' in page
        assert "&amp;t=" not in page

    def test_outdated_claims_are_flagged_where_they_appear(self):
        page = webpage.render_html(SUMMARY)
        # In the one-minute version, so a reader who stops there is warned.
        minute = page[page.index('id="minute"'):page.index('id="changed"')]
        assert "Heads-up from the web check" in minute and "GPT-4 is the frontier model" in minute
        # In the section the claim belongs to.
        section = page[page.index('<section id="s1">'):]
        assert 'class="update outdated"' in section
        assert "Several newer frontier models exist." in section
        # And in the freshness line at the top, with the counts.
        assert "Checked against the web on 2026-10-08:" in page
        assert "1 outdated" in page and "1 still open" in page and "1 still accurate" in page

    def test_freshness_says_when_no_check_ran(self):
        unchecked = {**SUMMARY, "check": None,
                     "meta": {**SUMMARY["meta"], "research": False, "checked_on": ""}}
        page = webpage.render_html(unchecked)
        assert "Not checked against the web" in page and "about 2 months ago" in page
        failed = {**unchecked, "meta": {**unchecked["meta"], "research_error": "timed out"}}
        assert "The web check did not run:</strong> timed out" in webpage.render_html(failed)

    def test_reading_time_is_shown(self):
        page = webpage.render_html(SUMMARY)
        assert "<b>1 minute</b> core idea" in page
        assert "min</b> full summary" in page
        assert "instead of <b>2h 00m</b> listening" in page
        # Never claimed for a recording shorter than its own brief.
        short = {**SUMMARY, "meta": {**SUMMARY["meta"], "duration": 60.0}}
        assert "listening</span>" not in webpage.render_html(short)

    def test_provenance_is_on_the_page(self):
        page = webpage.render_html({**SUMMARY,
                                    "meta": {**SUMMARY["meta"], "cost_usd": 1.234}})
        assert "Claude claude-opus-5-5" in page
        assert "claims checked against the web on 2026-10-08" in page
        assert "$1.23 of usage" in page

    def test_an_auto_fallback_is_stated_in_the_footer(self):
        page = webpage.render_html({**SUMMARY, "meta": {
            **SUMMARY["meta"], "skipped": ["claude-code: the `claude` command was not found"]}})
        assert "The preferred writer was not available" in page
        assert "was not found" in page

    def test_inline_markup(self):
        assert webpage.inline("a **bold** word") == "a <strong>bold</strong> word"
        assert webpage.inline("an *italic* word") == "an <em>italic</em> word"
        assert webpage.inline("call `fn()`") == "call <code>fn()</code>"
        assert webpage.inline("[label](https://x.test)") == (
            '<a href="https://x.test" rel="noopener noreferrer">label</a>')

    def test_model_text_cannot_inject_markup(self):
        hostile = {**SUMMARY, "title": '<script>alert(1)</script>',
                   "takeaways": ['<img src=x onerror="alert(1)">'],
                   "sources": [{"group": "g", "links": [
                       {"label": "js", "url": "javascript:alert(1)"},
                       {"label": "data", "url": "data:text/html,<script>"}]}]}
        page = webpage.render_html(hostile)
        # Escaped, so it renders as visible text rather than as markup.
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
        assert "<img" not in page
        assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in page
        # A non-http target is dropped entirely, not rendered as a dead link.
        assert "javascript:" not in page and "data:text/html" not in page

    def test_link_markup_only_accepts_http_and_anchors(self):
        # A non-http target is left as literal text: no anchor is produced at all.
        assert webpage.inline("[x](javascript:alert(1))") == "[x](javascript:alert(1))"
        assert webpage.inline("[x](#s2)") == '<a href="#s2" rel="noopener noreferrer">x</a>'

    def test_markup_is_stripped_from_the_title_tag(self):
        page = webpage.render_html({**SUMMARY, "title": "The **real** [story](https://x.test)"})
        assert "<title>The real story</title>" in page

    def test_no_en_or_em_dashes_reach_the_page(self):
        """The same character class as `grep -P '[\\x{2012}-\\x{2015}\\x{2212}]|&[mn]dash;'`."""
        import re as _re
        dashed = schema.plan({"title": "A — B", "kicker": "k – k",
                              "subtitle": "s &mdash; s", "takeaways": ["t – t"],
                              "gist": "x — y", "stats": [], "sections": []})
        page = webpage.render_html({**dashed, "sections": [], "meta": {}})
        assert not _re.search(r"[‒-―−]|&[mn]dash;", page)
        assert not _re.search(r"[‒-―−]|&[mn]dash;", webpage.render_html(SUMMARY))

    def test_survives_a_minimal_summary(self):
        page = webpage.render_html({"title": "Bare", "sections": [], "meta": {}})
        assert "<h1>Bare</h1>" in page
        assert page.rstrip().endswith("</html>")


class TestHelpers:
    def test_youtube_ids(self):
        assert webpage.youtube_id("https://www.youtube.com/watch?v=qOvc9IUKEIc&list=x") == \
            "qOvc9IUKEIc"
        assert webpage.youtube_id("https://youtu.be/qOvc9IUKEIc?t=4") == "qOvc9IUKEIc"
        assert webpage.youtube_id("https://youtube.com/live/qOvc9IUKEIc") == "qOvc9IUKEIc"
        assert webpage.youtube_id("https://vimeo.com/123456") == ""
        assert webpage.youtube_id("not a url") == ""

    def test_seconds(self):
        assert webpage.seconds("1:02:03") == 3723
        assert webpage.seconds("[12:30]") == 750
        assert webpage.seconds("soon") is None

    def test_age_wording(self):
        assert webpage._age({"age_days": 0}) == "today"
        assert webpage._age({"age_days": 66}) == "about 2 months ago"
        assert webpage._age({"age_days": 12}) == "12 days ago"
        assert webpage._age({}) == ""


class TestMarkdown:
    def test_structure(self):
        md = webpage.render_markdown(SUMMARY)
        assert md.startswith("# Kokotajlo: no one is ready\n")
        assert "**Source:** https://www.youtube.com/watch?v=_g4l7YkDQwA" in md
        assert "**Published:** 2026-08-03" in md
        assert "## The 1-minute version" in md
        assert "- **70%** catastrophe" in md
        assert "Heads-up from the web check" in md
        assert "## What has changed since it was published" in md
        assert "- **Outdated:** GPT-4 is the frontier model" in md
        assert "## Key terms" in md
        assert md.index("## Key terms") < md.index("## The full summary")
        assert "## 1. Timelines [0:05:00](https://www.youtube.com/watch?v=_g4l7YkDQwA&t=300s)" in md
        assert "**Median:**" in md
        assert '> "My median is 2029."' in md
        assert "| Plan | Odds |" in md
        assert "**What to do with this:**" in md
        assert "## Test yourself" in md and "*Answer:* It compounds." in md
        assert "## Worth watching in the original" in md
        assert "**Primary:** [AI 2027](https://ai-2027.com/about)" in md

    def test_no_en_or_em_dashes_reach_the_markdown(self):
        """A literal em dash in the quote attribution put one in every summary's .md file."""
        import re as _re
        md = webpage.render_markdown(SUMMARY)
        assert not _re.search(r"[‒-―−]|&[mn]dash;", md)

    def test_survives_a_minimal_summary(self):
        assert webpage.render_markdown({"title": "Bare", "meta": {}}).startswith("# Bare")


class TestVerdicts:
    ALLOWED = {"https://news.test/a", "https://docs.test/b"}

    def test_a_verdict_without_a_returned_source_is_downgraded(self):
        """'Outdated' with no link is the model's opinion presented as a finding."""
        out = schema.verdicts({"checks": [
            {"claim": "X is the newest model", "at": "0:01:00", "section": 1,
             "status": "outdated", "now": "Y is newer.", "as_of": "2026-09",
             "sources": [{"label": "made up", "url": "https://invented.test"}]},
        ]}, self.ALLOWED, 3)
        check = out["checks"][0]
        assert check["status"] == "unverified"
        assert check["sources"] == []
        assert "No source the search returned" in check["now"] and "Y is newer." in check["now"]

    def test_backed_verdicts_keep_their_status_and_sort_worst_first(self):
        out = schema.verdicts({"checks": [
            {"claim": "a", "status": "confirmed", "section": 1,
             "sources": [{"label": "d", "url": "https://docs.test/b/"}]},
            {"claim": "b", "status": "OUTDATED", "section": 2,
             "sources": [{"label": "n", "url": "https://news.test/a"}]},
            {"claim": "c", "status": "made-up-status", "section": 99, "sources": []},
        ]}, self.ALLOWED, 3)
        assert [c["status"] for c in out["checks"]] == ["outdated", "confirmed", "unverified"]
        assert [c["section"] for c in out["checks"]] == [2, 1, 0]

    def test_a_development_needs_a_source(self):
        out = schema.verdicts({"developments": [
            {"headline": "Backed", "detail": "d", "date": "2026-09-01", "section": 1,
             "sources": [{"label": "n", "url": "https://news.test/a"}]},
            {"headline": "Unbacked", "detail": "d", "date": "", "section": 1, "sources": []},
        ]}, self.ALLOWED, 3)
        assert [d["headline"] for d in out["developments"]] == ["Backed"]

    def test_a_date_field_must_hold_a_date(self):
        out = schema.verdicts({"developments": [
            {"headline": "h", "detail": "d", "date": "Various; version numbers from docs and",
             "section": 1, "sources": [{"label": "n", "url": "https://news.test/a"}]},
            {"headline": "h2", "detail": "d", "date": "2026-09-01", "section": 1,
             "sources": [{"label": "n", "url": "https://news.test/a"}]}]}, self.ALLOWED, 3)
        assert [d["date"] for d in out["developments"]] == ["", "2026-09-01"]

    def test_garbage_is_survivable(self):
        assert schema.verdicts(None, set(), 0) == {"overall": "", "checks": [],
                                                    "developments": []}


# --------------------------------------------------------------------------- orchestration

class FakeProvider:
    """Answers each pass with something schema-shaped, and records what it was asked."""

    name = "fake"
    label = "Fake model"
    supports_research = True

    def __init__(self, context_chars: int = 1_000_000, sections: int = 3):
        self.context_chars = context_chars
        self.sections = sections
        self.calls: list[tuple[str, str, str]] = []   # (kind, prompt, cache_prefix)
        self.research_calls = 0
        self.research_prompts: list[str] = []

    def ask_json(self, system, prompt, schema_, *, max_tokens=8000, cache_prefix=""):
        if schema_ is schema.NOTES_SCHEMA:
            self.calls.append(("notes", prompt, cache_prefix))
            return {"topics": ["timelines"], "claims": ["median 2029"],
                    "numbers": ["2029"], "quotes": ["My median is 2029."]}
        if schema_ in (schema.PLAN_SCHEMA, schema.PLAN_WITH_CLAIMS_SCHEMA):
            self.calls.append(("plan", prompt, cache_prefix))
            return {
                "title": "Fake title", "kicker": "k", "subtitle": "s",
                "gist": "The whole idea.", "takeaways": ["**a** one", "**b** two"],
                "why_it_matters": "t",
                "stats": [{"value": "70%", "label": "risk", "tone": "red"}],
                "sections": [{"heading": f"H{i}", "subheading": f"sub{i}", "focus": f"f{i}",
                              "starts_at": f"0:0{i}:00"} for i in range(self.sections)],
                "claims": [{"claim": "Model X is the best", "at": "0:00:06", "section": 1,
                            "time_sensitive": True}],
            }
        if schema_ is schema.SECTION_SCHEMA:
            self.calls.append(("section", prompt, cache_prefix))
            return {"heading": "", "subheading": "", "gist": "The point.",
                    "paragraphs": ["p, see [a paper](https://invented.test/paper)"],
                    "key_points": ["k"],
                    "blocks": [{"kind": "example", "heading": "Because",
                                "paragraphs": ["p"], "bullets": []}],
                    "quotes": [], "table": {}, "recommendation": {"paragraphs": ["do it"]}}
        if schema_ is schema.VERDICTS_SCHEMA:
            self.calls.append(("verify", prompt, cache_prefix))
            return {"overall": "One claim is stale.",
                    "checks": [{"claim": "Model X is the best", "at": "0:00:06", "section": 1,
                                "status": "outdated", "now": "Model Y is better.",
                                "as_of": "2026-09",
                                "sources": [{"label": "Found", "url": "https://found.test/page"}]},
                               {"claim": "Unbacked", "at": "", "section": 2,
                                "status": "incorrect", "now": "says who",
                                "sources": [{"label": "x", "url": "https://only-in-prose.test"}]}],
                    "developments": []}
        self.calls.append(("close", prompt, cache_prefix))
        return {"glossary": [{"term": "T", "meaning": "m"}],
                "quiz": [{"question": "q?", "answer": "a."}],
                "worth_watching": [],
                "sources": [{"group": "Primary", "links": [
                    {"label": "found", "url": "https://found.test/page"},
                    {"label": "prose only", "url": "https://only-in-prose.test"}]}],
                "footer": "footer"}

    def ask_text(self, system, prompt, *, max_tokens=8000):
        return "text"

    def research(self, system, prompt, *, max_tokens=32000):
        self.research_calls += 1
        self.research_prompts.append(prompt)
        # The report names a URL its tools never returned; only the tool list counts.
        return ("CLAIM 1: outdated. SOURCES: https://found.test/page https://only-in-prose.test",
                [{"url": "https://found.test/page", "title": "Found", "kind": "search"}])


class VerboseProvider(FakeProvider):
    """Returns notes long enough to need folding, each naming the parts it came from."""

    def ask_json(self, system, prompt, schema_, *, max_tokens=8000, cache_prefix=""):
        if schema_ is schema.NOTES_SCHEMA:
            self.calls.append(("notes", prompt, cache_prefix))
            tags = sorted(set(re.findall(r"MARK\d+", prompt)))
            return {"topics": tags or ["untagged"], "claims": ["x" * 200],
                    "numbers": ["1"], "quotes": ["q"]}
        return super().ask_json(system, prompt, schema_, max_tokens=max_tokens,
                                cache_prefix=cache_prefix)


def _marked_turns(count: int = 12):
    """Turns carrying a unique marker each, so it is visible which survived a fold."""
    return [Turn(speaker="P", raw="S", start=float(i * 10), end=float(i * 10 + 9),
                 text=f"MARK{i} " + "filler words here " * 6) for i in range(count)]


class TestBuild:
    def test_full_context_path(self):
        provider = FakeProvider()
        out = build.summarize(_turns(), STATS, RESULT, provider)

        kinds = [c[0] for c in provider.calls]
        assert kinds == ["plan"] + ["section"] * 3 + ["close"]
        assert out["title"] == "Fake title"
        assert [s["number"] for s in out["sections"]] == [1, 2, 3]
        # The section fill lost its own heading; the plan's is used instead of "Section 2".
        assert out["sections"][1]["heading"] == "H1"
        assert out["meta"]["label"] == "Fake model"
        assert out["meta"]["research"] is False

    def test_transcript_is_cached_not_repeated(self):
        """The transcript goes in the cached prefix, so the per-section prompts stay small."""
        provider = FakeProvider()
        build.summarize(_turns(), STATS, RESULT, provider)
        prefixes = {prefix for _, _, prefix in provider.calls}
        assert len(prefixes) == 1                       # byte-identical across every call
        prefix = prefixes.pop()
        assert prefix.startswith("TRANSCRIPT:")
        assert "My median is 2029." in prefix
        for kind, prompt, _ in provider.calls:
            assert "My median is 2029." not in prompt, f"{kind} duplicated the transcript"

    def test_small_context_maps_the_transcript_to_notes_first(self):
        provider = FakeProvider(context_chars=90, sections=3)
        build.summarize(_turns(), STATS, RESULT, provider)
        kinds = [c[0] for c in provider.calls]
        assert kinds.count("notes") >= 2
        assert kinds[kinds.count("notes")] == "plan"
        prefix = provider.calls[-1][2]
        assert prefix.startswith("TRANSCRIPT NOTES:")
        assert "median 2029" in prefix

    def test_long_notes_are_folded_not_truncated(self):
        """The failure this guards: a digest that outgrows the window and loses the recording's
        tail, so a two-hour podcast gets a summary of its first half."""
        provider = VerboseProvider(context_chars=160)
        logs: list[str] = []
        build.summarize(_marked_turns(), STATS, RESULT, provider, on_log=logs.append)

        prefix = provider.calls[-1][2]
        assert prefix.startswith("TRANSCRIPT NOTES:")
        assert len(prefix) <= build._digest_budget(provider) + len("TRANSCRIPT NOTES:\n")
        assert any("merging" in m for m in logs)
        # The last part of the recording survived the fold, which truncation would have cut.
        assert "MARK11" in prefix
        assert "MARK0" in prefix

    def test_a_digest_that_already_fits_is_not_folded(self):
        provider = FakeProvider(context_chars=90)
        logs: list[str] = []
        build.summarize(_turns(), STATS, RESULT, provider, on_log=logs.append)
        assert not any("merging" in m for m in logs)

    def test_research_pass_runs_and_its_links_are_allowed(self):
        provider = FakeProvider()
        out = build.summarize(_turns(), STATS, RESULT, provider, research=True)
        assert provider.research_calls == 1
        kinds = [c[0] for c in provider.calls]
        assert kinds == ["plan", "verify"] + ["section"] * 3 + ["close"]
        assert out["meta"]["research"] is True
        assert out["meta"]["checked_on"]
        urls = [link["url"] for g in out["sources"] for link in g["links"]]
        assert "https://found.test/page" in urls
        # The recording itself is always linked, whatever the model returned.
        assert RESULT["source"] in urls

    def test_sources_come_from_what_the_tools_returned_not_the_report(self):
        """The old guard trusted any URL in the research prose, so a URL the model wrote down
        without ever opening it passed. Only tool results count now."""
        out = build.summarize(_turns(), STATS, RESULT, FakeProvider(), research=True)
        urls = [link["url"] for g in out["sources"] for link in g["links"]]
        assert "https://only-in-prose.test" not in urls
        # A verdict whose only source is prose-only is downgraded, not shown as a finding.
        statuses = {c["claim"]: c["status"] for c in out["check"]["checks"]}
        assert statuses == {"Model X is the best": "outdated", "Unbacked": "unverified"}

    def test_the_claims_from_the_plan_are_what_gets_checked(self):
        provider = FakeProvider()
        build.summarize(_turns(), STATS, RESULT, provider, research=True)
        prompt = provider.research_prompts[0]
        assert "1. Model X is the best [0:00:06] (section 1) (time-sensitive)" in prompt
        # The research call never carries the transcript itself.
        assert "My median is 2029." not in prompt
        plan_prompt = next(p for k, p, _ in provider.calls if k == "plan")
        assert "- claims: up to 16" in plan_prompt

    def test_an_outdated_claim_is_attached_to_its_section_and_its_prompt(self):
        provider = FakeProvider()
        out = build.summarize(_turns(), STATS, RESULT, provider, research=True)
        assert [c["claim"] for c in out["sections"][0]["updates"]] == ["Model X is the best"]
        assert out["sections"][1]["updates"] == []          # the unbacked one is unverified
        first_section = next(p for k, p, _ in provider.calls if k == "section")
        assert "WEB CHECK FOR THIS SECTION" in first_section
        assert "OUTDATED: \"Model X is the best\"" in first_section

    def test_a_failed_web_check_still_writes_the_brief(self):
        """The research call was unguarded: one CLI timeout there lost the whole page."""
        class Offline(FakeProvider):
            def research(self, system, prompt, *, max_tokens=32000):
                raise providers.SummaryError("`claude -p` did not finish within 900s")

        logs: list[str] = []
        out = build.summarize(_turns(), STATS, RESULT, Offline(), research=True,
                              on_log=logs.append)
        assert out["sections"]
        assert out["check"] is None
        assert out["meta"]["research"] is False
        assert "did not finish" in out["meta"]["research_error"]
        assert any("web check failed" in m for m in logs)
        assert "The web check did not run" in webpage.render_html(out)

    def test_inline_links_in_sections_are_held_to_the_allow_list(self):
        out = build.summarize(_turns(), STATS, RESULT, FakeProvider(), research=True)
        assert out["sections"][0]["paragraphs"] == ["p, see a paper"]

    def test_the_publication_date_and_uploader_material_reach_the_plan(self):
        provider = FakeProvider()
        result = {**RESULT, "published": "2026-08-03",
                  "description": "Links: https://ai-2027.com/about",
                  "chapters": [{"title": "Intro", "start": 0.0},
                               {"title": "Timelines", "start": 754.0}]}
        out = build.summarize(_turns(), STATS, result, provider)
        plan_prompt = next(p for k, p, _ in provider.calls if k == "plan")
        assert "Published on: 2026-08-03 (" in plan_prompt
        assert "[00:12:34] Timelines" in plan_prompt
        assert "https://ai-2027.com/about" in plan_prompt
        assert out["meta"]["published"] == "2026-08-03"
        assert isinstance(out["meta"]["age_days"], int)

    def test_a_link_from_the_description_is_allowed_in_sources(self):
        class Cites(FakeProvider):
            def ask_json(self, system, prompt, schema_, *, max_tokens=8000, cache_prefix=""):
                out = super().ask_json(system, prompt, schema_, max_tokens=max_tokens,
                                       cache_prefix=cache_prefix)
                if schema_ is schema.CLOSE_SCHEMA:
                    out["sources"] = [{"group": "Mentioned", "links": [
                        {"label": "AI 2027", "url": "https://ai-2027.com/about"}]}]
                return out

        out = build.summarize(_turns(), STATS,
                              {**RESULT, "description": "see https://ai-2027.com/about"},
                              Cites())
        urls = [link["url"] for g in out["sources"] for link in g["links"]]
        assert "https://ai-2027.com/about" in urls

    def test_without_research_a_link_the_model_invented_is_dropped(self):
        provider = FakeProvider()
        out = build.summarize(_turns(), STATS, RESULT, provider)
        urls = [link["url"] for g in out["sources"] for link in g["links"]]
        assert "https://found.test/page" not in urls
        assert urls == [RESULT["source"]]

    def test_research_is_skipped_when_the_provider_cannot_search(self):
        provider = FakeProvider()
        provider.supports_research = False
        logs: list[str] = []
        out = build.summarize(_turns(), STATS, RESULT, provider,
                              research=True, on_log=logs.append)
        assert provider.research_calls == 0
        assert out["meta"]["research"] is False
        assert any("cannot search" in m for m in logs)

    def test_progress_is_monotonic_and_completes(self):
        seen: list[float] = []
        build.summarize(_turns(), STATS, RESULT, FakeProvider(), on_progress=seen.append)
        assert seen == sorted(seen)
        assert seen[-1] == pytest.approx(1.0)
        assert all(0.0 <= f <= 1.0 for f in seen)

    def test_progress_keeps_moving_while_notes_are_folded(self):
        """Folding is several more model calls; a bar frozen through them reads as a hang."""
        seen: list[float] = []
        build.summarize(_marked_turns(), STATS, RESULT, VerboseProvider(context_chars=160),
                        on_progress=seen.append)
        assert seen == sorted(seen)
        notes_span = [f for f in seen if f < build.PASS_WEIGHTS["notes"]]
        # More progress reports than there were chunks means the fold reported too.
        assert len(notes_span) > 12
        assert seen[-1] == pytest.approx(1.0)

    def test_cancellation_stops_between_passes(self):
        provider = FakeProvider()
        with pytest.raises(build.Cancelled):
            build.summarize(_turns(), STATS, RESULT, provider, should_cancel=lambda: True)
        assert provider.calls == []

    def test_one_failed_section_does_not_lose_the_page(self):
        """Measured failure mode: the local model overran its output window on section 2 of 4."""
        class Flaky(FakeProvider):
            def ask_json(self, system, prompt, schema_, *, max_tokens=8000, cache_prefix=""):
                if schema_ is schema.SECTION_SCHEMA and "Write section 2," in prompt:
                    raise providers.SummaryError("Ollama returned unparseable JSON")
                return super().ask_json(system, prompt, schema_, max_tokens=max_tokens,
                                        cache_prefix=cache_prefix)

        logs: list[str] = []
        out = build.summarize(_turns(), STATS, RESULT, Flaky(sections=4), on_log=logs.append)
        assert [s["heading"] for s in out["sections"]] == ["H0", "H2", "H3"]
        assert any("section 2 failed and was skipped" in m for m in logs)
        # The surviving sections are renumbered nowhere: their ids still match their TOC entries.
        page = webpage.render_html(out)
        for section in out["sections"]:
            assert f'<a href="#{section["id"]}">' in page

    def test_every_section_failing_is_an_error(self):
        class Dead(FakeProvider):
            def ask_json(self, system, prompt, schema_, *, max_tokens=8000, cache_prefix=""):
                if schema_ is schema.SECTION_SCHEMA:
                    raise providers.SummaryError("nope")
                return super().ask_json(system, prompt, schema_, max_tokens=max_tokens,
                                        cache_prefix=cache_prefix)

        with pytest.raises(providers.SummaryError, match="no usable sections"):
            build.summarize(_turns(), STATS, RESULT, Dead())

    def test_a_small_context_provider_is_asked_for_less(self):
        big, small = FakeProvider(), FakeProvider(context_chars=90)
        build.summarize(_turns(), STATS, RESULT, big)
        build.summarize(_turns(), STATS, RESULT, small)
        big_prompt = next(p for k, p, _ in big.calls if k == "section")
        small_prompt = next(p for k, p, _ in small.calls if k == "section")
        assert "Keep the whole section to about 150 words" in big_prompt
        assert "one or two short paragraphs" in small_prompt
        assert "optional box" not in small_prompt
        # No tables on the small path: they are the most token-hungry part of the schema.
        assert "leave columns and rows empty." in small_prompt
        assert "only if this topic genuinely is a comparison" not in small_prompt

    def test_the_summary_is_shorter_than_listening(self):
        """Measured: unbudgeted sections for a 10-minute video ran longer than its transcript."""
        assert build.section_budget(2_400, 3) == 150       # a 10-minute video: about a page
        assert build.section_budget(20_000, 12) == 250     # a two-hour interview
        assert build.section_budget(60_000, 12) == 250     # capped at 3,000 words in total
        assert build.section_budget(100, 3) == 150         # never below a readable minimum
        assert build.section_budget(30_000, 4) == 350      # never above a readable maximum

    def test_a_long_budget_allows_more_prose_and_boxes(self):
        provider = FakeProvider(sections=3)
        long_turns = [Turn(speaker="P", raw="S", start=float(i), end=float(i) + 1,
                           text="word " * 100) for i in range(60)]
        build.summarize(long_turns, STATS, RESULT, provider)
        prompt = next(p for k, p, _ in provider.calls if k == "section")
        assert "two to four paragraphs" in prompt and "zero to three optional boxes" in prompt
        budget = int(re.search(r"about (\d+) words", prompt).group(1))
        assert 220 < budget <= build.SECTION_MAX_WORDS

    def test_sections_scale_to_the_recording(self):
        assert build._section_count(12, 0) == 12                 # unknown length: trust the ask
        assert build._section_count(12, 100) == 3                # a 100 s clip: the floor
        assert build._section_count(12, 1500) == 4               # 25 minutes
        assert build._section_count(12, 7200) == 12              # two hours: the ceiling
        assert build._section_count(4, 7200) == 4                # never above what was asked

    def test_a_short_clip_is_not_asked_for_twelve_sections(self):
        """Measured: twelve sections asked of a 100 s clip returned a plan with none at all."""
        provider = FakeProvider(sections=3)
        build.summarize(_turns(), STATS, {**RESULT, "duration": 100.0}, provider)
        plan_prompt = next(p for k, p, _ in provider.calls if k == "plan")
        assert "exactly 3 topics" in plan_prompt

    def test_the_plan_is_retried_once_with_a_smaller_ask(self):
        class Overreaching(FakeProvider):
            """Returns no sections the first time, like a model cut off mid-plan."""

            def __init__(self, **kw):
                super().__init__(**kw)
                self.plans = 0

            def ask_json(self, system, prompt, schema_, *, max_tokens=8000, cache_prefix=""):
                out = super().ask_json(system, prompt, schema_, max_tokens=max_tokens,
                                       cache_prefix=cache_prefix)
                if schema_ is schema.PLAN_SCHEMA:
                    self.plans += 1
                    if self.plans == 1:
                        out["sections"] = []
                return out

        provider = Overreaching(sections=8)
        logs: list[str] = []
        out = build.summarize(_turns(), STATS, RESULT, provider, on_log=logs.append)
        assert provider.plans == 2
        assert any("retrying with" in m for m in logs)
        assert out["sections"]
        # The retry drops the per-section focus note, which is what overruns the output.
        first, second = (p for k, p, _ in provider.calls if k == "plan")
        assert "'focus' note" in first and "'focus' note" not in second

    def test_no_sections_after_the_retry_is_still_an_error(self):
        class Hopeless(FakeProvider):
            def ask_json(self, system, prompt, schema_, *, max_tokens=8000, cache_prefix=""):
                out = super().ask_json(system, prompt, schema_, max_tokens=max_tokens,
                                       cache_prefix=cache_prefix)
                if schema_ is schema.PLAN_SCHEMA:
                    out["sections"] = []
                return out

        with pytest.raises(providers.SummaryError, match="no section plan"):
            build.summarize(_turns(), STATS, RESULT, Hopeless())

    def test_no_sections_is_an_error_not_an_empty_page(self):
        provider = FakeProvider(sections=0)
        with pytest.raises(providers.SummaryError):
            build.summarize(_turns(), STATS, RESULT, provider)

    def test_empty_transcript_is_an_error(self):
        with pytest.raises(providers.SummaryError):
            build.summarize([], STATS, RESULT, FakeProvider())

    def test_output_renders(self):
        """The contract that matters: whatever build produces, the renderers accept."""
        out = build.summarize(_turns(), STATS, RESULT, FakeProvider())
        page = webpage.render_html(out)
        assert "<h1>Fake title</h1>" in page
        assert page.rstrip().endswith("</html>")
        assert webpage.render_markdown(out).startswith("# Fake title")

    def test_transcript_text_carries_timestamps_and_speakers(self):
        text = build.transcript_text(_turns())
        assert text.startswith("[00:00:00] Person 1: ")
        assert "[00:00:12] Person 2:" in text
        assert "\n" not in text.split("\n")[0]        # each turn is one line


# --------------------------------------------------------------------------- selection

class TestClaudeCodeProvider:
    """The no-API-key path: `claude -p`, its JSON envelope, and the resumed session."""

    @pytest.fixture(autouse=True)
    def _no_install_dirs(self, monkeypatch):
        """Hermetic: this machine really has ~/.npm-global/bin/claude, which the fallback
        search would otherwise find in every test that pretends the CLI is missing."""
        monkeypatch.setattr(providers, "CLAUDE_CODE_CANDIDATES", ())
        monkeypatch.setattr(providers, "CLAUDE_CODE_CANDIDATE_GLOBS", ())
        monkeypatch.delenv("NPM_CONFIG_PREFIX", raising=False)
        monkeypatch.delenv("npm_config_prefix", raising=False)

    def _provider(self, monkeypatch, envelopes):
        """A provider whose subprocess returns canned envelopes, recording each invocation."""
        provider = providers.ClaudeCodeProvider()
        calls: list[dict] = []
        queue = list(envelopes)

        def fake_run(args, **kw):
            calls.append({"args": args, "input": kw.get("input", ""), "cwd": kw.get("cwd"),
                          "env": kw.get("env")})
            body = queue.pop(0) if queue else {"is_error": False, "result": "{}"}
            # A string is a raw stream-json stdout; a dict is a json envelope.
            out = body if isinstance(body, str) else json.dumps(body)
            return subprocess.CompletedProcess(args, 0, out, "")

        monkeypatch.setattr(providers.shutil, "which", lambda _b: "/usr/bin/claude")
        monkeypatch.setattr(providers.subprocess, "run", fake_run)
        return provider, calls

    def test_the_command_is_sandboxed_and_the_prompt_goes_on_stdin(self, monkeypatch):
        provider, calls = self._provider(
            monkeypatch, [{"is_error": False, "result": '{"topics": ["a"]}',
                           "session_id": "sess-1", "total_cost_usd": 0.04}])
        out = provider.ask_json("SYSTEM", "QUESTION", schema.NOTES_SCHEMA,
                                cache_prefix="TRANSCRIPT: lots of words")
        assert out == {"topics": ["a"]}
        args = calls[0]["args"]
        assert "-p" in args and "--output-format" in args and "json" in args
        assert "--restricted" in args                       # no bash, no code, no WebFetch
        assert "--strict-mcp-config" in args                # none of the user's MCP servers
        assert args[args.index("--tools") + 1] == ""         # a JSON pass gets no tools at all
        assert args[args.index("--output-format") + 1] == "json"
        assert args[args.index("--model") + 1] == "claude-opus-5-5"
        assert args[args.index("--effort") + 1] == "high"
        assert args[args.index("--system-prompt") + 1] == "SYSTEM"
        # A 145 kB transcript has no business in argv.
        assert "TRANSCRIPT: lots of words" in calls[0]["input"]
        assert not any("TRANSCRIPT" in a for a in args)
        # The schema travels in the user turn, so a resumed session still gets the right shape.
        assert "JSON Schema" in calls[0]["input"]
        assert provider.cost_usd == pytest.approx(0.04)

    def test_it_runs_outside_the_project_so_no_claude_md_is_inherited(self, monkeypatch):
        provider, calls = self._provider(monkeypatch, [{"is_error": False, "result": "{}"}])
        provider.ask_json("s", "p", schema.NOTES_SCHEMA)
        assert calls[0]["cwd"] == providers.CLAUDE_CODE_CWD
        assert "Sinribe" not in str(calls[0]["cwd"])

    def test_the_transcript_is_sent_once_and_later_passes_resume(self, monkeypatch):
        provider, calls = self._provider(monkeypatch, [
            {"is_error": False, "result": "{}", "session_id": "sess-1"},
            {"is_error": False, "result": "{}", "session_id": "sess-1"},
            {"is_error": False, "result": "{}", "session_id": "sess-1"},
        ])
        prefix = "TRANSCRIPT: the whole two hours"
        for _ in range(3):
            provider.ask_json("s", "ask", schema.NOTES_SCHEMA, cache_prefix=prefix)

        assert "--resume" not in calls[0]["args"]
        assert prefix in calls[0]["input"]
        for call in calls[1:]:
            assert call["args"][call["args"].index("--resume") + 1] == "sess-1"
            # The point of resuming: the transcript is already in the session.
            assert prefix not in call["input"]

    def test_a_changed_transcript_starts_a_new_session(self, monkeypatch):
        provider, calls = self._provider(monkeypatch, [
            {"is_error": False, "result": "{}", "session_id": "sess-1"},
            {"is_error": False, "result": "{}", "session_id": "sess-2"},
        ])
        provider.ask_json("s", "p", schema.NOTES_SCHEMA, cache_prefix="first")
        provider.ask_json("s", "p", schema.NOTES_SCHEMA, cache_prefix="second")
        assert "--resume" not in calls[1]["args"]
        assert "second" in calls[1]["input"]

    STREAM = "\n".join(json.dumps(e) for e in [
        {"type": "system", "subtype": "init", "tools": ["WebFetch", "WebSearch"]},
        {"type": "user", "tool_use_result": {"query": "q", "results": [
            {"tool_use_id": "t1", "content": [
                {"title": "Release notes", "url": "https://docs.test/notes"},
                {"title": "Blog", "url": "https://blog.test/post"}]},
            "a summary string the tool also returns"]}},
        {"type": "user", "tool_use_result": {"url": "https://docs.test/opened", "code": 200,
                                             "codeText": "OK", "result": "page text"}},
        {"type": "user", "tool_use_result": {"url": "https://dead.test", "code": 404,
                                             "codeText": "Not Found", "result": ""}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "found it",
         "session_id": "sess-9", "total_cost_usd": 0.31, "permission_denials": []},
    ])

    def test_research_searches_reads_and_does_not_join_the_summary_session(self, monkeypatch):
        provider, calls = self._provider(monkeypatch, [
            {"is_error": False, "result": "{}", "session_id": "sess-1"}, self.STREAM])
        provider.ask_json("s", "p", schema.NOTES_SCHEMA, cache_prefix="T")
        report, sources = provider.research("s", "research this")
        assert report == "found it"
        args = calls[1]["args"]
        tools = args[args.index("--tools") + 1:args.index("--tools") + 3]
        assert tools == ["WebSearch", "WebFetch"]         # --restricted drops WebFetch otherwise
        assert args[args.index("--allowedTools") + 1] == "WebSearch"
        assert args[args.index("--permission-mode") + 1] == "dontAsk"
        assert args[args.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in args and "--no-session-persistence" in args
        assert "--resume" not in args
        assert provider.cost_usd == pytest.approx(0.31)

    def test_research_sources_are_what_the_tools_returned(self, monkeypatch):
        provider, _ = self._provider(monkeypatch, [self.STREAM])
        _report, sources = provider.research("s", "q")
        assert [s["url"] for s in sources] == [
            "https://docs.test/notes", "https://blog.test/post", "https://docs.test/opened"]
        # A fetch that failed is not a page anyone read.
        assert all("dead.test" not in s["url"] for s in sources)

    def test_a_research_stream_without_a_result_is_an_error(self, monkeypatch):
        provider, _ = self._provider(monkeypatch, ['{"type": "system", "subtype": "init"}'])
        with pytest.raises(providers.SummaryError, match="no result envelope"):
            provider.research("s", "q")

    def test_it_is_found_outside_path_where_npm_put_it(self, monkeypatch, tmp_path):
        """The bug that made Claude Code unusable from the app menu: the desktop session's PATH
        has no ~/.npm-global/bin, so `which claude` found nothing."""
        binary = tmp_path / "npm-global" / "bin" / "claude"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setattr(providers.shutil, "which", lambda _b: None)
        monkeypatch.setattr(providers, "CLAUDE_CODE_CANDIDATES", (str(binary),))
        provider = providers.ClaudeCodeProvider()
        assert provider._which(patient=False) == str(binary)
        # Its folder goes on the subprocess PATH, for anything the CLI itself looks up.
        assert provider._env()["PATH"].split(":")[0] == str(binary.parent)

    def test_an_explicit_binary_path_is_honoured(self, monkeypatch, tmp_path):
        binary = tmp_path / "claude"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setattr(providers.shutil, "which", lambda _b: None)
        assert providers.ClaudeCodeProvider(binary=str(binary))._which(patient=False) == \
            str(binary)

    def test_a_plain_call_is_given_no_tools(self, monkeypatch):
        provider, calls = self._provider(monkeypatch, [{"is_error": False, "result": "{}"}])
        provider.ask_json("s", "p", schema.NOTES_SCHEMA)
        assert "--allowedTools" not in calls[0]["args"]

    def test_a_fenced_answer_is_unwrapped(self, monkeypatch):
        provider, _ = self._provider(monkeypatch, [
            {"is_error": False, "result": '```json\n{"topics": ["a"]}\n```'}])
        assert provider.ask_json("s", "p", schema.NOTES_SCHEMA) == {"topics": ["a"]}

    def test_a_truncated_answer_is_salvaged(self, monkeypatch):
        provider, _ = self._provider(monkeypatch, [
            {"is_error": False, "result": '{"topics": ["kept"], "claims": ["half a cl'}])
        logs: list[str] = []
        provider.on_log = logs.append
        assert provider.ask_json("s", "p", schema.NOTES_SCHEMA) == {"topics": ["kept"]}
        assert any("repairing" in m for m in logs)

    def test_an_error_envelope_is_reported_not_parsed(self, monkeypatch):
        provider, _ = self._provider(monkeypatch, [
            {"is_error": True, "subtype": "error_max_turns", "result": "gave up"}])
        with pytest.raises(providers.SummaryError, match="error_max_turns"):
            provider.ask_json("s", "p", schema.NOTES_SCHEMA)

    def test_a_nonzero_exit_is_reported(self, monkeypatch):
        provider = providers.ClaudeCodeProvider()
        monkeypatch.setattr(providers.shutil, "which", lambda _b: "/usr/bin/claude")
        monkeypatch.setattr(providers.subprocess, "run",
                            lambda args, **kw: subprocess.CompletedProcess(
                                args, 1, "", "not logged in"))
        with pytest.raises(providers.SummaryError, match="not logged in"):
            provider.ask_json("s", "p", schema.NOTES_SCHEMA)

    def test_denied_tools_are_surfaced(self, monkeypatch):
        provider, _ = self._provider(monkeypatch, [
            {"is_error": False, "result": "{}",
             "permission_denials": [{"tool_name": "Read"}, {"tool_name": "Read"}]}])
        logs: list[str] = []
        provider.on_log = logs.append
        provider.ask_json("s", "p", schema.NOTES_SCHEMA)
        assert any("denied Read" in m for m in logs)

    def test_a_missing_cli_is_an_actionable_unavailable(self, monkeypatch):
        monkeypatch.setattr(providers.shutil, "which", lambda _b: None)
        with pytest.raises(providers.Unavailable, match="npm install"):
            providers.ClaudeCodeProvider().probe()

    def test_the_probe_does_not_wait_because_it_runs_on_the_gui_thread(self, monkeypatch):
        """The dropdown calls this on every change; a 3 s freeze to confirm 'not installed' is
        worse than the answer is useful. Retries belong mid-job, not here."""
        looks = {"n": 0}

        def counting(_binary):
            looks["n"] += 1
            return None

        monkeypatch.setattr(providers.shutil, "which", counting)
        monkeypatch.setattr(providers, "CLAUDE_CODE_RESOLVE_WAIT", 99)   # would hang if used
        with pytest.raises(providers.Unavailable):
            providers.ClaudeCodeProvider().probe()
        assert looks["n"] == 1

    def test_its_own_autoupdater_is_disabled_for_our_subprocess(self, monkeypatch):
        """Measured: `claude` auto-updated 70 s into a job and the next call lost the binary."""
        provider, calls = self._provider(monkeypatch, [{"is_error": False, "result": "{}"}])
        captured = {}

        def fake_run(args, **kw):
            captured.update(kw.get("env") or {})
            return subprocess.CompletedProcess(args, 0, json.dumps(
                {"is_error": False, "result": "{}"}), "")

        monkeypatch.setattr(providers.subprocess, "run", fake_run)
        provider.ask_json("s", "p", schema.NOTES_SCHEMA)
        assert captured.get("DISABLE_AUTOUPDATER") == "1"
        assert "PATH" in captured                       # the rest of the environment survives

    def test_a_binary_being_replaced_is_waited_for_not_failed(self, monkeypatch):
        monkeypatch.setattr(providers, "CLAUDE_CODE_RESOLVE_WAIT", 0)
        provider = providers.ClaudeCodeProvider()
        logs: list[str] = []
        provider.on_log = logs.append
        attempts = {"n": 0}

        def flaky(_binary):
            attempts["n"] += 1
            return "/usr/bin/claude" if attempts["n"] > 2 else None

        monkeypatch.setattr(providers.shutil, "which", flaky)
        monkeypatch.setattr(providers.subprocess, "run",
                            lambda args, **kw: subprocess.CompletedProcess(
                                args, 0, json.dumps({"is_error": False, "result": "{}"}), ""))
        provider.ask_json("s", "p", schema.NOTES_SCHEMA)
        assert attempts["n"] == 3
        assert any("reappeared" in m for m in logs)

    def test_the_binary_is_resolved_once_per_job(self, monkeypatch):
        provider = providers.ClaudeCodeProvider()
        seen = {"n": 0}

        def counting(_binary):
            seen["n"] += 1
            return "/usr/bin/claude"

        monkeypatch.setattr(providers.shutil, "which", counting)
        monkeypatch.setattr(providers.Path, "exists", lambda self: True)
        monkeypatch.setattr(providers.subprocess, "run",
                            lambda args, **kw: subprocess.CompletedProcess(
                                args, 0, json.dumps({"is_error": False, "result": "{}"}), ""))
        for _ in range(5):
            provider.ask_json("s", "p", schema.NOTES_SCHEMA)
        assert seen["n"] == 1

    def test_a_logged_out_cli_is_caught_before_the_job_runs(self, monkeypatch):
        monkeypatch.setattr(providers.shutil, "which", lambda _b: "/usr/bin/claude")
        monkeypatch.setattr(providers.subprocess, "run",
                            lambda args, **kw: subprocess.CompletedProcess(
                                args, 0, json.dumps({"loggedIn": False}), ""))
        with pytest.raises(providers.Unavailable, match="auth login"):
            providers.ClaudeCodeProvider().probe()

    def test_a_logged_in_cli_passes(self, monkeypatch):
        monkeypatch.setattr(providers.shutil, "which", lambda _b: "/usr/bin/claude")
        monkeypatch.setattr(providers.subprocess, "run",
                            lambda args, **kw: subprocess.CompletedProcess(
                                args, 0, json.dumps({"loggedIn": True,
                                                     "authMethod": "claude.ai"}), ""))
        providers.ClaudeCodeProvider().probe()

    def test_an_old_cli_that_prints_no_json_is_not_rejected(self, monkeypatch):
        """A working binary is most of the check; a bad login surfaces on the first call."""
        monkeypatch.setattr(providers.shutil, "which", lambda _b: "/usr/bin/claude")
        monkeypatch.setattr(providers.subprocess, "run",
                            lambda args, **kw: subprocess.CompletedProcess(
                                args, 0, "Logged in as someone", ""))
        providers.ClaudeCodeProvider().probe()

    def test_a_timeout_is_reported_as_such(self, monkeypatch):
        provider = providers.ClaudeCodeProvider(timeout=1.0)
        monkeypatch.setattr(providers.shutil, "which", lambda _b: "/usr/bin/claude")

        def boom(args, **kw):
            raise subprocess.TimeoutExpired(args, 1.0)

        monkeypatch.setattr(providers.subprocess, "run", boom)
        with pytest.raises(providers.SummaryError, match="did not finish"):
            provider.ask_json("s", "p", schema.NOTES_SCHEMA)


def _unavailable(msg):
    def probe(self):
        raise providers.Unavailable(msg)
    return probe


class TestProviderSelection:
    """`auto` order matters: Claude Code first because it needs nothing configured."""

    def _stub(self, monkeypatch, **outcomes):
        """Set each provider's probe to pass (None) or fail with the given message."""
        classes = {"claude-code": providers.ClaudeCodeProvider,
                   "claude": providers.ClaudeProvider,
                   "ollama": providers.OllamaProvider}
        for name, cls in classes.items():
            outcome = outcomes.get(name.replace("-", "_"), "not stubbed")
            monkeypatch.setattr(cls, "probe",
                                (lambda self: None) if outcome is None
                                else _unavailable(outcome))

    def test_auto_prefers_claude_code_over_the_api(self, monkeypatch):
        """The whole point: no API key required, because the CLI already has a login."""
        self._stub(monkeypatch, claude_code=None, claude=None, ollama=None)
        assert providers.pick({"summary_provider": "auto"}).name == "claude-code"

    def test_auto_falls_through_to_the_api_then_the_local_model(self, monkeypatch):
        self._stub(monkeypatch, claude_code="no cli", claude=None, ollama=None)
        assert providers.pick({"summary_provider": "auto"}).name == "claude"
        self._stub(monkeypatch, claude_code="no cli", claude="no key", ollama=None)
        assert providers.pick({"summary_provider": "auto"}).name == "ollama"

    def test_explicit_choice_is_probed_and_its_failure_reported(self, monkeypatch):
        self._stub(monkeypatch, claude_code=None, claude=None, ollama="no ollama")
        with pytest.raises(providers.Unavailable, match="no ollama"):
            providers.pick({"summary_provider": "ollama"})

    def test_an_explicit_choice_is_never_silently_swapped(self, monkeypatch):
        """Asking for the API and getting the CLI instead would be a different bill."""
        self._stub(monkeypatch, claude_code=None, claude="no key", ollama=None)
        with pytest.raises(providers.Unavailable, match="no key"):
            providers.pick({"summary_provider": "claude"})

    def test_auto_records_what_it_skipped(self, monkeypatch):
        """A silent fallback is how a brief was written by a 4B model while the user believed
        Claude wrote it. The reason travels with the provider so the job can say it."""
        self._stub(monkeypatch, claude_code="the `claude` command was not found",
                   claude="no key", ollama=None)
        chosen = providers.pick({"summary_provider": "auto"})
        assert chosen.name == "ollama"
        assert chosen.skipped == ["claude-code: the `claude` command was not found",
                                  "claude: no key"]
        self._stub(monkeypatch, claude_code=None, claude=None, ollama=None)
        assert providers.pick({"summary_provider": "auto"}).skipped == []

    def test_auto_reports_every_reason_when_nothing_works(self, monkeypatch):
        self._stub(monkeypatch, claude_code="no cli", claude="no key", ollama="no ollama")
        with pytest.raises(providers.Unavailable) as e:
            providers.pick({"summary_provider": "auto"})
        for reason in ("no cli", "no key", "no ollama"):
            assert reason in str(e.value)

    def test_describe_never_raises(self, monkeypatch):
        self._stub(monkeypatch, claude_code="x", claude="y", ollama="z")
        assert providers.describe({}) == "unavailable"

    def test_the_model_is_configurable_on_both_cloud_paths(self):
        for name, label in (("claude", "Claude claude-sonnet-5"),
                            ("claude-code", "Claude Code (claude-sonnet-5)")):
            provider = providers.build_provider({"summary_model": "claude-sonnet-5"}, name)
            assert provider.model == "claude-sonnet-5"
            assert provider.label == label

    def test_the_default_model_is_opus(self):
        assert providers.build_provider({}, "claude").model == "claude-opus-5-5"
        assert providers.build_provider({}, "claude-code").model == "claude-opus-5-5"

    def test_an_absurd_effort_falls_back_to_high(self):
        assert providers.build_provider({"summary_effort": "banana"}, "claude").effort == "high"

    def test_can_research_matches_the_providers(self):
        assert providers.can_research("claude-code") is True
        assert providers.can_research("claude") is True
        assert providers.can_research("ollama") is False

    def test_the_local_provider_refuses_to_fake_research(self):
        with pytest.raises(providers.Unavailable, match="web research"):
            providers.OllamaProvider().research("s", "p")
