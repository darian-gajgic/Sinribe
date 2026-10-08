"""Renders a brief as a standalone HTML page, and as its Markdown twin.

The page is a single file with no external requests: the CSS is inline, there is no JavaScript
and no font or script is fetched. That is not minimalism for its own sake. These pages are opened
from a file:// path, kept in a folder next to the transcript, and are expected to still render in
five years and on a machine with no network, which rules out a CDN.

The order is the order a reader who wants to learn fast needs it in: the one-minute version, then
what has changed since the recording (when the web check ran), then the full summary one topic at
a time, then the terms and a quiz to check it stuck. Timestamps link into the video where the
source is YouTube, so any point can be heard in the speaker's own words.

Model text is escaped and then given back a deliberately small vocabulary of inline markup
(**bold**, *italics*, `code`, [label](url)). Escaping first and re-introducing markup second is
the order that matters: doing it the other way round lets a stray angle bracket in a transcript
quote close a tag. Link targets are checked, because everything here was written by a language
model and a javascript: URL in a summary is still a javascript: URL.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..textfmt import human_duration

# kind -> (css class, fallback heading)
BLOCK_STYLE = {
    "example": ("b-example", "Example"),
    "evidence": ("b-evidence", "Evidence"),
    "forecast": ("b-forecast", "Forecast"),
    "caveat": ("b-caveat", "Caveat"),
    "howto": ("b-howto", "How to"),
    "definition": ("b-definition", "Definition"),
}
# status -> (css class, label shown to the reader)
STATUS_STYLE = {
    "outdated": ("st-outdated", "Outdated"),
    "incorrect": ("st-incorrect", "Incorrect"),
    "disputed": ("st-disputed", "Disputed"),
    "open": ("st-open", "Still open"),
    "confirmed": ("st-confirmed", "Still accurate"),
    "unverified": ("st-unverified", "Unverified"),
}
# A block longer than this gets the full width of the grid rather than one column, so a dense
# argument is not squeezed into a narrow measure beside a three-line block.
WIDE_CHARS = 620
# Silent reading speed for non-fiction, in words per minute. Used for the reading-time line, so
# it errs slow: a brief is read to be learned from, not skimmed.
READING_WPM = 220

_INLINE_LINK = re.compile(r"\[([^\]\[]{1,160})\]\((https?://[^\s)]{1,500}|#[\w-]{1,80})\)")
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.S)
_ITALIC = re.compile(r"(?<![\*\w])\*(?=\S)([^\*]+?)(?<=\S)\*(?!\*)", re.S)
_CODE = re.compile(r"`([^`]+)`")
_YOUTUBE_ID = re.compile(r"^[\w-]{6,20}$")
_MARKUP = re.compile(r"\*\*|`|\[([^\]]*)\]\([^)]*\)")


def _esc(text: str) -> str:
    return html.escape(str(text or ""), quote=True)


def inline(text: str) -> str:
    """Escape `text`, then re-apply the small inline markup the prompts allow."""
    out = _esc(text)
    out = _INLINE_LINK.sub(
        lambda m: f'<a href="{m.group(2)}" rel="noopener noreferrer">{m.group(1)}</a>', out)
    out = _CODE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = _ITALIC.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    return out


def _safe_url(url: str) -> str | None:
    """http(s) and in-page anchors only. Anything else is dropped, not rendered inert."""
    u = str(url or "").strip()
    if u.startswith("#") and len(u) < 120:
        return _esc(u)
    return _esc(u) if u[:8].lower().startswith(("http://", "https:/")) else None


# --------------------------------------------------------------------------- helpers

def seconds(stamp: str) -> int | None:
    """'1:02:03' or '2:03' as seconds, or None."""
    parts = str(stamp or "").strip().strip("[]").split(":")
    if not 2 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        return None
    total = 0
    for p in parts:
        total = total * 60 + int(p)
    return total


def youtube_id(url: str) -> str:
    """The video id of a YouTube link, or "" for anything else."""
    try:
        parsed = urlparse(str(url or ""))
    except ValueError:
        return ""
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        vid = parsed.path.strip("/").split("/")[0]
    elif host in ("youtube.com", "music.youtube.com"):
        if parsed.path == "/watch":
            vid = (parse_qs(parsed.query).get("v") or [""])[0]
        else:
            bits = parsed.path.strip("/").split("/")
            vid = bits[1] if len(bits) > 1 and bits[0] in ("live", "shorts", "embed") else ""
    else:
        return ""
    return vid if _YOUTUBE_ID.match(vid or "") else ""


def jump_url(source: str, stamp: str) -> str:
    """A link that starts the video at `stamp`, when the source is YouTube."""
    vid, at = youtube_id(source), seconds(stamp)
    if not vid or at is None:
        return ""
    return f"https://www.youtube.com/watch?v={vid}&t={at}s"


def _stamp(source: str, stamp: str) -> str:
    """A timestamp, linked into the video when that is possible."""
    if not stamp:
        return ""
    url = jump_url(source, stamp)
    if url:
        return f'<a class="ts" href="{_esc(url)}" rel="noopener noreferrer">{_esc(stamp)}</a>'
    return f'<span class="ts">{_esc(stamp)}</span>'


def _words(*chunks: object) -> int:
    count = 0
    for chunk in chunks:
        if isinstance(chunk, str):
            count += len(_MARKUP.sub(r"\1", chunk).split())
        elif isinstance(chunk, list):
            count += _words(*chunk)
        elif isinstance(chunk, dict):
            count += _words(*chunk.values())
    return count


def minute_words(summary: dict) -> int:
    return _words(summary.get("gist", ""), summary.get("takeaways", []),
                  summary.get("why_it_matters", ""))


def full_words(summary: dict) -> int:
    return sum(_words(s.get("gist", ""), s.get("paragraphs", []), s.get("key_points", []),
                      [b.get("paragraphs", []) + b.get("bullets", []) for b in s.get("blocks", [])],
                      s.get("recommendation") or {})
               for s in summary.get("sections", []))


def _minutes(words: int) -> int:
    return max(1, round(words / READING_WPM))


def _age(meta: dict) -> str:
    days = meta.get("age_days")
    if days is None:
        return ""
    if days == 0:
        return "today"
    if days < 60:
        return f"{days} days ago"
    if days < 730:
        return f"about {round(days / 30.4)} months ago"
    return f"about {round(days / 365.25)} years ago"


def _check_counts(check: dict | None) -> dict[str, int]:
    counts = {k: 0 for k in STATUS_STYLE}
    for c in (check or {}).get("checks", []):
        counts[c.get("status", "unverified")] = counts.get(c.get("status", "unverified"), 0) + 1
    return counts


# --------------------------------------------------------------------------- CSS

_CSS = """
  :root{
    --bg:#f6f7f9; --card:#ffffff; --ink:#1a2130; --muted:#5b6575; --line:#e4e8ee;
    --accent:#2563eb; --accent-soft:#eff4ff;
    --risk:#b91c1c; --risk-soft:#fdf0ef;
    --ok:#15803d; --ok-soft:#effaf1;
    --warn:#a15c07; --warn-soft:#fdf5e7;
    --violet:#6d28d9; --violet-soft:#f5f1fd;
    --radius:14px;
    --shadow:0 1px 2px rgba(16,24,40,.05), 0 6px 20px -8px rgba(16,24,40,.08);
  }
  @media (prefers-color-scheme: dark){
    :root{
      --bg:#0e1117; --card:#161b24; --ink:#e8ecf3; --muted:#9aa4b5; --line:#252c3a;
      --accent:#7aa2ff; --accent-soft:#1a2336;
      --risk:#f0868a; --risk-soft:#2a1a1c;
      --ok:#6fd394; --ok-soft:#14251b;
      --warn:#efb45c; --warn-soft:#2b2214;
      --violet:#b79df5; --violet-soft:#221a33;
      --shadow:0 1px 2px rgba(0,0,0,.4);
    }
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:16px/1.62 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Ubuntu,Cantarell,sans-serif;}
  a{color:var(--accent);text-decoration:none}
  a:hover{text-decoration:underline}
  a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px;
    border-radius:4px}

  .layout{max-width:1240px;margin:0 auto;padding:0 20px;display:grid;
    grid-template-columns:250px minmax(0,1fr);gap:32px}
  @media (max-width:960px){.layout{grid-template-columns:1fr}}

  nav.toc{position:sticky;top:16px;align-self:start;padding:18px 4px;font-size:.86rem;
    max-height:calc(100vh - 32px);overflow:auto}
  nav.toc .toc-title{font-weight:700;text-transform:uppercase;letter-spacing:.08em;
    font-size:.7rem;color:var(--muted);margin:14px 0 8px 10px}
  nav.toc .toc-title:first-child{margin-top:0}
  nav.toc a{display:block;color:var(--muted);padding:5px 10px;border-radius:8px}
  nav.toc a:hover{color:var(--ink);background:var(--card);text-decoration:none}
  nav.toc a .n{display:inline-block;width:1.6em;color:var(--accent);font-weight:600;
    font-variant-numeric:tabular-nums}
  nav.toc a .flag{color:var(--warn);font-weight:700}
  @media (max-width:960px){
    nav.toc{position:static;max-height:none;padding:0 0 8px}
    nav.toc .links{display:flex;flex-wrap:wrap;gap:4px}
    nav.toc a{border:1px solid var(--line);background:var(--card)}
  }

  main{padding:24px 0 80px;min-width:0;max-width:880px}

  header.hero{padding:40px 0 8px}
  .kicker{color:var(--accent);font-weight:700;text-transform:uppercase;letter-spacing:.12em;
    font-size:.72rem;margin-bottom:10px}
  h1{font-size:clamp(1.7rem,3.4vw,2.5rem);line-height:1.15;margin:0 0 12px;
    letter-spacing:-.02em;text-wrap:balance}
  .sub{color:var(--muted);max-width:62ch;margin:0 0 14px;font-size:1.05rem}
  .facts{display:flex;flex-wrap:wrap;gap:6px 16px;color:var(--muted);font-size:.86rem;
    margin:0 0 14px}
  .reading{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px}
  .reading span{background:var(--card);border:1px solid var(--line);border-radius:99px;
    padding:3px 12px;font-size:.8rem;color:var(--muted)}
  .reading b{color:var(--ink);font-weight:600}

  .fresh{border:1px solid var(--line);background:var(--card);border-radius:var(--radius);
    padding:12px 16px;margin:0 0 8px;font-size:.92rem}
  .fresh.warn{border-color:var(--warn);background:var(--warn-soft)}
  .fresh.ok{border-color:var(--ok);background:var(--ok-soft)}
  .fresh.bad{border-color:var(--risk);background:var(--risk-soft)}
  .fresh strong{font-weight:700}

  section{margin-top:44px;scroll-margin-top:16px}
  section > h2{display:flex;align-items:baseline;flex-wrap:wrap;gap:4px 12px;font-size:1.35rem;
    letter-spacing:-.01em;margin:0 0 6px;line-height:1.3;text-wrap:balance}
  section > h2 .num{color:var(--accent);font-size:.95rem;font-weight:800;
    font-variant-numeric:tabular-nums}
  h2 .ts{font-size:.8rem;font-weight:600}
  .part{font-size:.75rem;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
    margin:56px 0 0;padding-top:14px;border-top:1px solid var(--line)}
  .sec-sub{color:var(--muted);font-size:.95rem;margin:0 0 14px;max-width:75ch}
  .sec-gist{font-size:1.06rem;margin:0 0 12px;max-width:72ch}
  .prose p{margin:0 0 12px;max-width:72ch}

  .minute{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    padding:22px 26px;box-shadow:var(--shadow);margin-top:22px}
  .minute h2{margin-bottom:12px}
  .minute .gist{font-size:1.08rem;margin:0 0 14px;max-width:70ch}
  .minute ul.takeaways{margin:0 0 14px;padding-left:20px}
  .minute ul.takeaways li{margin:7px 0;max-width:70ch}
  .minute .why{margin:0;color:var(--ink);max-width:70ch}
  .minute .heads{margin:16px 0 0;padding:12px 16px;border:1px solid var(--warn);
    background:var(--warn-soft);border-radius:10px;font-size:.94rem}
  .minute .heads ul{margin:6px 0 0;padding-left:20px} .minute .heads li{margin:4px 0}
  .watch{list-style:none;padding:0;margin:12px 0}
  .watch li{background:var(--card);border:1px solid var(--line);border-radius:12px;
    padding:10px 16px;margin:8px 0}
  .watch .ts{font-weight:700;margin-right:8px;font-variant-numeric:tabular-nums}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;
    margin:18px 0 0}
  .stats-title{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
    font-weight:700;margin:18px 0 0}
  .stat{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
  .stat .v{font-size:1.3rem;font-weight:800;letter-spacing:-.02em}
  .stat .l{font-size:.78rem;color:var(--muted);line-height:1.35;margin-top:2px}
  .stat.red .v{color:var(--risk)} .stat.blue .v{color:var(--accent)}
  .stat.green .v{color:var(--ok)} .stat.violet .v{color:var(--violet)}

  .pill{display:inline-block;font-size:.7rem;font-weight:700;text-transform:uppercase;
    letter-spacing:.06em;border-radius:99px;padding:2px 9px;border:1px solid currentColor;
    white-space:nowrap}
  .st-outdated{color:var(--warn)} .st-incorrect{color:var(--risk)}
  .st-disputed{color:var(--violet)} .st-confirmed{color:var(--ok)} .st-open{color:var(--accent)}
  .st-unverified{color:var(--muted)}
  .overall{font-size:1.02rem;max-width:72ch;margin:0 0 14px}
  .verdict{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    padding:14px 18px;margin:10px 0;box-shadow:var(--shadow)}
  .verdict.warn{border-color:var(--warn)} .verdict.bad{border-color:var(--risk)}
  .verdict.disp{border-color:var(--violet)}
  .verdict .said{margin:8px 0 6px}
  .verdict .said .lbl,.verdict .now .lbl{font-size:.72rem;text-transform:uppercase;
    letter-spacing:.06em;color:var(--muted);font-weight:700;margin-right:6px}
  .verdict .now{margin:0 0 6px}
  .verdict .srcs{font-size:.82rem;color:var(--muted);margin:0}
  .dev{margin:10px 0;padding:0 0 10px;border-bottom:1px solid var(--line)}
  .dev:last-child{border-bottom:0}
  .dev .when{color:var(--muted);font-size:.82rem;margin-right:6px;
    font-variant-numeric:tabular-nums}
  .update{background:var(--warn-soft);border:1px solid var(--warn);border-radius:12px;
    padding:12px 16px;margin:0 0 16px}
  .update.incorrect{background:var(--risk-soft);border-color:var(--risk)}
  .update.disputed{background:var(--violet-soft);border-color:var(--violet)}
  .update p{margin:4px 0}
  .update .head{font-size:.74rem;text-transform:uppercase;letter-spacing:.07em;font-weight:700}

  .keys{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    padding:14px 20px;margin:4px 0 14px;box-shadow:var(--shadow)}
  .keys h4,.block h4,.reco h4{margin:0 0 8px;font-size:.78rem;text-transform:uppercase;
    letter-spacing:.07em;display:flex;align-items:center;gap:7px}
  .keys ul{margin:0;padding-left:20px} .keys li{margin:5px 0}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:0 0 6px}
  @media (max-width:820px){.grid{grid-template-columns:1fr}}
  .block{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    padding:14px 18px;box-shadow:var(--shadow)}
  .block.wide{grid-column:1/-1}
  .block h4 .dot{width:9px;height:9px;border-radius:50%;flex:none}
  .b-example h4{color:var(--accent)} .b-example .dot{background:var(--accent)}
  .b-evidence h4{color:var(--ok)} .b-evidence .dot{background:var(--ok)}
  .b-forecast h4{color:var(--violet)} .b-forecast .dot{background:var(--violet)}
  .b-caveat h4{color:var(--risk)} .b-caveat .dot{background:var(--risk)}
  .b-howto h4{color:var(--warn)} .b-howto .dot{background:var(--warn)}
  .b-definition h4{color:var(--ink)} .b-definition .dot{background:var(--muted)}
  .block p{margin:0 0 8px} .block p:last-child{margin-bottom:0}
  .block ul{margin:4px 0 4px;padding-left:19px}
  .block li{margin:5px 0}
  .block li::marker,.keys li::marker{color:var(--muted)}

  .quote{border-left:3px solid var(--accent);background:var(--accent-soft);
    border-radius:0 10px 10px 0;padding:10px 14px;margin:10px 0;font-style:italic}
  .quote .who{display:block;font-style:normal;font-size:.78rem;color:var(--muted);
    margin-top:4px}

  .reco{background:var(--ok-soft);border:1px solid var(--ok);border-radius:var(--radius);
    padding:14px 20px;margin-top:14px}
  .reco h4{color:var(--ok)}
  .reco p{margin:0 0 8px} .reco p:last-child{margin:0}
  .reco ul{margin:4px 0 0;padding-left:19px} .reco li{margin:4px 0}

  table{width:100%;border-collapse:collapse;font-size:.9rem;margin:8px 0}
  th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);
    vertical-align:top}
  th{font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
  .tbl-wrap{overflow-x:auto;margin-top:14px}
  .tbl-cap{font-size:.78rem;color:var(--muted);margin:6px 2px 0}

  dl.terms{display:grid;grid-template-columns:minmax(120px,max-content) 1fr;gap:8px 18px;
    background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    padding:16px 20px;margin:12px 0;box-shadow:var(--shadow)}
  dl.terms dt{font-weight:700}
  dl.terms dd{margin:0;color:var(--ink)}
  @media (max-width:620px){dl.terms{grid-template-columns:1fr} dl.terms dd{margin-bottom:8px}}

  details{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    padding:12px 18px;margin:10px 0;box-shadow:var(--shadow)}
  details summary{cursor:pointer;font-weight:600}
  details[open] summary{margin-bottom:8px}
  details .answer{margin:0;color:var(--ink)}
  details ul{margin:4px 0;padding-left:20px}

  .src{font-size:.86rem;background:var(--card);border:1px solid var(--line);
    border-radius:var(--radius);padding:12px 18px}
  .src p{margin:6px 0}
  footer{margin-top:56px;color:var(--muted);font-size:.82rem;border-top:1px solid var(--line);
    padding-top:18px}
  footer p{margin:6px 0}

  @media print{
    :root{--bg:#fff;--card:#fff;--ink:#111;--muted:#444;--line:#ccc;--shadow:none}
    nav.toc{display:none}
    .layout{display:block;max-width:none;padding:0}
    main{padding:0;max-width:none}
    section{margin-top:26px}
    .block,.reco,.quote,details,.stat,.verdict,.update,.keys{break-inside:avoid}
    details{display:block} details > *{display:block}
    a{color:inherit;text-decoration:underline}
  }
"""


# --------------------------------------------------------------------------- blocks

def _paras(items: list[str]) -> str:
    return "".join(f"<p>{inline(p)}</p>" for p in items or [])


def _bullets(items: list[str]) -> str:
    if not items:
        return ""
    return "<ul>" + "".join(f"<li>{inline(b)}</li>" for b in items) + "</ul>"


def _block(block: dict, force_wide: bool = False) -> str:
    css, fallback = BLOCK_STYLE.get(block.get("kind", ""), BLOCK_STYLE["example"])
    size = sum(len(p) for p in block.get("paragraphs", [])) \
        + sum(len(b) for b in block.get("bullets", []))
    wide = " wide" if force_wide or size > WIDE_CHARS else ""
    heading = block.get("heading") or fallback
    return (f'<div class="block {css}{wide}"><h4><span class="dot"></span>{inline(heading)}</h4>'
            f'{_paras(block.get("paragraphs", []))}{_bullets(block.get("bullets", []))}</div>')


def _blocks(blocks: list[dict]) -> str:
    if not blocks:
        return ""
    out = []
    for i, block in enumerate(blocks):
        # An odd block count leaves a hole in a two-column grid; widening the last one fills it.
        last_and_odd = (i == len(blocks) - 1) and len(blocks) % 2 == 1
        out.append(_block(block, force_wide=last_and_odd))
    return '<div class="grid">' + "".join(out) + "</div>"


def _quotes(quotes: list[dict], source: str) -> str:
    out = []
    for q in quotes or []:
        bits = [inline(q["who"])] if q.get("who") else []
        if q.get("at"):
            bits.append(_stamp(source, q["at"]))
        attribution = f'<span class="who">{" · ".join(bits)}</span>' if bits else ""
        out.append(f'<div class="quote">{inline(q["text"])}{attribution}</div>')
    return "".join(out)


def _table(table: dict | None) -> str:
    if not table:
        return ""
    head = "".join(f"<th>{inline(c)}</th>" for c in table["columns"])
    body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>"
                   for row in table["rows"])
    caption = f'<p class="tbl-cap">{inline(table["caption"])}</p>' if table.get("caption") else ""
    return (f'<div class="tbl-wrap"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>{caption}")


def _reco(reco: dict | None) -> str:
    if not reco:
        return ""
    return (f'<div class="reco"><h4>What to do with this</h4>{_paras(reco.get("paragraphs", []))}'
            f'{_bullets(reco.get("bullets", []))}</div>')


def _links(links: list[dict]) -> str:
    out = []
    for link in links or []:
        url = _safe_url(link.get("url", ""))
        if url:
            out.append(f'<a href="{url}" rel="noopener noreferrer">'
                       f'{_esc(link.get("label") or url)}</a>')
    return " · ".join(out)


def _update(c: dict, source: str) -> str:
    """The callout at the top of a section whose content the web check found stale."""
    css, label = STATUS_STYLE.get(c["status"], STATUS_STYLE["outdated"])
    when = f" (as of {_esc(c['as_of'])})" if c.get("as_of") else ""
    links = _links(c.get("sources", []))
    return (f'<div class="update {c["status"]}"><p class="head {css}">{_esc(label)} since the '
            f'recording</p><p><strong>Said:</strong> {inline(c["claim"])} '
            f'{_stamp(source, c.get("at", ""))}</p>'
            + (f'<p><strong>Now{when}:</strong> {inline(c["now"])}</p>' if c.get("now") else "")
            + (f'<p class="srcs">{links}</p>' if links else "") + "</div>")


# --------------------------------------------------------------------------- page parts

def _stat_class(stat: dict) -> str:
    tone = stat.get("tone", "neutral")
    return "stat" + (f" {tone}" if tone in ("red", "blue", "green", "violet") else "")


def _hero(summary: dict) -> str:
    meta = summary.get("meta", {})
    source = meta.get("source", "")
    facts = []
    if meta.get("uploader"):
        facts.append(_esc(meta["uploader"]))
    if meta.get("published"):
        age = _age(meta)
        facts.append(f"Published {_esc(meta['published'])}" + (f" ({_esc(age)})" if age else ""))
    if meta.get("duration"):
        facts.append(_esc(human_duration(float(meta["duration"]))))
    if source.startswith("http"):
        label = "Watch on YouTube" if youtube_id(source) else "Original"
        facts.append(f'<a href="{_esc(source)}" rel="noopener noreferrer">{label}</a>')

    reading = ["<span><b>1 minute</b> core idea</span>"]
    words = full_words(summary)
    if words:
        reading.append(f"<span><b>{_minutes(words)} min</b> full summary</span>")
    duration = float(meta.get("duration") or 0.0)
    # Only claimed when it is true: a brief that takes longer to read than the recording takes
    # to hear has not saved anyone any time.
    if duration and _minutes(words + minute_words(summary)) * 60 < duration:
        reading.append(f"<span>instead of <b>{_esc(human_duration(duration))}"
                       f"</b> listening</span>")

    return (
        '<header class="hero">'
        + (f'<p class="kicker">{inline(summary["kicker"])}</p>' if summary.get("kicker") else "")
        + f'<h1>{inline(summary.get("title") or "Summary")}</h1>'
        + (f'<p class="sub">{inline(summary["subtitle"])}</p>' if summary.get("subtitle") else "")
        + (f'<p class="facts">{"".join(f"<span>{f}</span>" for f in facts)}</p>'
           if facts else "")
        + f'<div class="reading">{"".join(reading)}</div>'
        + _freshness(summary)
        + "</header>"
    )


def _freshness(summary: dict) -> str:
    """One line on how current the facts on this page are. Always there when it can be."""
    meta = summary.get("meta", {})
    check = summary.get("check")
    age = _age(meta)
    published = (f"Published {_esc(meta['published'])}, {_esc(age)}. " if meta.get("published")
                 and age else "")
    if check:
        counts = _check_counts(check)
        warnings = counts["outdated"] + counts["incorrect"] + counts["disputed"]
        parts = [f"{counts[k]} {STATUS_STYLE[k][1].lower()}" for k in STATUS_STYLE if counts[k]]
        css = "warn" if warnings else "ok"
        detail = ", ".join(parts) if parts else "no checkable claims"
        jump = ' <a href="#changed">See what changed.</a>' if warnings or check.get(
            "developments") else ""
        return (f'<div class="fresh {css}">{published}<strong>Checked against the web on '
                f'{_esc(meta.get("checked_on", ""))}:</strong> {_esc(detail)}.{jump}</div>')
    if meta.get("research_error"):
        return (f'<div class="fresh bad">{published}<strong>The web check did not run:</strong> '
                f'{_esc(meta["research_error"][:300])}. The facts here are as of the recording.'
                f'</div>')
    if published:
        return (f'<div class="fresh">{published}Not checked against the web, so the facts here '
                f'are as of the recording.</div>')
    return ""


def _minute(summary: dict) -> str:
    stats = "".join(
        f'<div class="{_stat_class(s)}"><div class="v">{inline(s["value"])}</div>'
        f'<div class="l">{inline(s["label"])}</div></div>'
        for s in summary.get("stats", []))
    takeaways = "".join(f"<li>{inline(t)}</li>" for t in summary.get("takeaways", []))
    return (
        '<section id="minute" class="minute"><h2>The 1-minute version</h2>'
        + (f'<p class="gist">{inline(summary["gist"])}</p>' if summary.get("gist") else "")
        + (f'<ul class="takeaways">{takeaways}</ul>' if takeaways else "")
        + (f'<p class="why"><strong>Why it matters:</strong> '
           f'{inline(summary["why_it_matters"])}</p>' if summary.get("why_it_matters") else "")
        + (f'<p class="stats-title">Numbers to remember</p><div class="stats">{stats}</div>'
           if stats else "")
        + _heads_up(summary)
        + "</section>")


def _heads_up(summary: dict) -> str:
    """The web check's warnings, repeated inside the one-minute version.

    Someone who reads only the first minute should not walk away with a fact the check found
    stale. The full detail is further down; this is the line that stops the damage.
    """
    warnings = [c for c in (summary.get("check") or {}).get("checks", [])
                if c.get("status") in ("outdated", "incorrect", "disputed")]
    if not warnings:
        return ""
    items = "".join(
        f"<li><strong>{_esc(STATUS_STYLE[c['status']][1])}:</strong> {inline(c['claim'])}"
        + (f" <em>Now:</em> {inline(c['now'])}" if c.get("now") else "") + "</li>"
        for c in warnings[:3])
    more = (f' <a href="#changed">{len(warnings) - 3} more</a>' if len(warnings) > 3 else "")
    return (f'<div class="heads"><strong>Heads-up from the web check:</strong> some of what '
            f'this recording says no longer holds.{more}<ul>{items}</ul></div>')


def _verdict(c: dict, source: str) -> str:
    css, label = STATUS_STYLE.get(c["status"], STATUS_STYLE["unverified"])
    frame = {"outdated": " warn", "incorrect": " bad", "disputed": " disp"}.get(c["status"], "")
    when = f" (as of {_esc(c['as_of'])})" if c.get("as_of") else ""
    links = _links(c.get("sources", []))
    return (f'<div class="verdict{frame}"><span class="pill {css}">{_esc(label)}</span>'
            f'<p class="said"><span class="lbl">Said</span>{inline(c["claim"])} '
            f'{_stamp(source, c.get("at", ""))}</p>'
            + (f'<p class="now"><span class="lbl">Now{when}</span>{inline(c["now"])}</p>'
               if c.get("now") else "")
            + (f'<p class="srcs">{links}</p>' if links else "") + "</div>")


def _changed(summary: dict) -> str:
    check = summary.get("check")
    if not check:
        return ""
    source = summary.get("meta", {}).get("source", "")
    checks = check.get("checks", [])
    warn = [c for c in checks if c["status"] in ("outdated", "incorrect", "disputed")]
    ok = [c for c in checks if c["status"] == "confirmed"]
    unknown = [c for c in checks if c["status"] == "unverified"]
    out = ['<section id="changed"><h2>What has changed since it was published</h2>']
    if check.get("overall"):
        out.append(f'<p class="overall">{inline(check["overall"])}</p>')
    out += [_verdict(c, source) for c in warn]
    if not warn and not check.get("overall"):
        out.append('<p class="overall">None of the checked claims has been overtaken since the '
                   'recording.</p>')
    if check.get("developments"):
        devs = "".join(
            '<div class="dev">'
            + (f'<span class="when">{_esc(d["date"])}</span>' if d.get("date") else "")
            + f'<strong>{inline(d["headline"])}</strong> {inline(d.get("detail", ""))}'
            + (f' <span class="srcs">{_links(d.get("sources", []))}</span>'
               if d.get("sources") else "") + "</div>"
            for d in check["developments"])
        out.append(f'<h3>New since the recording</h3>{devs}')
    if ok:
        out.append(f'<details><summary>Still accurate ({len(ok)})</summary>'
                   + "".join(_verdict(c, source) for c in ok) + "</details>")
    if unknown:
        out.append(f'<details><summary>Could not be verified ({len(unknown)})</summary>'
                   + "".join(_verdict(c, source) for c in unknown) + "</details>")
    out.append("</section>")
    return "".join(out)


def _section(s: dict, source: str) -> str:
    return (
        f'<section id="{_esc(s["id"])}">'
        f'<h2><span class="num">{s["number"]:02d}</span> {inline(s["heading"])}'
        + (f" {_stamp(source, s['starts_at'])}" if s.get("starts_at") else "") + "</h2>"
        + (f'<p class="sec-sub">{inline(s["subheading"])}</p>' if s.get("subheading") else "")
        + "".join(_update(c, source) for c in s.get("updates", []))
        + (f'<p class="sec-gist"><strong>{inline(s["gist"])}</strong></p>' if s.get("gist")
           else "")
        + (f'<div class="prose">{_paras(s.get("paragraphs", []))}</div>'
           if s.get("paragraphs") else "")
        + (f'<div class="keys"><h4>Key points</h4>{_bullets(s["key_points"])}</div>'
           if s.get("key_points") else "")
        + _blocks(s.get("blocks", []))
        + _quotes(s.get("quotes", []), source)
        + _table(s.get("table"))
        + _reco(s.get("recommendation"))
        + "</section>")


def _toc(summary: dict) -> str:
    def link(anchor: str, number: str, label: str, flag: bool = False) -> str:
        mark = ' <span class="flag" title="Has updates from the web check">!</span>' if flag \
            else ""
        return (f'<a href="#{_esc(anchor)}"><span class="n">{_esc(number)}</span> '
                f'{_esc(label)}{mark}</a>')

    top = [link("minute", "", "The 1-minute version")]
    if summary.get("check"):
        top.append(link("changed", "", "What has changed"))
    if summary.get("glossary"):
        top.append(link("terms", "", "Key terms"))
    body = [link(s["id"], f"{s['number']}", s["heading"], bool(s.get("updates")))
            for s in summary.get("sections", [])]
    tail = []
    if summary.get("worth_watching"):
        tail.append(link("watch", "", "Worth watching"))
    if summary.get("quiz"):
        tail.append(link("quiz", "", "Test yourself"))
    if summary.get("sources"):
        tail.append(link("sources", "", "Sources"))
    return ('<nav class="toc" aria-label="Contents">'
            f'<p class="toc-title">Start here</p><div class="links">{"".join(top)}</div>'
            f'<p class="toc-title">Full summary</p><div class="links">{"".join(body)}</div>'
            + (f'<p class="toc-title">Keep it</p><div class="links">{"".join(tail)}</div>'
               if tail else "") + "</nav>")


def _meta_footer(summary: dict) -> str:
    """Provenance, stated on the page itself.

    A generated page that does not say what generated it is a page you cannot judge later. This
    line is how a reader knows whether they are looking at Opus with a web check behind it or a
    4B model working from the transcript alone.
    """
    meta = summary.get("meta", {})
    bits = []
    if meta.get("media_title"):
        bits.append(f'Summarised from the Sinribe transcript of "{_esc(meta["media_title"])}"')
    if meta.get("duration"):
        bits.append(human_duration(float(meta["duration"])))
    if meta.get("speakers"):
        bits.append(f"{len(meta['speakers'])} speakers")
    line = ", ".join(bits)
    written = f"Written by {_esc(meta.get('label') or 'a language model')}"
    if meta.get("generated_at"):
        written += f" on {_esc(meta['generated_at'])}"
    if meta.get("cost_usd"):
        written += f" (${float(meta['cost_usd']):.2f} of usage)"
    if meta.get("research"):
        written += f", claims checked against the web on {_esc(meta.get('checked_on', ''))}"
    skipped = ""
    if meta.get("skipped"):
        skipped = ("<p>The preferred writer was not available: "
                   + "; ".join(_esc(s[:200]) for s in meta["skipped"]) + ".</p>")
    return (f"<footer>{f'<p>{line}.</p>' if line else ''}<p>{written}.</p>{skipped}"
            + (f"<p>{inline(summary['footer'])}</p>" if summary.get("footer") else "")
            + "</footer>")


def render_html(summary: dict) -> str:
    """The whole page as one self-contained HTML string."""
    meta = summary.get("meta", {})
    source = meta.get("source", "")
    body: list[str] = [_minute(summary), _changed(summary)]
    # Terms before the detail they are used in: learning the vocabulary first is what lets the
    # summary be read once rather than twice.
    if summary.get("glossary"):
        terms = "".join(f"<dt>{inline(t['term'])}</dt><dd>{inline(t['meaning'])}</dd>"
                        for t in summary["glossary"])
        body.append(f'<section id="terms"><h2>Key terms</h2><dl class="terms">{terms}</dl>'
                    "</section>")
    if summary.get("sections"):
        body.append('<p class="part">The full summary</p>')
        body += [_section(s, source) for s in summary["sections"]]
    if summary.get("worth_watching"):
        items = "".join(f'<li>{_stamp(source, w["at"])}<strong>{inline(w["title"])}</strong>'
                        + (f" {inline(w['why'])}" if w.get("why") else "") + "</li>"
                        for w in summary["worth_watching"])
        body.append('<section id="watch"><h2>Worth watching in the original</h2>'
                    f'<ul class="watch">{items}</ul></section>')
    if summary.get("quiz"):
        items = "".join(f'<details><summary>{inline(q["question"])}</summary>'
                        f'<p class="answer">{inline(q["answer"])}</p></details>'
                        for q in summary["quiz"])
        body.append('<section id="quiz"><h2>Test yourself</h2><p class="sec-sub">Answer in your '
                    'head first, then open the question to check.</p>' + items + "</section>")
    if summary.get("sources"):
        groups = []
        for group in summary["sources"]:
            links = _links(group["links"])
            if links:
                groups.append(f'<p><strong>{_esc(group["group"])}:</strong> {links}</p>')
        if groups:
            body.append('<section id="sources"><h2>Sources</h2><div class="src">'
                        + "".join(groups) + "</div></section>")

    lang = _esc((meta.get("language") or "en")[:8]) or "en"
    return (
        f'<!doctype html>\n<html lang="{lang}">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(_plain(summary.get('title') or 'Summary'))}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        '<div class="layout">\n'
        + _toc(summary) + "\n<main>\n"
        + _hero(summary) + "\n"
        + "\n".join(b for b in body if b) + "\n"
        + _meta_footer(summary)
        + "\n</main>\n</div>\n</body>\n</html>\n"
    )


def _plain(text: str) -> str:
    """Model text with the inline markup removed, for places markup cannot go (the <title>)."""
    return _MARKUP.sub(r"\1", str(text or "")).replace("*", "")


# --------------------------------------------------------------------------- markdown twin

def _md_block(block: dict) -> str:
    _, fallback = BLOCK_STYLE.get(block.get("kind", ""), BLOCK_STYLE["example"])
    out = [f"**{block.get('heading') or fallback}:**"]
    out += list(block.get("paragraphs", []))
    out += [f"- {b}" for b in block.get("bullets", [])]
    return "\n".join(out)


def _md_stamp(source: str, stamp: str) -> str:
    if not stamp:
        return ""
    url = jump_url(source, stamp)
    return f"[{stamp}]({url})" if url else f"[{stamp}]"


def _md_verdict(c: dict, source: str) -> list[str]:
    label = STATUS_STYLE.get(c["status"], STATUS_STYLE["unverified"])[1]
    out = [f"- **{label}:** {c['claim']} {_md_stamp(source, c.get('at', ''))}".rstrip()]
    if c.get("now"):
        out.append("  - Now" + (f" (as of {c['as_of']})" if c.get("as_of") else "")
                   + f": {c['now']}")
    if c.get("sources"):
        out.append("  - " + " · ".join(f"[{s['label']}]({s['url']})" for s in c["sources"]))
    return out


def render_markdown(summary: dict) -> str:
    """The same content as Markdown, for reading in an editor or pasting into notes."""
    meta = summary.get("meta", {})
    source = meta.get("source", "")
    out: list[str] = [f"# {summary.get('title') or 'Summary'}", ""]
    if summary.get("kicker"):
        out += [f"### {summary['kicker']}", ""]
    if source:
        out += [f"**Source:** {source}"]
    if meta.get("published"):
        age = _age(meta)
        out += [f"**Published:** {meta['published']}" + (f" ({age})" if age else "")]
    if meta.get("duration"):
        out += [f"**Length:** {human_duration(float(meta['duration']))}"]
    out += [f"**Written by:** {meta.get('label', 'a language model')}"
            + (f" on {meta['generated_at']}" if meta.get("generated_at") else "")
            + (f", claims checked against the web on {meta.get('checked_on', '')}"
               if meta.get("research") else ""), ""]
    if meta.get("research_error"):
        out += [f"> The web check did not run: {meta['research_error'][:300]}", ""]
    if summary.get("subtitle"):
        out += [summary["subtitle"], ""]

    out += ["## The 1-minute version", ""]
    if summary.get("gist"):
        out += [summary["gist"], ""]
    out += [f"- {t}" for t in summary.get("takeaways", [])]
    if summary.get("takeaways"):
        out += [""]
    if summary.get("why_it_matters"):
        out += [f"**Why it matters:** {summary['why_it_matters']}", ""]
    if summary.get("stats"):
        out += ["**Numbers to remember:**", ""]
        out += [f"- **{s['value']}** {s['label']}" for s in summary["stats"]] + [""]
    warnings = [c for c in (summary.get("check") or {}).get("checks", [])
                if c.get("status") in ("outdated", "incorrect", "disputed")]
    if warnings:
        out += ["> **Heads-up from the web check:** some of what this recording says no longer "
                "holds."]
        out += [f"> - **{STATUS_STYLE[c['status']][1]}:** {c['claim']}"
                + (f" *Now:* {c['now']}" if c.get("now") else "") for c in warnings[:3]]
        out += [""]

    check = summary.get("check")
    if check:
        out += ["## What has changed since it was published", ""]
        if check.get("overall"):
            out += [check["overall"], ""]
        for c in check.get("checks", []):
            out += _md_verdict(c, source)
        if check.get("checks"):
            out += [""]
        if check.get("developments"):
            out += ["### New since the recording", ""]
            for d in check["developments"]:
                links = " · ".join(f"[{s['label']}]({s['url']})" for s in d.get("sources", []))
                out += ["- " + (f"{d['date']}: " if d.get("date") else "")
                        + f"**{d['headline']}** {d.get('detail', '')} {links}".rstrip()]
            out += [""]

    if summary.get("glossary"):
        out += ["## Key terms", ""]
        out += [f"- **{t['term']}:** {t['meaning']}" for t in summary["glossary"]] + [""]

    out += ["## The full summary", ""]
    out += [f"{s['number']}. {s['heading']}" for s in summary.get("sections", [])] + [""]
    for s in summary.get("sections", []):
        stamp = _md_stamp(source, s.get("starts_at", ""))
        out += ["---", "", f"## {s['number']}. {s['heading']}" + (f" {stamp}" if stamp else ""),
                ""]
        if s.get("subheading"):
            out += [f"*{s['subheading']}*", ""]
        for c in s.get("updates", []):
            out += [f"> **{STATUS_STYLE[c['status']][1]} since the recording:** {c['claim']}"]
            if c.get("now"):
                out += ["> Now" + (f" (as of {c['as_of']})" if c.get("as_of") else "")
                        + f": {c['now']}"]
            out += [""]
        if s.get("gist"):
            out += [f"**{s['gist']}**", ""]
        out += [p + "\n" for p in s.get("paragraphs", [])]
        if s.get("key_points"):
            out += ["**Key points:**"] + [f"- {k}" for k in s["key_points"]] + [""]
        for block in s.get("blocks", []):
            out += [_md_block(block), ""]
        for q in s.get("quotes", []):
            who = " · ".join(x for x in (q.get("who"), _md_stamp(source, q.get("at", ""))) if x)
            out += [f"> \"{q['text']}\"" + (f"\n> {who}" if who else ""), ""]
        if s.get("table"):
            table = s["table"]
            out += ["| " + " | ".join(table["columns"]) + " |",
                    "|" + "|".join(["---"] * len(table["columns"])) + "|"]
            out += ["| " + " | ".join(c.replace("|", "\\|") for c in row) + " |"
                    for row in table["rows"]]
            out += [""]
            if table.get("caption"):
                out += [f"*{table['caption']}*", ""]
        if s.get("recommendation"):
            reco = s["recommendation"]
            out += ["**What to do with this:**"] + list(reco.get("paragraphs", []))
            out += [f"- {b}" for b in reco.get("bullets", [])] + [""]

    if summary.get("worth_watching"):
        out += ["---", "", "## Worth watching in the original", ""]
        out += [f"- {_md_stamp(source, w['at'])} **{w['title']}** {w.get('why', '')}".rstrip()
                for w in summary["worth_watching"]] + [""]
    if summary.get("quiz"):
        out += ["---", "", "## Test yourself", ""]
        for i, q in enumerate(summary["quiz"]):
            out += [f"{i + 1}. {q['question']}", f"   - *Answer:* {q['answer']}"]
        out += [""]
    if summary.get("sources"):
        out += ["---", "", "## Sources", ""]
        for group in summary["sources"]:
            links = " · ".join(f"[{link['label']}]({link['url']})" for link in group["links"])
            out += [f"**{group['group']}:** {links}", ""]

    if summary.get("footer"):
        out += ["---", "", summary["footer"], ""]
    return "\n".join(out).rstrip() + "\n"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
