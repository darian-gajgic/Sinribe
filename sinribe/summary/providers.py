"""Where the words come from: the Claude API, or the Ollama already on this box.

Two providers behind one interface, because the two halves of this feature have genuinely
different requirements. The summary is only worth reading if the model can hold a two-hour
interview in its head and find the thread through it, which on this machine means the Claude API.
But Sinribe's whole premise is that it runs offline, so the local path has to exist and has to
produce something honest rather than nothing.

The interface is deliberately narrow -- ask a question, get schema-shaped JSON back -- and the
strategy that uses it lives in build.py. What differs between providers is not the questions but
how much transcript fits in one: `context_chars` is what build.py branches on, and it is the only
capability flag that changes the shape of the work.

Failures here are loud. The pipeline's other optional stage (enrich.py) swallows everything and
returns {} on the grounds that a nice-to-have must never fail a four-hour job, and that is still
right for a stage nobody asked for. This one the user ticked a box for: if the provider is absent
or the key is wrong, they need to be told, not handed a transcript with the summary quietly
missing. So the provider is probed before the job starts and raises `Unavailable` with the fix.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Protocol

from ..config import CACHE_DIR, anthropic_key

# Kept well inside Opus 5's 1M-token window. A two-hour interview transcript is about 145 kB of
# text, so this leaves room for an order of magnitude more recording than anyone will feed it,
# and build.py never has to chunk on this path.
CLAUDE_CONTEXT_CHARS = 600_000
# gemma3:4b's window is nominally large but Ollama serves it at whatever num_ctx we ask for, and
# a 4B model's attention over 30 kB of transcript is not worth the tokens anyway. This is the
# per-call text budget that produced coherent chunk notes in testing.
OLLAMA_CONTEXT_CHARS = 6_000
# num_ctx covers the prompt AND the generation, which is the trap here: with num_ctx 8192 and a
# 6 kB prompt, a section fill asking for 4096 tokens of JSON runs out of window and is cut off
# mid-string. Grammar-constrained sampling does not save you from that -- the output is valid
# right up to where it stops, and then it is unparseable. Measured on the Kokotajlo transcript:
# section 2 died at character 5345 with an unterminated string, and the closing pass at 43.
# So the window is sized per call from the prompt actually being sent, and capped where a 4B
# model on a 16 GB card still runs at a sensible speed.
OLLAMA_NUM_PREDICT = 6_144
OLLAMA_MIN_CTX = 8_192
OLLAMA_MAX_CTX = 32_768
# Rough characters per token for prose. Deliberately pessimistic: under-estimating the prompt is
# what produces a silently truncated answer, over-estimating it just costs a little VRAM.
CHARS_PER_TOKEN = 3.0

# How many searches and page reads one research pass may run on the API. Checking a dozen
# claims properly takes roughly one search each plus a few pages opened to confirm a date or a
# figure; past this the model starts re-reading what it already found.
RESEARCH_MAX_SEARCHES = 20
RESEARCH_MAX_FETCHES = 10

# Where the `claude` CLI is run from. Deliberately not the project directory and not the output
# directory: Claude Code discovers a CLAUDE.md by walking up from its working directory, and a
# summariser that quietly inherits a project's coding conventions is a summariser whose output
# you cannot reason about. Verified empty in print mode from a directory like this one.
CLAUDE_CODE_CWD = CACHE_DIR / "claude-code"
# The CLI updates itself in the background, and npm replaces the symlink to do it. Measured the
# hard way: a summary lost its plan pass when `claude` auto-updated 70 seconds into the job and
# the next call could not find the binary. Disabled for our subprocess only, which also keeps one
# job on one version instead of switching models under itself halfway through.
CLAUDE_CODE_ENV = {"DISABLE_AUTOUPDATER": "1"}
# How many times to re-look for the binary before giving up. An npm replacement window is about a
# second, so a couple of tries a second apart covers it without masking a real absence.
CLAUDE_CODE_RESOLVE_TRIES = 4
CLAUDE_CODE_RESOLVE_WAIT = 1.0
# Where the installers put the CLI, checked after PATH. A program launched from the desktop menu
# does not see the PATH an interactive shell builds in ~/.bashrc, which is exactly where a user
# npm prefix is added. This list is what makes the default provider work from the app menu.
CLAUDE_CODE_CANDIDATES = (
    "~/.npm-global/bin/claude",
    "~/.local/bin/claude",
    "~/.claude/local/claude",
    "~/.bun/bin/claude",
    "~/.volta/bin/claude",
    "/usr/local/bin/claude",
    "/opt/homebrew/bin/claude",
)
CLAUDE_CODE_CANDIDATE_GLOBS = ("~/.nvm/versions/node/*/bin/claude",)
# The research pass may search and open pages, nothing else.
RESEARCH_TOOLS = ("WebSearch", "WebFetch")

# The current Opus. Both cloud providers take it: the API by id, the CLI as a full model name.
DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _bracket_state(text: str) -> str | None:
    """The closers `text` still needs, or None if it ends inside a string.

    Only structure matters here, so the scan tracks two things: whether we are inside a string
    literal (where brackets are just characters) and the stack of containers still open.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
        elif ch == "\\":
            escape = in_string
        elif ch == '"':
            in_string = not in_string
        elif in_string:
            continue
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
    return None if in_string else "".join(reversed(stack))


def salvage_json(text: str) -> dict | None:
    """Recover the complete part of a JSON object that was cut off mid-answer.

    A grammar-constrained decoder that runs out of room does not produce malformed JSON, it
    produces *correct* JSON that stops. Ninety percent of a section is still a section, and the
    schema layer already treats missing fields as empty, so closing the open brackets recovers a
    usable answer where strict parsing throws the whole call away.

    Works backwards from the end to the last position that parses, so nothing partial survives:
    a half-written string or a dangling comma is dropped rather than guessed at.
    """
    text = (text or "").strip()
    if not text.startswith("{"):
        return None
    for cut in range(len(text), 0, -1):
        head = text[:cut]
        # Only try where a complete value could have ended: a closed string or container, or the
        # last character of a number, `true`, `false` or `null`.
        if head[-1] not in '"}]0123456789el':      # string/object/array/number/true|false/null
            continue
        head = head.rstrip().rstrip(",")
        closers = _bracket_state(head)
        if closers is None:
            continue
        try:
            value = json.loads(head + closers)
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


class Unavailable(RuntimeError):
    """This provider cannot run, and here is what to do about it. Raised before work starts."""


class SummaryError(RuntimeError):
    """The provider ran and failed. The transcript is still fine; the page is not."""


class Provider(Protocol):
    name: str
    label: str
    context_chars: int
    supports_research: bool

    def ask_json(self, system: str, prompt: str, schema: dict, *,
                 max_tokens: int = 8000, cache_prefix: str = "") -> dict: ...

    def ask_text(self, system: str, prompt: str, *, max_tokens: int = 8000) -> str: ...

    def research(self, system: str, prompt: str, *,
                 max_tokens: int = 32_000) -> tuple[str, list[dict]]:
        """Free text written with web search and page reads, plus every source the tools
        actually returned as {url, title, kind}. Only providers with supports_research."""
        ...


# --------------------------------------------------------------------------- Claude

class _Logging:
    """A place for a provider to say something the user should see.

    build.summarize sets this so a provider can report a recovered answer without either
    swallowing it or raising. Defaults to a no-op so a provider is usable on its own.
    """

    on_log: Callable[[str], None] = staticmethod(lambda _msg: None)
    # Providers "auto" tried before this one, each with the reason it could not run.
    skipped: list[str] = []


class ClaudeProvider(_Logging):
    """Claude via the official SDK. Opus 5.5 by default; the model is a config key, not a constant.

    Every call streams. Section fills ask for up to 16k output tokens and the SDK's non-streaming
    path would rather time out than wait for that, which is exactly the failure mode you do not
    want two minutes into a paid call.
    """

    name = "claude"
    supports_research = True
    context_chars = CLAUDE_CONTEXT_CHARS

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "high",
                 timeout: float = 900.0):
        self.model = model or DEFAULT_MODEL
        # Set explicitly on every call: Opus 5.5 defaults to "medium", one level below what the
        # section fills want.
        self.effort = effort if effort in EFFORTS else "high"
        self.label = f"Claude {self.model}"
        self._timeout = timeout
        self._client = None
        # Server-side refusal fallbacks are a beta, so the parameter can be rejected outright by
        # an SDK older than the feature. Rather than pin a version, the first rejection turns it
        # off for the rest of the run and the call is retried plain.
        self._use_fallbacks = True

    # -- plumbing ----------------------------------------------------------------
    def _sdk(self):
        try:
            import anthropic
        except ImportError as e:
            raise Unavailable(
                "the `anthropic` package is not installed in Sinribe's venv. Install it with:\n"
                "    uv pip install --python .venv/bin/python anthropic\n"
                "or choose the local Ollama provider in Options."
            ) from e
        return anthropic

    def client(self):
        if self._client is None:
            anthropic = self._sdk()
            key = anthropic_key()
            try:
                # No explicit key is not the same as no credentials: the SDK also resolves
                # ANTHROPIC_AUTH_TOKEN, an `ant auth login` profile and workload identity, so it
                # is given the chance to find one. Whether it did is what probe() then checks --
                # construction itself succeeds either way.
                self._client = (anthropic.Anthropic(api_key=key, timeout=self._timeout)
                                if key else anthropic.Anthropic(timeout=self._timeout))
            except Exception as e:  # noqa: BLE001 - the SDK's own no-credentials error
                raise Unavailable(
                    f"no Claude API credentials: {e}. Set ANTHROPIC_API_KEY, or write the key to "
                    f"~/.config/sinribe/anthropic_key, or choose the local Ollama provider in "
                    f"Options."
                ) from e
        return self._client

    def probe(self) -> None:
        """Cheap pre-flight: the SDK imports, and a credential actually resolved. No request.

        Constructing the client is NOT enough on its own. `anthropic.Anthropic()` with nothing
        configured returns a perfectly good client object and only raises at request time, so a
        probe that stops at construction reports "ready" on a machine with no key at all, and
        the user finds out after the transcription instead of before it. What is checked here is
        the thing the SDK itself checks when the request goes out: that one of api_key,
        auth_token or credentials is set, whichever of the environment, a profile or workload
        identity supplied it.
        """
        client = self.client()
        if not any(getattr(client, attr, None)
                   for attr in ("api_key", "auth_token", "credentials")):
            raise Unavailable(
                "no Claude API credentials were found. Set ANTHROPIC_API_KEY, write the key to "
                "~/.config/sinribe/anthropic_key, or run `ant auth login`. Alternatively choose "
                "the local Ollama provider in Options."
            )

    def _content(self, prompt: str, cache_prefix: str) -> list[dict]:
        """User content, with the transcript in its own cached block.

        The transcript is the same bytes on every call of a job and it dwarfs everything else, so
        it goes first behind a cache breakpoint: the plan pass pays for it once and the dozen
        section fills that follow read it from cache. Prefix order matters -- anything that varies
        per call has to come after the breakpoint or there is nothing stable to match.
        """
        content: list[dict] = []
        if cache_prefix:
            content.append({"type": "text", "text": cache_prefix,
                            "cache_control": {"type": "ephemeral"}})
        content.append({"type": "text", "text": prompt})
        return content

    def _stream(self, **kwargs):
        """One streamed request, with refusal fallbacks when the SDK understands them."""
        client = self.client()
        anthropic = self._sdk()
        if self._use_fallbacks:
            try:
                with client.beta.messages.stream(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                ) as stream:
                    return stream.get_final_message()
            except (TypeError, anthropic.BadRequestError):
                # Either this SDK has no `fallbacks` parameter or the account is not in the beta.
                # Neither is worth failing the summary over; drop the feature and carry on.
                self._use_fallbacks = False
        with client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    def _text_of(self, message) -> str:
        # stop_details is populated only for refusals, so this is the one stop_reason worth
        # branching on: everything else either produced content or raised.
        if getattr(message, "stop_reason", None) == "refusal":
            detail = getattr(message, "stop_details", None)
            category = getattr(detail, "category", None) or "unspecified"
            raise SummaryError(
                f"Claude declined to summarise this recording (category: {category}). "
                f"The transcript itself was written normally."
            )
        return "".join(b.text for b in message.content if b.type == "text").strip()

    def _call(self, **kwargs):
        anthropic = self._sdk()
        try:
            return self._stream(**kwargs)
        except anthropic.AuthenticationError as e:
            raise Unavailable(f"Claude rejected the API key: {e}") from e
        except anthropic.NotFoundError as e:
            raise Unavailable(
                f"the model {self.model} is not available to this account: {e}") from e
        except anthropic.RateLimitError as e:
            raise SummaryError(f"Claude rate limit reached: {e}") from e
        except anthropic.APIStatusError as e:
            raise SummaryError(f"Claude API error {e.status_code}: {e}") from e
        except anthropic.APIConnectionError as e:
            raise SummaryError(f"could not reach the Claude API: {e}") from e

    # -- interface ---------------------------------------------------------------
    def ask_json(self, system: str, prompt: str, schema: dict, *,
                 max_tokens: int = 8000, cache_prefix: str = "") -> dict:
        message = self._call(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": self._content(prompt, cache_prefix)}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": schema}},
        )
        text = self._text_of(message)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Structured output guarantees valid JSON, so this means the response was truncated
            # against max_tokens rather than malformed. What arrived is still usable.
            recovered = salvage_json(text)
            if recovered is not None:
                self.on_log(f"Claude's answer was cut off at {len(text)} characters "
                            f"(stop_reason {getattr(message, 'stop_reason', '?')}); "
                            f"using the part that arrived")
                return recovered
            raise SummaryError(
                f"Claude's answer was not complete JSON ({e}); "
                f"stop_reason was {getattr(message, 'stop_reason', '?')}") from e

    def ask_text(self, system: str, prompt: str, *, max_tokens: int = 8000) -> str:
        message = self._call(model=self.model, max_tokens=max_tokens, system=system,
                             messages=[{"role": "user", "content": prompt}],
                             output_config={"effort": self.effort})
        return self._text_of(message)

    def research(self, system: str, prompt: str, *,
                 max_tokens: int = 32_000) -> tuple[str, list[dict]]:
        """Free-form text written with web search and web fetch available.

        Structured output is deliberately NOT combined with the web tools here. The research
        pass wants citations and running prose, the JSON passes want a fixed shape, and asking
        for both at once buys a constrained decoder fighting a tool loop. The report is turned
        into structured verdicts by a separate tool-free call, and the sources are taken from
        the tool result blocks, which is the only record of what a search really returned.
        """
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "output_config": {"effort": self.effort},
            "tools": [{"type": "web_search_20260209", "name": "web_search",
                       "max_uses": RESEARCH_MAX_SEARCHES},
                      {"type": "web_fetch_20260209", "name": "web_fetch",
                       "max_uses": RESEARCH_MAX_FETCHES}],
        }
        messages: list = [{"role": "user", "content": prompt}]
        chunks: list[str] = []
        sources: list[dict] = []
        # A server-tool turn can stop with pause_turn partway through its searches. That is not an
        # error and not the end of the answer: the turn is resumed by sending it back.
        for _ in range(6):
            message = self._call(messages=messages, **kwargs)
            chunks.append(self._text_of(message))
            sources += _api_sources(message)
            if getattr(message, "stop_reason", None) != "pause_turn":
                break
            messages = messages + [{"role": "assistant", "content": message.content}]
        return "\n".join(c for c in chunks if c).strip(), sources


def _api_sources(message) -> list[dict]:
    """Every URL a web_search or web_fetch result block carried, plus cited URLs."""
    found: list[dict] = []
    for block in getattr(message, "content", None) or []:
        kind = getattr(block, "type", "")
        content = getattr(block, "content", None)
        if kind == "web_search_tool_result" and isinstance(content, list):
            for hit in content:
                url = str(getattr(hit, "url", "") or "")
                if url.startswith("http"):
                    found.append({"url": url, "title": str(getattr(hit, "title", "") or ""),
                                  "kind": "search",
                                  "page_age": str(getattr(hit, "page_age", "") or "")})
        elif kind == "web_fetch_tool_result" and getattr(content, "type", "") == "web_fetch_result":
            url = str(getattr(content, "url", "") or "")
            if url.startswith("http"):
                found.append({"url": url, "title": "", "kind": "opened"})
        elif kind == "text":
            for cite in getattr(block, "citations", None) or []:
                url = str(getattr(cite, "url", "") or "")
                if url.startswith("http"):
                    found.append({"url": url, "title": str(getattr(cite, "title", "") or ""),
                                  "kind": "cited"})
    return found


# --------------------------------------------------------------------------- Claude Code

class ClaudeCodeProvider(_Logging):
    """Claude through the `claude` CLI in print mode, using the login already on this machine.

    This is the provider that needs no API key, and it is the default for a reason: the page this
    whole feature imitates was written this way. `claude -p` authenticates with the same
    claude.ai session the user's own Claude Code uses, so a summary costs subscription usage
    rather than a separate key and a separate bill.

    What it gives up against the API provider is decoder-enforced structured output: there is no
    `output_config.format` here, so the schema goes in the prompt and the answer is parsed
    hopefully rather than guaranteed. In practice Opus returns clean JSON with no fence and no
    preamble, and `salvage_json` covers the rest.

    What it gains is a real conversation. Each job runs as ONE resumed session: the transcript is
    sent once, and every later pass continues that session, so the transcript is read from cache
    (measured: 2 input tokens against 11,677 read from cache on a resumed turn) and each section
    is written with the earlier sections already in view. That is better for coherence than
    handing every section an identical cached prefix, and worse for cost: every resumed turn
    re-reads the conversation so far.
    """

    name = "claude-code"
    supports_research = True
    context_chars = CLAUDE_CONTEXT_CHARS

    def __init__(self, model: str = DEFAULT_MODEL, timeout: float = 900.0,
                 binary: str = "claude", effort: str = "high"):
        self.model = model or DEFAULT_MODEL
        self.effort = effort if effort in EFFORTS else "high"
        self.label = f"Claude Code ({self.model})"
        self._timeout = timeout
        self._binary = binary or "claude"
        # One session per job, so the transcript is sent once. Reset if the prefix ever changes.
        self._session_id: str | None = None
        self._prefix_sent = ""
        # Resolved once and reused, so a job is not re-running `which` fifteen times against a
        # path its own updater may be rewriting.
        self._resolved: str | None = None
        # What this job has cost so far, as the CLI reports it. Subscription usage rather than a
        # bill, but the user should still be told what a summary spends.
        self.cost_usd = 0.0

    # -- plumbing ----------------------------------------------------------------
    def _locate(self) -> str | None:
        """PATH first, then the places the installers put the CLI.

        PATH alone is not enough, and this is the bug that made the default provider unusable
        from the app menu: `npm install -g` with a user prefix puts `claude` in ~/.npm-global/bin
        and adds that to PATH in ~/.bashrc, which only interactive shells read. A program started
        from the desktop gets the session's PATH instead, so `which claude` found nothing and
        the summary quietly went to the 4B local model.
        """
        if os.sep in self._binary:
            path = Path(self._binary).expanduser()
            return str(path) if path.is_file() and os.access(path, os.X_OK) else None
        found = shutil.which(self._binary)
        if found:
            return found
        if self._binary != "claude":
            return None
        candidates = [Path(c).expanduser() for c in CLAUDE_CODE_CANDIDATES]
        prefix = os.environ.get("NPM_CONFIG_PREFIX") or os.environ.get("npm_config_prefix")
        if prefix:
            candidates.insert(0, Path(prefix).expanduser() / "bin" / "claude")
        for pattern in CLAUDE_CODE_CANDIDATE_GLOBS:
            # Newest node version first, which is the one an nvm user is most likely running.
            candidates += [Path(m) for m in sorted(glob.glob(os.path.expanduser(pattern)),
                                                   reverse=True)]
        for path in candidates:
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        return None

    def _which(self, patient: bool = True) -> str:
        """Locate the CLI, tolerating the moment its own updater is replacing the symlink.

        `patient` is the difference between the two callers. Mid-job, waiting a few seconds for
        an npm replacement to finish is obviously right. At probe time it is obviously wrong:
        the probe runs on the GUI thread every time the provider dropdown changes, and a
        three-second freeze to confirm something the user can see is not installed is worse than
        the answer is useful.
        """
        if self._resolved and Path(self._resolved).exists():
            return self._resolved
        tries = CLAUDE_CODE_RESOLVE_TRIES if patient else 1
        for attempt in range(tries):
            found = self._locate()
            if found:
                if attempt:
                    self.on_log(f"`{self._binary}` reappeared after {attempt} "
                                f"{'retry' if attempt == 1 else 'retries'}; it was most likely "
                                f"being replaced by its own updater")
                self._resolved = found
                return found
            if attempt + 1 < tries:
                time.sleep(CLAUDE_CODE_RESOLVE_WAIT)
        raise Unavailable(
            f"the `{self._binary}` command was not found on PATH or in "
            f"~/.npm-global/bin, ~/.local/bin or ~/.claude/local, so the Claude Code provider "
            f"cannot run. Install it with `npm install -g @anthropic-ai/claude-code`, or set "
            f"\"claude_binary\" in ~/.config/sinribe/config.json to its full path."
        )

    def _env(self) -> dict:
        env = dict(os.environ)
        env.update(CLAUDE_CODE_ENV)
        if self._resolved:
            # Whatever the CLI itself shells out to lives next to it more often than not.
            folder = str(Path(self._resolved).parent)
            if folder not in env.get("PATH", "").split(os.pathsep):
                env["PATH"] = folder + os.pathsep + env.get("PATH", "")
        return env

    def probe(self) -> None:
        """`claude auth status` — a local read of the stored login, no API call and no cost."""
        binary = self._which(patient=False)
        try:
            out = subprocess.run([binary, "auth", "status"], capture_output=True, text=True,
                                 timeout=30, check=False, cwd=self._cwd(), env=self._env())
        except (OSError, subprocess.TimeoutExpired) as e:
            raise Unavailable(f"could not run `{self._binary} auth status`: {e}") from e
        try:
            status = json.loads(out.stdout or "{}")
        except json.JSONDecodeError:
            # An older CLI may not print JSON here. Not worth failing over: a working binary is
            # most of the check, and a bad login surfaces on the first call either way.
            status = {}
        if status and not status.get("loggedIn"):
            raise Unavailable(
                f"`{self._binary}` is installed but not logged in. Run `{self._binary} auth "
                f"login` once, or choose a different provider in Options."
            )

    def _cwd(self) -> Path:
        CLAUDE_CODE_CWD.mkdir(parents=True, exist_ok=True)
        return CLAUDE_CODE_CWD

    def _args(self, system: str, resume: str | None, research: bool) -> list[str]:
        args = [self._which(), "-p", "--model", self.model, "--effort", self.effort,
                # No Bash, no code execution, no WebFetch unless named below, no user settings,
                # no MCP servers, no skills: a summariser has no business touching this machine,
                # and every MCP server the user has configured would otherwise start per call.
                "--restricted", "--strict-mcp-config", "--disable-slash-commands",
                "--system-prompt", system]
        if research:
            # The tools it needs and the mode that lets it run without a human at the keyboard.
            # stream-json is what exposes each search's result list and each fetched page, so
            # the sources can be taken from what the tools returned instead of from the prose.
            args += ["--output-format", "stream-json", "--verbose",
                     "--tools", *RESEARCH_TOOLS, "--allowedTools", *RESEARCH_TOOLS,
                     "--permission-mode", "dontAsk", "--no-session-persistence"]
        else:
            args += ["--output-format", "json", "--tools", ""]
        if resume:
            args += ["--resume", resume]
        return args

    def _run(self, system: str, body: str, *, resume: str | None,
             research: bool) -> tuple[str, str, list[dict]]:
        """One `claude -p` call. Returns (text, session_id, sources the tools returned)."""
        args = self._args(system, resume, research)
        try:
            # The prompt goes on stdin, not in argv: a two-hour transcript is 145 kB and argv is
            # not the place for it.
            out = subprocess.run(args, input=body, capture_output=True, text=True,
                                 timeout=self._timeout, check=False, cwd=self._cwd(),
                                 env=self._env())
        except subprocess.TimeoutExpired as e:
            raise SummaryError(
                f"`{self._binary} -p` did not finish within {self._timeout:.0f}s") from e
        except OSError as e:
            raise SummaryError(f"could not run `{self._binary} -p`: {e}") from e

        if out.returncode != 0 and not (out.stdout or "").strip():
            detail = (out.stderr or "").strip()[:400] or f"exit {out.returncode}"
            raise SummaryError(f"`{self._binary} -p` failed: {detail}")
        envelope, sources = (_stream_envelope(out.stdout) if research
                             else (_json_envelope(out.stdout), []))
        if envelope is None:
            detail = (out.stderr or out.stdout or "").strip()[:300]
            raise SummaryError(f"`{self._binary} -p` returned no result envelope: {detail}")

        self.cost_usd += float(envelope.get("total_cost_usd") or 0.0)
        if envelope.get("is_error") or out.returncode != 0:
            raise SummaryError(
                f"Claude Code reported an error ({envelope.get('subtype') or out.returncode}): "
                f"{str(envelope.get('result') or out.stderr or '')[:300]}")
        denials = envelope.get("permission_denials") or []
        if denials:
            names = ", ".join(sorted({str(d.get("tool_name") or "?") for d in denials}))
            self.on_log(f"Claude Code was denied {names} on this call; it is sandboxed to "
                        f"reading nothing and running nothing")
        return (str(envelope.get("result") or ""), str(envelope.get("session_id") or ""),
                sources)

    def _ask(self, system: str, prompt: str, cache_prefix: str) -> str:
        """Send `prompt`, resuming the job's session when the transcript is already in it."""
        resume = self._session_id if (
            cache_prefix and cache_prefix == self._prefix_sent and self._session_id) else None
        body = prompt if resume else (
            f"{cache_prefix}\n\n{prompt}" if cache_prefix else prompt)
        text, session_id, _ = self._run(system, body, resume=resume, research=False)
        if session_id and not resume and cache_prefix:
            self._session_id, self._prefix_sent = session_id, cache_prefix
        return text

    # -- interface ---------------------------------------------------------------
    def ask_json(self, system: str, prompt: str, schema: dict, *,
                 max_tokens: int = 8000, cache_prefix: str = "") -> dict:
        # The schema goes in the USER turn rather than the system prompt on purpose: a resumed
        # session carries the system prompt it was created with, and each pass needs a different
        # schema. Put it in the system prompt and every section after the first would be asked
        # for the plan's shape.
        asked = (f"{prompt}\n\nReply with a single JSON object and nothing else: no "
                 f"markdown fence, no commentary before or after. It must match this "
                 f"JSON Schema:\n"
                 f"{json.dumps(schema, separators=(',', ':'))}")
        text = _unfence(self._ask(system, asked, cache_prefix))
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            recovered = salvage_json(text)
            if recovered is not None:
                self.on_log(f"Claude Code's answer needed repairing at {len(text)} characters; "
                            f"using the part that parsed")
                return recovered
            raise SummaryError(
                f"Claude Code did not return usable JSON ({e}); "
                f"the answer began {text[:160]!r}") from e

    def ask_text(self, system: str, prompt: str, *, max_tokens: int = 8000) -> str:
        text, _sid, _ = self._run(system, prompt, resume=None, research=False)
        return text.strip()

    def research(self, system: str, prompt: str, *,
                 max_tokens: int = 32_000) -> tuple[str, list[dict]]:
        """Search and read the web. Returns the report and every source the tools returned.

        Runs in its own session, never resumed and never saved: a different job, a different
        system prompt, and nothing it reads should displace the transcript in the summary's
        session.
        """
        text, _sid, sources = self._run(system, prompt, resume=None, research=True)
        return text.strip(), sources


def _json_envelope(stdout: str) -> dict | None:
    try:
        envelope = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return None
    return envelope if isinstance(envelope, dict) else None


def _stream_envelope(stdout: str) -> tuple[dict | None, list[dict]]:
    """The final result event of a stream-json run, plus the sources its tools returned.

    WebSearch reports its hits as `tool_use_result.results[].content[] = {title, url}`, and
    WebFetch reports the page it opened as `tool_use_result = {url, code, ...}`. Those are the
    only URLs that count as seen: the report's own prose is the model's word for it.
    """
    envelope: dict | None = None
    sources: list[dict] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            envelope = event
        elif event.get("type") == "user":
            result = event.get("tool_use_result")
            if isinstance(result, dict):
                sources += _tool_sources(result)
    return envelope, sources


def _tool_sources(result: dict) -> list[dict]:
    if "url" in result and "code" in result:
        # A fetch. Only a page that actually came back counts as opened.
        code = result.get("code")
        ok = isinstance(code, int) and 200 <= code < 400
        return [{"url": str(result["url"]), "title": "", "kind": "opened"}] if ok else []
    found = []
    for hit in result.get("results") or []:
        if not isinstance(hit, dict):
            continue
        for item in hit.get("content") or []:
            if isinstance(item, dict) and str(item.get("url", "")).startswith("http"):
                found.append({"url": str(item["url"]), "title": str(item.get("title") or ""),
                              "kind": "search"})
    return found


def _unfence(text: str) -> str:
    """Strip a ```json fence if the model wrapped its answer in one."""
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


# --------------------------------------------------------------------------- Ollama

class OllamaProvider(_Logging):
    """The local model, talked to exactly the way enrich.py talks to it: raw urllib, no SDK.

    Schema-constrained rather than merely asked-nicely: Ollama accepts a JSON Schema in `format`
    and constrains sampling to it, which is what makes a 4B model usable as a structured
    summariser at all. It still writes thin prose, and build.py compensates by asking smaller
    questions of it -- but every answer parses.
    """

    name = "ollama"
    supports_research = False
    context_chars = OLLAMA_CONTEXT_CHARS

    def __init__(self, url: str = "http://127.0.0.1:11435", model: str = "gemma3:4b",
                 timeout: float = 240.0):
        self.url = url.rstrip("/")
        self.model = model
        self.label = f"{model} (local)"
        self._timeout = timeout
        self._last_done_reason = ""

    def probe(self) -> None:
        from ..enrich import available
        if not available(self.url, self.model):
            raise Unavailable(
                f"Ollama at {self.url} is not serving {self.model}. Start it and run "
                f"`ollama pull {self.model}`, or choose the Claude provider in Options."
            )

    def _window(self, prompt_chars: int, predict: int) -> int:
        """A context window that holds this prompt and its answer, within the cap."""
        need = int(prompt_chars / CHARS_PER_TOKEN) + predict + 512
        return max(OLLAMA_MIN_CTX, min(OLLAMA_MAX_CTX, need))

    def _options(self, prompt: str, max_tokens: int) -> dict:
        predict = min(max_tokens, OLLAMA_NUM_PREDICT)
        return {"temperature": 0, "num_predict": predict,
                "num_ctx": self._window(len(prompt), predict)}

    def _generate(self, payload: dict) -> str:
        req = urllib.request.Request(
            self.url + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError,
                json.JSONDecodeError, TimeoutError) as e:
            raise SummaryError(f"Ollama call failed: {e}") from e
        # Kept so a truncated answer can say WHY it was truncated. "length" means the window ran
        # out, which is a sizing bug on our side, not a model that had nothing more to say.
        self._last_done_reason = str(body.get("done_reason") or "")
        return body.get("response") or ""

    def ask_json(self, system: str, prompt: str, schema: dict, *,
                 max_tokens: int = 8000, cache_prefix: str = "") -> dict:
        body = f"{cache_prefix}\n\n{prompt}" if cache_prefix else prompt
        # Same pattern-completion framing as enrich.py: presented as Input/Output the model
        # transforms the text instead of replying to it. Without it, a transcript containing a
        # question comes back answered.
        framed = f"Input: {body}\nOutput:"
        text = self._generate({
            "model": self.model,
            "system": system,
            "prompt": framed,
            "stream": False,
            "format": schema,
            "keep_alive": "5m",
            "options": self._options(framed, max_tokens),
        })
        try:
            return json.loads(text or "{}")
        except json.JSONDecodeError as e:
            recovered = salvage_json(text)
            if recovered is not None:
                self.on_log(f"{self.model} was cut off at {len(text)} characters "
                            f"(stopped because: {self._last_done_reason or 'unknown'}); "
                            f"using the part that arrived")
                return recovered
            raise SummaryError(
                f"Ollama returned unparseable JSON after {len(text)} characters "
                f"(stopped because: {self._last_done_reason or 'unknown'}): {e}") from e

    def ask_text(self, system: str, prompt: str, *, max_tokens: int = 8000) -> str:
        framed = f"Input: {prompt}\nOutput:"
        return self._generate({
            "model": self.model,
            "system": system,
            "prompt": framed,
            "stream": False,
            "keep_alive": "5m",
            "options": self._options(framed, max_tokens),
        }).strip()

    def research(self, system: str, prompt: str, *,
                 max_tokens: int = 32_000) -> tuple[str, list[dict]]:
        raise Unavailable("web research needs a Claude provider; the local model is offline.")


# --------------------------------------------------------------------------- selection

AnyProvider = ClaudeCodeProvider | ClaudeProvider | OllamaProvider

# What "auto" tries, in order. Claude Code first because it needs nothing configured: it uses the
# login the user already has, which is how the page this feature imitates was written in the
# first place. The API provider is the option for someone who has a key and wants the
# decoder-enforced schema; the local model is the offline floor.
AUTO_ORDER = ("claude-code", "claude", "ollama")
PROVIDER_NAMES = AUTO_ORDER


def build_provider(cfg: dict, name: str) -> AnyProvider:
    model = str(cfg.get("summary_model") or DEFAULT_MODEL)
    if name == "claude-code":
        return ClaudeCodeProvider(model=model,
                                  timeout=float(cfg.get("summary_timeout", 900)),
                                  binary=str(cfg.get("claude_binary") or "claude"),
                                  effort=str(cfg.get("summary_effort") or "high"))
    if name == "claude":
        return ClaudeProvider(model=model,
                              effort=str(cfg.get("summary_effort") or "high"),
                              timeout=float(cfg.get("summary_timeout", 900)))
    return OllamaProvider(url=str(cfg.get("llm_url") or "http://127.0.0.1:11435"),
                          model=str(cfg.get("llm_model") or "gemma3:4b"),
                          timeout=float(cfg.get("llm_timeout", 180)) + 120.0)


def pick(cfg: dict) -> AnyProvider:
    """The provider this config asks for, probed and ready, or `Unavailable` explaining why not.

    Called before the job's expensive stages, so a missing login costs a dialog rather than two
    hours followed by a dialog.
    """
    want = str(cfg.get("summary_provider") or "auto")
    if want in PROVIDER_NAMES:
        provider = build_provider(cfg, want)
        provider.probe()
        return provider

    reasons = []
    for name in AUTO_ORDER:
        provider = build_provider(cfg, name)
        try:
            provider.probe()
        except Unavailable as e:
            reasons.append(f"  {name}: {e}")
            continue
        # Kept on the provider so the caller can say out loud that "auto" fell back, and why.
        # A silent fallback is how a brief came to be written by a 4B model while the user
        # believed Claude was writing it.
        provider.skipped = [r.strip() for r in reasons]
        return provider
    raise Unavailable("no summary provider is usable.\n" + "\n".join(reasons))


def can_research(name: str) -> bool:
    """Can this provider search the web? Asked of the class, so the UI cannot drift from it."""
    return {"claude-code": ClaudeCodeProvider, "claude": ClaudeProvider,
            "ollama": OllamaProvider}.get(name, OllamaProvider).supports_research


def describe(cfg: dict) -> str:
    """One line for the UI: which provider a summary would use right now, without running one."""
    try:
        return pick(cfg).label
    except Unavailable:
        return "unavailable"

