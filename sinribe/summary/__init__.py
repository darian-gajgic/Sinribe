"""Summary generation: transcript in, structured briefing out.

`providers` decides who writes it (the Claude API, or the local Ollama), `build` decides what to
ask them, `schema` decides what an answer is allowed to look like. Rendering the result as a page
lives in sinribe/render/webpage.py, so the summary can be re-rendered without being regenerated.
"""

from __future__ import annotations

from .build import Cancelled, summarize, transcript_text
from .providers import PROVIDER_NAMES, SummaryError, Unavailable, can_research, describe, pick

__all__ = ["summarize", "transcript_text", "pick", "describe", "can_research", "PROVIDER_NAMES",
           "Unavailable", "SummaryError", "Cancelled"]
