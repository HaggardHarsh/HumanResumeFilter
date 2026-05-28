"""
config.py — Central configuration for the AI Candidate Ranking Pipeline.

Reads API keys from environment variables and provides typed settings
for every stage of the pipeline.
"""

import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """All tuneable knobs for the ranking pipeline."""

    # ── Embedding ──────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Retrieval ──────────────────────────────────────────────────────
    retrieval_top_n: int = 50          # first-pass shortlist size
    hybrid_alpha: float = 0.70         # weight for semantic score (1-α → BM25)
    enable_hybrid_search: bool = True  # set False to use pure semantic search

    # ── LLM Re-Ranking ────────────────────────────────────────────────
    llm_top_n: int = 20               # how many candidates to send to LLM
    llm_provider: str = "auto"        # "groq" | "xai" | "openai" | "gemini" | "auto"
    llm_model: str = ""               # resolved per-provider if left blank
    llm_temperature: float = 0.2
    llm_max_retries: int = 3
    llm_retry_delay: float = 2.0      # seconds between retries

    # ── Output ─────────────────────────────────────────────────────────
    output_format: str = "csv"         # "csv" | "json"

    # ── Internal (resolved at runtime) ─────────────────────────────────
    _resolved_provider: str = field(default="", init=False, repr=False)
    _api_key: str = field(default="", init=False, repr=False)
    _base_url: str = field(default="", init=False, repr=False)
    _resolved_model: str = field(default="", init=False, repr=False)

    # ── Provider defaults ──────────────────────────────────────────────
    PROVIDER_DEFAULTS: dict = field(default_factory=lambda: {
        "xai": {
            "env_key": "XAI_API_KEY",
            "base_url": "https://api.x.ai/v1",
            "model": "grok-3-mini-fast",
        },
        "openai": {
            "env_key": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        },
        "groq": {
            "env_key": "GROQ_API_KEY",
            "base_url": "https://api.groq.com/openai/v1",
            "model": "llama-3.3-70b-versatile",
        },
        "gemini": {
            "env_key": "GEMINI_API_KEY",
            "base_url": "",  # uses google-generativeai SDK
            "model": "gemini-2.0-flash",
        },
    }, init=False, repr=False)

    def __post_init__(self) -> None:
        self._resolve_llm_provider()

    # ──────────────────────────────────────────────────────────────────
    def _resolve_llm_provider(self) -> None:
        """Detect which LLM provider to use based on available API keys."""
        if self.llm_provider != "auto":
            provider = self.llm_provider.lower()
            if provider not in self.PROVIDER_DEFAULTS:
                raise ValueError(
                    f"Unknown LLM provider '{provider}'. "
                    f"Choose from: {list(self.PROVIDER_DEFAULTS.keys())}"
                )
            defaults = self.PROVIDER_DEFAULTS[provider]
            api_key = os.getenv(defaults["env_key"], "")
            if not api_key:
                logger.warning(
                    "Provider '%s' selected but %s is not set. "
                    "LLM re-ranking will be skipped.",
                    provider,
                    defaults["env_key"],
                )
            self._resolved_provider = provider
            self._api_key = api_key
            self._base_url = defaults["base_url"]
            self._resolved_model = self.llm_model or defaults["model"]
            return

        # Auto-detect: try Groq → xAI → OpenAI → Gemini
        for provider, defaults in self.PROVIDER_DEFAULTS.items():
            api_key = os.getenv(defaults["env_key"], "")
            if api_key:
                logger.info("Auto-detected LLM provider: %s", provider)
                self._resolved_provider = provider
                self._api_key = api_key
                self._base_url = defaults["base_url"]
                self._resolved_model = self.llm_model or defaults["model"]
                return

        logger.warning(
            "No LLM API key found (checked GROQ_API_KEY, XAI_API_KEY, OPENAI_API_KEY, "
            "GEMINI_API_KEY). LLM re-ranking will be skipped; retrieval "
            "scores will be used instead."
        )
        self._resolved_provider = "none"
        self._api_key = ""
        self._base_url = ""
        self._resolved_model = ""

    # ── Convenience properties ────────────────────────────────────────
    @property
    def has_llm(self) -> bool:
        return bool(self._api_key)

    @property
    def resolved_provider(self) -> str:
        return self._resolved_provider

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def resolved_model(self) -> str:
        return self._resolved_model
