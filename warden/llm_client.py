"""Shared LLM client plumbing.

Used by BOTH the M5 workload generator (test harness) and the M3 classifier's
escalation oracle (the system under test). It lives here so the product never
has to import from the harness -- the two are separate layers and must stay
that way (DECISIONS.md D22).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

MODEL = "claude-haiku-4-5-20251001"

# Workload generation is a low-difficulty instruction-following task (write one
# realistic business sentence in a given style), so any competent model will do.
# Using a NON-Claude model here is in fact preferable: it removes the mild
# circularity of one model family both writing the corpus and adjudicating
# escalations in M3 (DECISIONS.md D20).
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI_COMPAT = "openai_compat"

# base_url / default model / env var holding the key, for known free tiers
KNOWN_PROVIDERS = {
    # NB: provider model catalogues rot. If a call 404s, list what the account
    # actually serves: GET <base_url>/models (the error below says so too).
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai",
               "gemini-2.0-flash", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free",
                   "OPENROUTER_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "mistral-small-latest", "MISTRAL_API_KEY"),
    "ollama": ("http://localhost:11434/v1", "llama3.1", "OLLAMA_API_KEY"),
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "llm_workload_cache.json"


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Read KEY=value lines from .env into os.environ without overwriting
    anything already set. Keeps the API key out of shell history and out of
    any transcript -- it only ever lives in an untracked file on disk."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))
class _Cache:
    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self._data: dict[str, str] = {}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def hash(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]

    def get(self, key: str) -> Optional[str]:
        return self._data.get(self.hash(key))

    def put(self, key: str, value: str) -> None:
        self._data[self.hash(key)] = value

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=1, sort_keys=True), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._data)


REFUSAL_PATTERN = re.compile(
    r"(i'm sorry|i am sorry|can't help|cannot help|can't assist|cannot assist|"
    r"unable to (help|assist|comply)|as an ai|i won't)", re.IGNORECASE)


class UnusableResponse(RuntimeError):
    """The model returned something that must never enter the corpus."""


def validate_text(text: str) -> str:
    """Reject anything unusable rather than caching it.

    A corpus row that is empty, a refusal, or a multi-sentence ramble is
    still a perfectly valid-looking row in writes.csv -- it would flow into
    published metrics with no visible symptom. The first pilot had 53 empty
    rows (26.5%) and one refusal before this existed."""
    cleaned = " ".join((text or "").split()).strip().strip('"')
    normalized = cleaned.replace("’", "'").replace("‘", "'")
    if not normalized:
        raise UnusableResponse("empty response (reasoning models can exhaust max_tokens)")
    if REFUSAL_PATTERN.search(normalized):
        raise UnusableResponse(f"refusal: {cleaned[:60]!r}")
    if len(cleaned) > 300:
        raise UnusableResponse(f"over-long ({len(cleaned)} chars), likely preamble or multi-sentence")
    return cleaned


class OpenAICompatClient:
    """Minimal /chat/completions client. Covers Groq, Gemini, OpenRouter,
    Mistral, Together and local Ollama -- they all speak this shape. Uses
    httpx (already a transitive dependency) rather than adding an SDK."""

    MAX_RETRIES = 6
    NETWORK_RETRIES = 9       # transport errors get a longer window than HTTP errors
    NETWORK_BACKOFF_CAP = 60  # ~9 min of tolerance for a network outage

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        # Reasoning models (e.g. gpt-oss) spend the token budget on internal
        # reasoning and return EMPTY content if it runs out -- that silently
        # poisoned 26% of the first pilot corpus. Low effort + headroom avoids
        # it; dropped automatically if the provider rejects the parameter.
        self._send_reasoning_effort = True

    def complete(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        import httpx

        # Loop long enough for the most patient branch; each branch below
        # enforces its own budget (HTTP errors give up sooner than network ones).
        for attempt in range(max(self.MAX_RETRIES, self.NETWORK_RETRIES)):
            body: dict = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
            if self._send_reasoning_effort:
                body["reasoning_effort"] = "low"

            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                    timeout=60.0,
                )
            except httpx.TransportError as exc:
                # Connection reset / DNS blip / read timeout / transient TLS
                # interception. A multi-hour corpus run WILL hit these, and an
                # outage can outlast a short retry window (an SSL verify failure
                # killed one run after ~60s of retries). Progress is checkpointed,
                # so patience is cheap and dying is expensive.
                #
                # NOT fixed by disabling certificate verification: if something
                # on the network is intercepting TLS, unverified requests would
                # hand it the API key.
                if attempt == self.NETWORK_RETRIES - 1:
                    raise RuntimeError(
                        f"network failure after {self.NETWORK_RETRIES} attempts spanning "
                        f"~{self.NETWORK_BACKOFF_CAP * self.NETWORK_RETRIES // 60} min: {exc}"
                    ) from exc
                time.sleep(min(2 ** attempt, self.NETWORK_BACKOFF_CAP) + 0.25)
                continue
            if resp.status_code == 400 and self._send_reasoning_effort:
                self._send_reasoning_effort = False  # provider doesn't support it
                continue
            if resp.status_code == 404:
                raise RuntimeError(
                    f"{self.base_url} has no model {self.model!r} (404). Provider model "
                    f"catalogues change over time -- list what this account actually serves "
                    f"with GET {self.base_url}/models, then set WARDEN_LLM_MODEL."
                )
            # free tiers rate-limit aggressively; honour Retry-After when given
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.MAX_RETRIES - 1:
                    resp.raise_for_status()
                wait = float(resp.headers.get("retry-after", 0) or 0) or min(2 ** attempt, 30)
                time.sleep(wait + 0.25)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        raise RuntimeError("unreachable")


class AnthropicClient:
    def __init__(self, api_key: str, model: str = MODEL):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, prompt: str, max_tokens: int = 150) -> str:
        resp = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


def resolve_model_id(provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """The model string that will be used, without building a client or
    requiring a key -- so a fully cached corpus replays offline."""
    load_dotenv()
    provider = provider or os.environ.get("WARDEN_LLM_PROVIDER", PROVIDER_ANTHROPIC)
    if model:
        return model
    if provider == PROVIDER_ANTHROPIC:
        return os.environ.get("WARDEN_LLM_MODEL", MODEL)
    if provider in KNOWN_PROVIDERS:
        return os.environ.get("WARDEN_LLM_MODEL", KNOWN_PROVIDERS[provider][1])
    return "unknown"


def resolve_client(provider: Optional[str] = None, model: Optional[str] = None):
    """Build a client from .env / environment. `provider` is 'anthropic' or one
    of KNOWN_PROVIDERS; defaults to WARDEN_LLM_PROVIDER, else 'anthropic'."""
    load_dotenv()
    provider = provider or os.environ.get("WARDEN_LLM_PROVIDER", PROVIDER_ANTHROPIC)

    if provider == PROVIDER_ANTHROPIC:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Either set it, or pick a free provider "
                f"via WARDEN_LLM_PROVIDER ({', '.join(KNOWN_PROVIDERS)}) plus that "
                "provider's key. Cached entries are reused without any key."
            )
        return AnthropicClient(key, model or MODEL)

    if provider not in KNOWN_PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected 'anthropic' or one of "
                         f"{list(KNOWN_PROVIDERS)}")

    base_url, default_model, key_var = KNOWN_PROVIDERS[provider]
    base_url = os.environ.get("WARDEN_LLM_BASE_URL", base_url)
    model = model or os.environ.get("WARDEN_LLM_MODEL", default_model)
    key = os.environ.get(key_var) or os.environ.get("WARDEN_LLM_API_KEY")
    if not key and provider != "ollama":
        raise RuntimeError(f"{key_var} is not set (needed for provider {provider!r}).")
    return OpenAICompatClient(base_url, key or "not-needed", model)


