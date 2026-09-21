"""M5: alternate LLM providers for workload generation.

Generation is a low-difficulty instruction-following task, so a free tier is
fine -- and a non-Claude generator is preferable, since it removes the mild
circularity of one model family both writing the corpus and adjudicating M3
escalations (DECISIONS.md D20).
"""
import pytest

from warden.llm_client import (
    KNOWN_PROVIDERS,
    OpenAICompatClient,
    _Cache,
    resolve_client,
    resolve_model_id,
)
from warden.llm_workload import LLMRenderer
from warden.renderers import RenderRequest

from tests.test_m5_llm_workload import FakeClient


REQ = RenderRequest(write_id=0, domain="ship_date", subject="the launch",
                    true_class="duplicate", is_false=False,
                    stored_text="It ships Friday.", stored_attrs={"day": "friday"},
                    style="terse")


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """Neutralize .env for every test here. Otherwise a developer who has a
    real key on disk gets different results than CI: load_dotenv() would
    re-populate the very variable a test just deleted."""
    monkeypatch.setattr("warden.llm_client.load_dotenv", lambda *a, **k: None)
    for var in ("WARDEN_LLM_PROVIDER", "WARDEN_LLM_MODEL", "WARDEN_LLM_BASE_URL",
                "WARDEN_LLM_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.delenv("WARDEN_LLM_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="unknown provider"):
        resolve_client(provider="definitely-not-a-provider")


def test_missing_provider_key_raises_naming_the_right_variable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("WARDEN_LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        resolve_client(provider="groq")


def test_provider_key_is_picked_up(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-value")
    client = resolve_client(provider="groq")
    assert isinstance(client, OpenAICompatClient)
    assert client.base_url == KNOWN_PROVIDERS["groq"][0]
    assert client.model == KNOWN_PROVIDERS["groq"][1]


def test_model_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("WARDEN_LLM_MODEL", "some-other-model")
    assert resolve_client(provider="groq").model == "some-other-model"
    assert resolve_model_id("groq") == "some-other-model"


def test_cache_is_namespaced_by_model(tmp_path):
    """Switching model must not silently replay another model's sentences."""
    import random
    cache = _Cache(tmp_path / "c.json")
    rng = random.Random(0)

    a = LLMRenderer(client=FakeClient(), cache=cache)
    b = LLMRenderer(client=FakeClient(), cache=cache)
    # distinct model ids => distinct cache namespaces
    a._client.model = "model-a"
    b._client.model = "model-b"

    text_a = a.render(REQ, rng)
    text_b = b.render(REQ, rng)
    assert a.calls_made == 1 and b.calls_made == 1  # b did NOT hit a's entry
    assert f"model-a::{REQ.cache_key()}" != f"model-b::{REQ.cache_key()}"
    assert text_a and text_b


def test_fully_cached_corpus_replays_with_no_key(tmp_path, monkeypatch):
    """model_id must not build a client -- otherwise a cached corpus could not
    be replayed offline, which is the whole point of caching."""
    import random
    cache = _Cache(tmp_path / "c.json")
    warm = LLMRenderer(client=FakeClient(), cache=cache)
    warm._client.model = "cached-model"
    expected = warm.render(REQ, random.Random(0))

    # now: no key anywhere, explicit model so the namespace matches
    for var in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "WARDEN_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cold = LLMRenderer(cache=cache, model="cached-model")
    assert cold.render(REQ, random.Random(0)) == expected
    assert cold.calls_made == 0


def test_openai_compat_client_posts_expected_shape(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        headers: dict = {}
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "  a sentence.  "}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json)
        return FakeResponse()

    import httpx
    monkeypatch.setattr(httpx, "post", fake_post)

    client = OpenAICompatClient("https://example.test/v1/", "secret", "m1")
    out = client.complete("sys", "user prompt")

    assert out.strip() == "a sentence."
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["model"] == "m1"
    assert [m["role"] for m in captured["json"]["messages"]] == ["system", "user"]


# --- corpus contamination guards (see DECISIONS.md D21) ----------------------

def test_validate_rejects_empty_refusal_and_overlong():
    from warden.llm_client import UnusableResponse, validate_text

    for bad in ("", "   ", "\n\n",
                "I'm sorry, but I can't help with that.",
                "I’m sorry, but I can’t help with that.",  # smart quotes
                "As an AI, I cannot assist with that request.",
                "x" * 400):
        with pytest.raises(UnusableResponse):
            validate_text(bad)


def test_validate_normalizes_but_preserves_good_text():
    from warden.llm_client import validate_text
    assert validate_text('  The launch ships on Friday.  ') == "The launch ships on Friday."
    assert validate_text('"Quoted claim."') == "Quoted claim."
    assert validate_text("Line one\nstill one sentence.") == "Line one still one sentence."


def test_unusable_output_is_never_cached(tmp_path):
    """The failure that poisoned 26% of the first pilot: an empty response
    cached as if it were a real sentence."""
    import random
    from warden.llm_client import _Cache, UnusableResponse
    from warden.llm_workload import LLMRenderer

    class EmptyClient:
        model = "always-empty"
        def complete(self, system, prompt, max_tokens=512):
            return ""

    cache = _Cache(tmp_path / "c.json")
    renderer = LLMRenderer(client=EmptyClient(), cache=cache)

    with pytest.raises(UnusableResponse):
        renderer.render(REQ, random.Random(0))
    assert len(cache) == 0  # nothing poisoned


def test_transient_bad_output_is_retried_then_accepted(tmp_path):
    import random
    from warden.llm_client import _Cache
    from warden.llm_workload import LLMRenderer

    class FlakyClient:
        model = "flaky"
        def __init__(self): self.n = 0
        def complete(self, system, prompt, max_tokens=512):
            self.n += 1
            return "" if self.n == 1 else "A real sentence."

    renderer = LLMRenderer(client=FlakyClient(), cache=_Cache(tmp_path / "c.json"))
    assert renderer.render(REQ, random.Random(0)) == "A real sentence."
