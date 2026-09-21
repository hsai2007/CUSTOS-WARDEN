"""Suite-wide isolation from real LLM credentials.

Once `.env` holds a working key, any test that resolves a client will happily
make a live, billable network call and pass or fail depending on whether the
developer has credentials on disk. That makes the suite non-hermetic and slow,
and it silently changed the meaning of two "fails loudly without a key" tests
(they stopped testing anything). See DECISIONS.md D22.

Tests that genuinely need a real key opt back in via the `live_llm` fixture.
"""
import pytest

PROVIDER_ENV_VARS = (
    "WARDEN_LLM_PROVIDER", "WARDEN_LLM_MODEL", "WARDEN_LLM_BASE_URL",
    "WARDEN_LLM_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY",
    "GEMINI_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY",
)


@pytest.fixture(autouse=True)
def no_ambient_credentials(monkeypatch, request):
    """Neutralize .env and provider keys for every test by default."""
    if "live_llm" in request.fixturenames:
        return
    monkeypatch.setattr("warden.llm_client.load_dotenv", lambda *a, **k: None)
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def live_llm():
    """Opt back in to real credentials. Skips unless one is actually present."""
    import os
    from warden.llm_client import load_dotenv
    load_dotenv()
    if not any(os.environ.get(v) for v in PROVIDER_ENV_VARS if v.endswith("API_KEY")):
        pytest.skip("no live LLM credentials available")
