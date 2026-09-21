"""M6: the three configs must be able to run against either M5 corpus, and
must never silently substitute one for the other.

Publishing template numbers under an LLM label is the one failure mode that
would invalidate the comparison without being visible in any output file,
so it is tested explicitly (DECISIONS.md D19).
"""
import json

import pytest

from warden.harness import COSINE_ONLY, NO_GATING, WARDEN_CONFIG
from warden.llm_workload import LLMRenderer, _Cache
from warden.results_gen import (
    CORPUS_LLM,
    CORPUS_TEMPLATE,
    ORACLE_STUB,
    build_renderer,
    corpus_fingerprint,
    run_all_configs,
)
from warden.workload import WorkloadConfig, generate

from tests.test_m5_llm_workload import FakeClient


def test_template_corpus_runs_all_three_configs():
    # pins the stub oracle: this covers corpus wiring, not adjudication, and
    # must not require network credentials to run
    events, dfs, elapsed = run_all_configs(seed=7, n_writes=120, num_keys=12,
                                           corpus=CORPUS_TEMPLATE, oracle=ORACLE_STUB)
    assert set(dfs) == {NO_GATING, COSINE_ONLY, WARDEN_CONFIG}
    for df in dfs.values():
        assert len(df) == len(events)
    assert all(elapsed[c] > 0 for c in dfs)


def test_unknown_corpus_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown corpus"):
        build_renderer("llm-ish")
    with pytest.raises(ValueError, match="unknown corpus"):
        run_all_configs(seed=1, n_writes=10, num_keys=4, corpus="")


def test_llm_corpus_without_key_or_cache_fails_loudly_not_silently(tmp_path):
    """The critical guard: a missing key must raise, never quietly produce
    a template corpus that would then be published as an LLM result.
    (Credentials are stripped suite-wide by conftest.no_ambient_credentials.)"""
    renderer = LLMRenderer(cache=_Cache(tmp_path / "empty.json"))
    with pytest.raises(RuntimeError, match="API_KEY"):
        generate(WorkloadConfig(n_writes=10, num_keys=3, seed=1), renderer=renderer)


def test_llm_corpus_differs_from_template_corpus(tmp_path):
    config = WorkloadConfig(n_writes=60, num_keys=8, seed=4)
    renderer = LLMRenderer(client=FakeClient(), cache=_Cache(tmp_path / "c.json"))

    template_events = generate(config)
    llm_events = generate(config, renderer=renderer)

    assert corpus_fingerprint(template_events) != corpus_fingerprint(llm_events)
    # but ground truth is untouched by the swap
    assert [(e.key, e.true_class, e.is_false) for e in template_events] == \
           [(e.key, e.true_class, e.is_false) for e in llm_events]


def test_fingerprint_is_stable_and_text_sensitive():
    config = WorkloadConfig(n_writes=50, num_keys=6, seed=9)
    a = corpus_fingerprint(generate(config))
    b = corpus_fingerprint(generate(config))
    c = corpus_fingerprint(generate(WorkloadConfig(n_writes=50, num_keys=6, seed=10)))
    assert a == b
    assert a != c


def test_provenance_file_records_corpus(tmp_path, monkeypatch):
    import warden.generate_results as gr

    monkeypatch.setattr(gr, "RESULTS_DIR", tmp_path)
    events = generate(WorkloadConfig(n_writes=40, num_keys=5, seed=2))
    record = gr.write_provenance(events, CORPUS_TEMPLATE, seed=2, n_writes=40, num_keys=5)

    written = json.loads((tmp_path / "corpus_provenance.json").read_text(encoding="utf-8"))
    assert written == record
    assert written["corpus"] == CORPUS_TEMPLATE
    assert written["corpus_fingerprint"] == corpus_fingerprint(events)
    assert written["unique_texts"] <= len(events)
