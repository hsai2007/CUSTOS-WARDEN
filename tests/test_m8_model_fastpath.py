"""M8 trained fast path. Unit tests inject logits, so they need no model,
no torch and no network. One integration test runs only if the trained
model is present in models/slow_stage_v3b."""
import importlib.util

import pytest

from warden.model_classifier import DEFAULT_MODEL_DIR, ModelFastPath
from warden.workload import CLASSES


class FakeModel(ModelFastPath):
    def __init__(self, logits, **kw):
        super().__init__(**kw)
        self.fixed = logits

    def _logits(self, stored_text, incoming_text):
        return dict(self.fixed)


class CountingOracle:
    def __init__(self, answer="update"):
        self.calls, self.answer = 0, answer

    def decide(self, stored_text, incoming_text):
        self.calls += 1
        return self.answer


def L(dup=0.0, ref=0.0, upd=0.0, con=0.0):
    return {"duplicate": dup, "refinement": ref, "update": upd, "contradiction": con}


def test_clear_update_stays_update():
    assert FakeModel(L(upd=6, con=1)).classify("a", "b").predicted_class == "update"


def test_doubtful_update_is_blocked_without_an_oracle():
    r = FakeModel(L(upd=3, con=1)).classify("a", "b")          # gap 2 < margin 3
    assert r.predicted_class == "contradiction" and not r.escalated


def test_doubtful_update_goes_to_the_llm_when_an_oracle_is_given():
    oracle = CountingOracle("update")
    clf = FakeModel(L(upd=3, con=1), oracle=oracle)
    r = clf.classify("a", "b")
    assert r.predicted_class == "update" and r.escalated and oracle.calls == 1
    assert clf.escalation_rate == 1.0


def test_margin_never_touches_other_classes():
    oracle = CountingOracle()
    for logits, expected in [(L(dup=5, upd=4.9), "duplicate"), (L(ref=5, con=4.9), "refinement"),
                             (L(con=5, upd=4.9), "contradiction")]:
        assert FakeModel(logits, oracle=oracle).classify("a", "b").predicted_class == expected
    assert oracle.calls == 0


def test_margin_zero_is_plain_argmax():
    assert FakeModel(L(upd=1.0, con=0.9), margin=0).classify("a", "b").predicted_class == "update"


def test_always_returns_one_of_the_four_classes():
    for logits in [L(dup=1), L(ref=1), L(upd=9), L(upd=1, con=0.5), L(con=1)]:
        assert FakeModel(logits).classify("a", "b").predicted_class in CLASSES


def test_missing_model_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelFastPath(model_dir=tmp_path).classify("a", "b")


@pytest.mark.slow
@pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "config.json").exists()
                    or importlib.util.find_spec("transformers") is None,
                    reason="trained model or transformers not installed")
def test_real_model_on_obvious_pairs():
    clf = ModelFastPath()
    assert clf.classify("The launch is on Friday.", "The launch is on Friday.").predicted_class == "duplicate"
    r = clf.classify("The infra budget is 250.", "The infra budget is 250, covering cloud and on-prem costs.")
    assert r.predicted_class == "refinement"
