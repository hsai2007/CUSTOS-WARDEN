import pytest

from warden.classifier import FastPathEscalator, CosineOnlyClassifier, StubOracle
from warden.workload import WorkloadConfig, generate, BOOTSTRAP, CLASSES


def sample_pairs(n=300, seed=11):
    events = generate(WorkloadConfig(n_writes=n, num_keys=25, seed=seed))
    return [(e.stored_text, e.incoming_text, e.true_class) for e in events if e.true_class != BOOTSTRAP]


def test_classifies_into_exactly_the_four_classes_never_anything_else():
    clf = FastPathEscalator()
    for stored, incoming, _ in sample_pairs():
        result = clf.classify(stored, incoming)
        assert result.predicted_class in CLASSES


def test_cosine_only_also_never_returns_anything_else():
    clf = CosineOnlyClassifier()
    for stored, incoming, _ in sample_pairs(n=30):
        result = clf.classify(stored, incoming)
        assert result.predicted_class in CLASSES


class CountingOracle(StubOracle):
    def __init__(self):
        self.calls = 0

    def decide(self, stored_text, incoming_text):
        self.calls += 1
        return "update"


def test_fast_path_never_calls_the_oracle_for_clear_duplicates_and_clear_mismatches():
    oracle = CountingOracle()
    clf = FastPathEscalator(oracle=oracle)
    fast_path_count = 0
    for stored, incoming, true_class in sample_pairs():
        result = clf.classify(stored, incoming)
        if not result.escalated:
            fast_path_count += 1
    assert fast_path_count > 0
    # every escalation increments oracle.calls exactly once; no calls on fast path
    assert oracle.calls == clf._escalated


def test_escalation_only_fires_on_the_ambiguous_band_and_rate_is_reported():
    clf = FastPathEscalator()
    escalated_true_classes = set()
    for stored, incoming, true_class in sample_pairs(n=500):
        result = clf.classify(stored, incoming)
        if result.escalated:
            escalated_true_classes.add(true_class)

    # only ever reached via (genuine or false) 'update' writes in this corpus
    assert escalated_true_classes <= {"update"}
    assert 0.0 <= clf.escalation_rate <= 1.0
    assert clf.escalation_rate > 0, "expected the ambiguous band to be exercised"


def test_identical_input_gives_identical_output_deterministic_on_fast_path():
    clf = FastPathEscalator()
    stored, incoming, _ = sample_pairs()[0]
    r1 = clf.classify(stored, incoming)
    r2 = clf.classify(stored, incoming)
    assert r1.predicted_class == r2.predicted_class
    assert r1.escalated == r2.escalated


def test_classify_ms_recorded_on_every_call_including_escalated():
    clf = FastPathEscalator()
    saw_escalated = False
    for stored, incoming, _ in sample_pairs(n=400):
        result = clf.classify(stored, incoming)
        assert result.classify_ms >= 0.0
        assert isinstance(result.classify_ms, float)
        if result.escalated:
            saw_escalated = True
    assert saw_escalated


def test_rule5_cosine_alone_misclassifies_and_fact_gate_corrects_it():
    """The M2 rule-5 pair reused here: near-identical wording, genuinely
    different resolved date. Cosine-only calls it a duplicate; the fact
    gate correctly refuses to."""
    stored_text = "The release ships on Friday."
    incoming_text = "The release ships on next Friday."

    cosine_clf = CosineOnlyClassifier()
    fact_clf = FastPathEscalator()

    cosine_result = cosine_clf.classify(stored_text, incoming_text)
    fact_result = fact_clf.classify(stored_text, incoming_text)

    assert cosine_result.predicted_class == "duplicate", "expected cosine-only to be fooled by high similarity"
    assert fact_result.predicted_class != "duplicate", "fact gate must not be fooled the same way"
