from warden.gate import WardenGate
from warden.store import Store
from warden.classifier import ClassificationResult


class FixedClassifier:
    """Test double: always returns the same predicted class, so Gate
    action semantics can be tested in isolation from real classification."""

    def __init__(self, predicted_class: str, escalated: bool = False):
        self.predicted_class = predicted_class
        self.escalated = escalated

    def classify(self, stored_text, incoming_text):
        return ClassificationResult(self.predicted_class, classify_ms=1.0, escalated=self.escalated)


def bootstrap(store: Store, key: str, text: str):
    store.write(key, text)


def test_duplicate_leaves_store_unchanged_version_unchanged():
    store = Store()
    bootstrap(store, "k", "v1")
    version_before = store.read("k").version

    gate = WardenGate(FixedClassifier("duplicate"))
    result = gate.process(store, "k", "v1", "v1 restated")

    assert result.action == "drop"
    assert store.read("k").value == "v1"
    assert store.read("k").version == version_before


def test_refinement_merges_and_increments_version():
    store = Store()
    bootstrap(store, "k", "v1")

    gate = WardenGate(FixedClassifier("refinement"))
    result = gate.process(store, "k", "v1", "v1 plus detail")

    assert result.action == "merge"
    merged = store.read("k").value
    assert "v1" in merged
    assert "v1 plus detail" in merged
    assert store.read("k").version == 2


def test_update_replaces_value_and_history_retains_the_old_one():
    store = Store()
    bootstrap(store, "k", "v1")

    gate = WardenGate(FixedClassifier("update"))
    result = gate.process(store, "k", "v1", "v2")

    assert result.action == "apply"
    assert store.read("k").value == "v2"
    assert store.read("k").version == 2
    history_values = [r.value for r in store.history("k")]
    assert history_values == ["v1", "v2"]  # old one retained, not destroyed


def test_contradiction_leaves_store_unchanged_and_is_recorded_as_blocked():
    store = Store()
    bootstrap(store, "k", "v1")
    version_before = store.read("k").version

    gate = WardenGate(FixedClassifier("contradiction"))
    result = gate.process(store, "k", "v1", "totally different value")

    assert result.action == "blocked"
    assert store.read("k").value == "v1"
    assert store.read("k").version == version_before


def test_every_action_is_logged_whatever_the_outcome():
    store = Store()
    bootstrap(store, "k", "v1")
    for cls in ("duplicate", "refinement", "update", "contradiction"):
        gate = WardenGate(FixedClassifier(cls))
        result = gate.process(store, "k", store.read("k").value, "incoming")
        # a GateResult is always produced, regardless of action taken
        assert result.action in ("drop", "merge", "apply", "blocked")
        assert result.predicted_class == cls
        assert result.classify_ms is not None
