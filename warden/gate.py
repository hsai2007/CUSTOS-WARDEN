"""M4: gate. Applies the class-conditional store action for whatever a
classifier decided, and always returns a result to log -- regardless of
outcome, every action is logged (see warden.harness.run_workload)."""
from __future__ import annotations

from warden.harness import GateResult
from warden.store import Store


class WardenGate:
    """Wraps any classifier (warden.classifier.FastPathEscalator or
    CosineOnlyClassifier) with the M4 action semantics:

        duplicate     -> drop   (store unchanged, version unchanged)
        refinement    -> merge  (merged value written, version increments)
        update        -> apply  (stored value replaced, history keeps the old one)
        contradiction -> blocked (store unchanged, write recorded as blocked)
    """

    def __init__(self, classifier):
        self.classifier = classifier

    def process(self, store: Store, key: str, stored_text, incoming_text) -> GateResult:
        result = self.classifier.classify(stored_text, incoming_text)
        cls = result.predicted_class

        if cls == "duplicate":
            action = "drop"
        elif cls == "refinement":
            store.write(key, self._merge(stored_text, incoming_text))
            action = "merge"
        elif cls == "update":
            store.write(key, incoming_text)
            action = "apply"
        else:  # contradiction
            action = "blocked"

        return GateResult(
            predicted_class=cls,
            action=action,
            classify_ms=result.classify_ms,
            escalated=result.escalated,
        )

    @staticmethod
    def _merge(stored_text: str, incoming_text: str) -> str:
        return f"{stored_text} | {incoming_text}"
