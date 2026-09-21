"""The stable API surface M7's LangGraph agents write through. It's a thin
wrapper over the same Store + Gate pair the M6 harness uses internally, so
an agent's write is subject to exactly the same WARDEN action semantics as
any other write in this project -- submit() is the entire contract."""
from __future__ import annotations

from dataclasses import dataclass

from warden.classifier import FastPathEscalator
from warden.gate import WardenGate
from warden.store import Store


@dataclass
class SubmitResult:
    key: str
    incoming_text: str
    predicted_class: str
    action: str
    escalated: bool


class WardenGateway:
    def __init__(self, classifier=None):
        self.store = Store()
        self.gate = WardenGate(classifier or FastPathEscalator())

    def submit(self, key: str, text: str) -> SubmitResult:
        existing = self.store.read(key)
        if existing is None:
            self.store.write(key, text)
            return SubmitResult(key, text, "bootstrap", "apply", False)

        result = self.gate.process(self.store, key, existing.value, text)
        return SubmitResult(key, text, result.predicted_class, result.action, result.escalated)
