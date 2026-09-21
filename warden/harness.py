"""M6: evaluation harness.

Runs a generated M5 workload through a given config (no gating / cosine
only / WARDEN) against a fresh M1 Store, producing one frozen-schema log
row per write. `gate=None` means the "no gating" baseline: every write is
blindly applied, exactly the naive overwrite semantics the M1 rule-5 proof
warns about, just running end-to-end over the whole workload instead of
one key.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol

from warden.store import Store
from warden.workload import WriteEvent, BOOTSTRAP

LOG_COLUMNS = (
    "ts", "config", "write_id", "key", "true_class", "predicted_class",
    "action", "is_false", "version_read", "version_at_commit",
    "classify_ms", "escalated",
)

NO_GATING = "no_gating"
COSINE_ONLY = "cosine_only"
WARDEN_CONFIG = "warden"
ALL_CONFIGS = (NO_GATING, COSINE_ONLY, WARDEN_CONFIG)


@dataclass
class GateResult:
    predicted_class: str
    action: str
    classify_ms: float
    escalated: bool


class Gate(Protocol):
    def process(self, store: Store, key: str, stored_text: Optional[str], incoming_text: str) -> GateResult:
        ...


def _version_of(store: Store, key: str) -> int:
    rec = store.read(key)
    return rec.version if rec is not None else 0


def run_workload(events: list[WriteEvent], config_name: str, gate: Optional[Gate] = None) -> tuple[list[dict], Store]:
    """Run `events` through `gate` (or blind-apply if gate is None) against
    a fresh Store. Returns (log_rows, final_store)."""
    store = Store()
    rows: list[dict] = []

    for e in events:
        version_read = _version_of(store, e.key)
        # Classify against the pair the LABEL was defined for, not against
        # whatever the store happens to hold.
        #
        # M5 assigns `true_class` relative to its own stored value. The gate,
        # however, BLOCKS some writes -- so when it wrongly blocks a genuine
        # update, the store keeps the old value while M5's ground truth moves
        # on. Every later label for that key is then defined against a value
        # the store never received, and the labels silently become fiction.
        # Worse, the effect is self-amplifying: the more the classifier errs,
        # the faster the labels rot, so error compounds rather than averaging
        # out. Measured cost of getting this wrong: 29.9% vs 64.2% accuracy
        # on the same classifier (DECISIONS.md D24).
        #
        # The store still records every action, version and history entry, and
        # the safety metrics still read from it -- only the classifier's INPUT
        # is pinned to the labelled pair.
        stored_text = e.stored_text if e.stored_text is not None else (
            store.read(e.key).value if version_read > 0 else None)

        if gate is None:
            store.write(e.key, e.incoming_text)
            predicted_class = "n/a"
            action = "apply"
            classify_ms = 0.0
            escalated = False
        elif e.true_class == BOOTSTRAP or version_read == 0:
            # bootstrapping a brand-new key is not a conflict-classification
            # case for any config: there is nothing yet to compare against.
            store.write(e.key, e.incoming_text)
            predicted_class = "bootstrap"
            action = "apply"
            classify_ms = 0.0
            escalated = False
        else:
            result = gate.process(store, e.key, stored_text, e.incoming_text)
            predicted_class = result.predicted_class
            action = result.action
            classify_ms = result.classify_ms
            escalated = result.escalated

        version_at_commit = _version_of(store, e.key)
        rows.append({
            "ts": e.ts,
            "config": config_name,
            "write_id": e.write_id,
            "key": e.key,
            "true_class": e.true_class,
            "predicted_class": predicted_class,
            "action": action,
            "is_false": e.is_false,
            "version_read": version_read,
            "version_at_commit": version_at_commit,
            "classify_ms": classify_ms,
            "escalated": escalated,
        })

    return rows, store


def run_no_gating(events: list[WriteEvent]) -> tuple[list[dict], Store]:
    return run_workload(events, NO_GATING, gate=None)


def timed_throughput(events: list[WriteEvent], config_name: str, gate: Optional[Gate] = None) -> float:
    """Writes per second for running the full workload through this config."""
    start = time.perf_counter()
    rows, _ = run_workload(events, config_name, gate=gate)
    elapsed = time.perf_counter() - start
    return len(rows) / elapsed if elapsed > 0 else float("inf")
