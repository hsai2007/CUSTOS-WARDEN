"""M6 evaluation harness tests.

Per the build order's explicit exception, M6 is built and its no-gating
baseline is run BEFORE M3 (the classifier) exists. The schema, reporting
guard, and no-gating baseline/checkpoint tests were green from that point.
Now that M3 (classifier) and M4 (gate) exist, this file also covers
running all three configs over the same seeded workload. The "all nine
output files" test lives separately (test_m6_output_files.py) since item
7 (scene_agents) is M7-only and can't exist until M7 is built.
"""
import pandas as pd
import pytest

from warden.classifier import FastPathEscalator, CosineOnlyClassifier
from warden.gate import WardenGate
from warden.harness import LOG_COLUMNS, run_no_gating, run_workload, NO_GATING, COSINE_ONLY, WARDEN_CONFIG
from warden.metrics import (
    false_facts_surviving,
    honest_wrongly_blocked,
    require_paired_reporting,
)
from warden.workload import WorkloadConfig, generate


FROZEN_SCHEMA = (
    "ts", "config", "write_id", "key", "true_class", "predicted_class",
    "action", "is_false", "version_read", "version_at_commit",
    "classify_ms", "escalated",
)


def test_schema_names_all_12_columns_explicitly_in_frozen_order():
    """A failure here is a DECISIONS.md decision, not a test to edit."""
    assert LOG_COLUMNS == FROZEN_SCHEMA
    assert len(LOG_COLUMNS) == 12


def test_no_gating_produces_all_12_columns_in_frozen_order():
    events = generate(WorkloadConfig(n_writes=50, num_keys=10, seed=1))
    rows, _ = run_no_gating(events)
    df = pd.DataFrame(rows)
    assert tuple(df.columns) == FROZEN_SCHEMA


def test_reporting_guard_rejects_false_facts_without_honest_blocked():
    with pytest.raises(ValueError):
        require_paired_reporting({"false_facts_surviving": 0.5})
    with pytest.raises(ValueError):
        require_paired_reporting({"honest_wrongly_blocked": 0.1})
    # both present, or neither present, is fine
    require_paired_reporting({"false_facts_surviving": 0.5, "honest_wrongly_blocked": 0.1})
    require_paired_reporting({"four_class_accuracy": 0.9})


def test_no_gating_baseline_checkpoint_false_facts_surviving_is_meaningfully_above_zero():
    """CRITICAL CHECKPOINT (per project spec): if false facts surviving
    under no gating is not meaningfully above zero, the whole premise of
    this project is void and there is nothing for M3/M4 to improve on."""
    events = generate(WorkloadConfig(n_writes=2000, num_keys=50, seed=7))
    rows, store = run_no_gating(events)
    df = pd.DataFrame(rows)

    ffs = false_facts_surviving(df)
    hwb = honest_wrongly_blocked(df)
    require_paired_reporting({"false_facts_surviving": ffs, "honest_wrongly_blocked": hwb})

    # no gating never blocks anything, so wrongly-blocked honest writes is trivially 0
    assert hwb == 0.0
    # and false writes always survive (nothing ever blocks) - must be way above 0
    assert ffs > 0.20, f"false facts surviving = {ffs}; expected meaningfully above zero"

    n_false = int(df["is_false"].sum())
    assert n_false > 0


def test_runs_all_three_configs_over_the_same_seeded_workload():
    events = generate(WorkloadConfig(n_writes=300, num_keys=20, seed=99))

    no_gating_rows, _ = run_workload(events, NO_GATING, gate=None)
    cosine_rows, _ = run_workload(events, COSINE_ONLY, gate=WardenGate(CosineOnlyClassifier()))
    warden_rows, _ = run_workload(events, WARDEN_CONFIG, gate=WardenGate(FastPathEscalator()))

    for rows, expected_config in (
        (no_gating_rows, NO_GATING), (cosine_rows, COSINE_ONLY), (warden_rows, WARDEN_CONFIG),
    ):
        df = pd.DataFrame(rows)
        assert tuple(df.columns) == LOG_COLUMNS
        assert len(df) == len(events)
        assert set(df["config"]) == {expected_config}
        # same input sequence: write_id/key/true_class must line up identically across configs
        assert list(df["write_id"]) == [e.write_id for e in events]
        assert list(df["key"]) == [e.key for e in events]
        assert list(df["true_class"]) == [e.true_class for e in events]
