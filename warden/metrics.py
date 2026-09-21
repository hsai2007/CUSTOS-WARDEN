"""M6: metrics computed purely from the frozen write-log schema.

Every number in results/ derives from these functions applied to the
`writes.csv` log (or an in-memory list-of-dicts/DataFrame with the same
columns) -- never from re-simulating anything.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from warden.workload import BOOTSTRAP, CLASSES


ACTIONS_THAT_WRITE = frozenset({"apply", "merge"})


def _non_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["true_class"] != BOOTSTRAP]


def false_facts_surviving(df: pd.DataFrame) -> float:
    """Fraction of false-content writes that actually entered the store.
    Only 'apply' and 'merge' write anything; 'drop' (duplicate) and
    'blocked' (contradiction) both leave the stored value untouched, so
    neither counts as the false content 'surviving'."""
    false_rows = df[df["is_false"] == True]  # noqa: E712
    if len(false_rows) == 0:
        return 0.0
    return float(false_rows["action"].isin(ACTIONS_THAT_WRITE).mean())


def honest_wrongly_blocked(df: pd.DataFrame) -> float:
    """Fraction of honest (non-false) writes that were incorrectly blocked."""
    honest_rows = df[df["is_false"] == False]  # noqa: E712
    if len(honest_rows) == 0:
        return 0.0
    return float((honest_rows["action"] == "blocked").mean())


def false_via_correct_classification(df: pd.DataFrame) -> Optional[float]:
    """Fraction of false writes that survived specifically THROUGH a
    correctly classified write (predicted_class == true_class) rather than
    through misclassification. None for configs that never classify
    (predicted_class is always 'n/a')."""
    classified = df[df["predicted_class"] != "n/a"]
    false_rows = classified[classified["is_false"] == True]  # noqa: E712
    if len(false_rows) == 0:
        return None
    correctly_classified_and_survived = false_rows[
        (false_rows["predicted_class"] == false_rows["true_class"])
        & (false_rows["action"].isin(ACTIONS_THAT_WRITE))
    ]
    return float(len(correctly_classified_and_survived) / len(false_rows))


def require_paired_reporting(metrics: dict) -> None:
    """Reporting guard: false_facts_surviving must never be reported
    without honest_wrongly_blocked alongside it (a safety number without
    its cost counterpart is misleading)."""
    has_false = "false_facts_surviving" in metrics
    has_honest = "honest_wrongly_blocked" in metrics
    if has_false != has_honest:
        raise ValueError(
            "false_facts_surviving must always be reported alongside "
            "honest_wrongly_blocked; refusing to report one without the other."
        )


def four_class_accuracy(df: pd.DataFrame) -> Optional[float]:
    classified = _non_bootstrap(df)
    classified = classified[classified["predicted_class"] != "n/a"]
    if len(classified) == 0:
        return None
    return float((classified["predicted_class"] == classified["true_class"]).mean())


def per_class_prf1(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    classified = _non_bootstrap(df)
    classified = classified[classified["predicted_class"] != "n/a"]
    out: dict[str, dict[str, float]] = {}
    if len(classified) == 0:
        return {c: {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")} for c in CLASSES}

    for cls in CLASSES:
        tp = int(((classified["predicted_class"] == cls) & (classified["true_class"] == cls)).sum())
        fp = int(((classified["predicted_class"] == cls) & (classified["true_class"] != cls)).sum())
        fn = int(((classified["predicted_class"] != cls) & (classified["true_class"] == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[cls] = {"precision": precision, "recall": recall, "f1": f1}
    return out


def confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    classified = _non_bootstrap(df)
    classified = classified[classified["predicted_class"] != "n/a"]
    matrix = pd.DataFrame(0, index=list(CLASSES), columns=list(CLASSES))
    for _, row in classified.iterrows():
        if row["true_class"] in CLASSES and row["predicted_class"] in CLASSES:
            matrix.loc[row["true_class"], row["predicted_class"]] += 1
    return matrix


def classify_cost_percentiles(df: pd.DataFrame) -> tuple[float, float]:
    classified = df[df["predicted_class"] != "n/a"]
    if len(classified) == 0:
        return (0.0, 0.0)
    p50 = float(np.percentile(classified["classify_ms"], 50))
    p95 = float(np.percentile(classified["classify_ms"], 95))
    return p50, p95


def classify_cost_by_path(df: pd.DataFrame) -> dict[str, Optional[float]]:
    """Cost split by tier instead of blended.

    WARDEN is deliberately two-tier: a cheap fact gate handling most writes
    and an expensive LLM adjudicator handling the ambiguous band. A single
    blended percentile describes neither tier and hides the architecture --
    it is a mix of ~0.1ms and ~700ms that corresponds to no real operation.
    Reporting both separately shows whether the cheap tier is actually
    carrying the traffic it was designed to carry (DECISIONS.md D23)."""
    classified = df[df["predicted_class"] != "n/a"]
    if len(classified) == 0:
        return {k: None for k in ("fast_p50", "fast_p95", "esc_p50", "esc_p95")}

    fast = classified[~classified["escalated"].astype(bool)]["classify_ms"]
    esc = classified[classified["escalated"].astype(bool)]["classify_ms"]
    pct = lambda s, q: float(np.percentile(s, q)) if len(s) else None  # noqa: E731
    return {
        "fast_p50": pct(fast, 50), "fast_p95": pct(fast, 95),
        "esc_p50": pct(esc, 50), "esc_p95": pct(esc, 95),
    }


def escalation_rate(df: pd.DataFrame) -> Optional[float]:
    classified = df[df["predicted_class"] != "n/a"]
    if len(classified) == 0:
        return None
    return float(classified["escalated"].mean())


def throughput_writes_per_sec(elapsed_seconds: float, n_writes: int) -> float:
    return n_writes / elapsed_seconds if elapsed_seconds > 0 else float("inf")
