"""Builds the frozen metrics table (and everything results/ needs) purely
from the writes.csv log produced by running all three configs over one
seeded M5 workload. No metric here is derived from warden.agents (M7) --
see DECISIONS.md and tests/test_m7_agents.py::test_no_results_module_imports_agents.
"""
from __future__ import annotations

import hashlib
import time

import pandas as pd

from warden.classifier import CosineOnlyClassifier, FastPathEscalator, LLMOracle, StubOracle
from warden.gate import WardenGate
from warden.harness import COSINE_ONLY, NO_GATING, WARDEN_CONFIG, run_workload
from warden.metrics import (
    classify_cost_by_path,
    classify_cost_percentiles,
    confusion_matrix,
    escalation_rate,
    false_facts_surviving,
    false_via_correct_classification,
    four_class_accuracy,
    honest_wrongly_blocked,
    per_class_prf1,
    require_paired_reporting,
    throughput_writes_per_sec,
)
from warden.workload import WorkloadConfig, generate

CONFLICTRAG_ACCURACY_TARGET = 0.743
CONFLICTRAG_WORST_F1_REFERENCE = 0.685
WORST_F1_TARGET = 0.70

METRIC_ROWS = [
    "False facts surviving in store",
    "Honest writes wrongly blocked",
    "Four-class accuracy",
    "Worst per-class F1",
    "Duplicate F1",
    "Refinement F1",
    "Update F1",
    "Contradiction F1",
    "Classification cost p50",
    "Classification cost p95",
    "LLM escalation rate",
    "Throughput, writes/s",
    "False facts via correctly-classified write",
]


CORPUS_TEMPLATE = "template"
CORPUS_LLM = "llm"


def build_renderer(corpus: str):
    """Renderer for a named corpus. Raises on an unknown name rather than
    defaulting -- silently falling back to templates would publish template
    numbers under an LLM label, which is the one failure mode that would
    invalidate the comparison without being visible (DECISIONS.md D19)."""
    if corpus == CORPUS_TEMPLATE:
        return None  # generate() defaults to TemplateRenderer
    if corpus == CORPUS_LLM:
        from warden.llm_workload import LLMRenderer
        return LLMRenderer()
    raise ValueError(f"unknown corpus {corpus!r}; expected {CORPUS_TEMPLATE!r} or {CORPUS_LLM!r}")


def corpus_fingerprint(events) -> str:
    """Short hash of the generated text, so a results file can be tied back
    to the exact corpus that produced it."""
    h = hashlib.sha256()
    for e in events:
        h.update(e.incoming_text.encode("utf-8"))
    return h.hexdigest()[:16]


ORACLE_STUB = "stub"
ORACLE_LLM = "llm"


def build_oracle(oracle: str):
    """The M3 escalation adjudicator.

    'stub' is not a placeholder -- it is the ablation arm representing the
    policy "trust the supersession language" (always answer update). 'llm' is
    the real adjudicator the architecture intends. The gap between the two
    runs is what isolates the value of escalation (DECISIONS.md D23)."""
    if oracle == ORACLE_STUB:
        return StubOracle()
    if oracle == ORACLE_LLM:
        return LLMOracle(model=ADJUDICATOR_MODEL)
    raise ValueError(f"unknown oracle {oracle!r}; expected {ORACLE_STUB!r} or {ORACLE_LLM!r}")


# Deliberately a DIFFERENT model from the one generating the corpus, so the
# adjudicator never judges text written by its own model family (D20).
ADJUDICATOR_MODEL = "qwen/qwen3.8-27b"


def run_all_configs(seed: int = 2026, n_writes: int = 3000, num_keys: int = 60,
                    corpus: str = CORPUS_TEMPLATE, oracle: str = ORACLE_LLM):
    """Run the SAME seeded workload through all three configs. Returns
    (events, {config: df}, {config: elapsed_seconds})."""
    events = generate(
        WorkloadConfig(n_writes=n_writes, num_keys=num_keys, seed=seed),
        renderer=build_renderer(corpus),
    )

    gates = {
        NO_GATING: None,
        COSINE_ONLY: WardenGate(CosineOnlyClassifier()),
        WARDEN_CONFIG: WardenGate(FastPathEscalator(oracle=build_oracle(oracle))),
    }

    dfs: dict[str, pd.DataFrame] = {}
    elapsed: dict[str, float] = {}
    for config_name, gate in gates.items():
        start = time.perf_counter()
        rows, _store = run_workload(events, config_name, gate=gate)
        elapsed[config_name] = time.perf_counter() - start
        dfs[config_name] = pd.DataFrame(rows)

    return events, dfs, elapsed


def _fmt_pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _fmt_ms(x):
    return "n/a" if x is None else f"{x:.2f}"


def _fmt_num(x):
    return "n/a" if x is None else f"{x:.1f}"


def build_metrics_table(dfs: dict[str, pd.DataFrame], elapsed: dict[str, float]) -> pd.DataFrame:
    ffs = {c: false_facts_surviving(dfs[c]) for c in dfs}
    hwb = {c: honest_wrongly_blocked(dfs[c]) for c in dfs}
    require_paired_reporting({"false_facts_surviving": ffs, "honest_wrongly_blocked": hwb})

    acc = {c: four_class_accuracy(dfs[c]) for c in dfs}
    prf1 = {c: per_class_prf1(dfs[c]) for c in dfs}
    cost = {c: classify_cost_percentiles(dfs[c]) for c in dfs}
    esc = {c: escalation_rate(dfs[c]) for c in dfs}
    thr = {c: throughput_writes_per_sec(elapsed[c], len(dfs[c])) for c in dfs}
    false_via_correct = {c: false_via_correct_classification(dfs[c]) for c in dfs}

    def worst_f1(config):
        vals = prf1[config]
        if vals is None or any(v["f1"] != v["f1"] for v in vals.values()):  # NaN check
            return None
        return min(v["f1"] for v in vals.values())

    rows = []

    def add_row(metric, target, no_gating_v, cosine_v, warden_v, met_fn, is_report=False):
        status = "n/a (report only)" if is_report else ("MET" if met_fn(warden_v) else "MISSED")
        rows.append({
            "Metric": metric, "Target": target,
            "No gating": no_gating_v, "Cosine only": cosine_v, "WARDEN": warden_v,
            "Status": status,
        })

    add_row("False facts surviving in store", "< 5%",
            _fmt_pct(ffs[NO_GATING]), _fmt_pct(ffs[COSINE_ONLY]), _fmt_pct(ffs[WARDEN_CONFIG]),
            lambda v: ffs[WARDEN_CONFIG] < 0.05)

    add_row("Honest writes wrongly blocked", "< 10%",
            _fmt_pct(hwb[NO_GATING]), _fmt_pct(hwb[COSINE_ONLY]), _fmt_pct(hwb[WARDEN_CONFIG]),
            lambda v: hwb[WARDEN_CONFIG] < 0.10)

    add_row("Four-class accuracy", f"> {CONFLICTRAG_ACCURACY_TARGET*100:.1f}%",
            "n/a", _fmt_pct(acc[COSINE_ONLY]), _fmt_pct(acc[WARDEN_CONFIG]),
            lambda v: (acc[WARDEN_CONFIG] or 0) > CONFLICTRAG_ACCURACY_TARGET)

    wf1 = {c: worst_f1(c) for c in dfs}
    add_row("Worst per-class F1", f"> {WORST_F1_TARGET:.2f}",
            "n/a", _fmt_num(wf1[COSINE_ONLY]), _fmt_num(wf1[WARDEN_CONFIG]),
            lambda v: (wf1[WARDEN_CONFIG] or 0) > WORST_F1_TARGET)

    for cls in ("duplicate", "refinement", "update", "contradiction"):
        def make_getter(c=cls):
            return lambda config: prf1[config][c]["f1"] if prf1[config] else None
        getter = make_getter()
        add_row(f"{cls.capitalize()} F1", "report",
                "n/a", _fmt_num(getter(COSINE_ONLY)), _fmt_num(getter(WARDEN_CONFIG)),
                lambda v: True, is_report=True)

    add_row("Classification cost p50", "< 10 ms",
            "n/a", _fmt_ms(cost[COSINE_ONLY][0]), _fmt_ms(cost[WARDEN_CONFIG][0]),
            lambda v: cost[WARDEN_CONFIG][0] < 10)

    add_row("Classification cost p95", "< 50 ms",
            "n/a", _fmt_ms(cost[COSINE_ONLY][1]), _fmt_ms(cost[WARDEN_CONFIG][1]),
            lambda v: cost[WARDEN_CONFIG][1] < 50)

    # Split cost by tier. The blended rows above stay (nothing is hidden), but
    # these are the numbers that describe the architecture honestly: the cheap
    # tier really is cheap, and the price of a judgment call is what it is.
    bypath = {c: classify_cost_by_path(dfs[c]) for c in dfs}
    add_row("  - fast path p50 / p95 (ms)", "report",
            "n/a",
            f"{_fmt_ms(bypath[COSINE_ONLY]['fast_p50'])} / {_fmt_ms(bypath[COSINE_ONLY]['fast_p95'])}",
            f"{_fmt_ms(bypath[WARDEN_CONFIG]['fast_p50'])} / {_fmt_ms(bypath[WARDEN_CONFIG]['fast_p95'])}",
            lambda v: True, is_report=True)
    add_row("  - escalated p50 / p95 (ms)", "report",
            "n/a",
            f"{_fmt_ms(bypath[COSINE_ONLY]['esc_p50'])} / {_fmt_ms(bypath[COSINE_ONLY]['esc_p95'])}",
            f"{_fmt_ms(bypath[WARDEN_CONFIG]['esc_p50'])} / {_fmt_ms(bypath[WARDEN_CONFIG]['esc_p95'])}",
            lambda v: True, is_report=True)

    add_row("LLM escalation rate", "< 15%",
            "n/a", _fmt_pct(esc[COSINE_ONLY]), _fmt_pct(esc[WARDEN_CONFIG]),
            lambda v: (esc[WARDEN_CONFIG] or 0) < 0.15)

    add_row("Throughput, writes/s", "> 100",
            _fmt_num(thr[NO_GATING]), _fmt_num(thr[COSINE_ONLY]), _fmt_num(thr[WARDEN_CONFIG]),
            lambda v: thr[WARDEN_CONFIG] > 100)

    add_row("False facts via correctly-classified write", "report",
            "n/a", _fmt_pct(false_via_correct[COSINE_ONLY]), _fmt_pct(false_via_correct[WARDEN_CONFIG]),
            lambda v: True, is_report=True)

    return pd.DataFrame(rows)


def raw_metrics(dfs: dict[str, pd.DataFrame], elapsed: dict[str, float]) -> dict:
    """Unformatted numeric values, for plotting (the table above formats
    everything to strings for CSV/PNG display)."""
    return {
        "false_facts_surviving": {c: false_facts_surviving(dfs[c]) for c in dfs},
        "honest_wrongly_blocked": {c: honest_wrongly_blocked(dfs[c]) for c in dfs},
        "four_class_accuracy": {c: four_class_accuracy(dfs[c]) for c in dfs},
        "per_class_prf1": {c: per_class_prf1(dfs[c]) for c in dfs},
        "confusion_matrix": {c: confusion_matrix(dfs[c]) for c in dfs},
        "cost_percentiles": {c: classify_cost_percentiles(dfs[c]) for c in dfs},
        "escalation_rate": {c: escalation_rate(dfs[c]) for c in dfs},
        "throughput": {c: throughput_writes_per_sec(elapsed[c], len(dfs[c])) for c in dfs},
        "false_via_correct": {c: false_via_correct_classification(dfs[c]) for c in dfs},
    }
