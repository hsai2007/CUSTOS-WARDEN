"""Produces every file under results/ from one full-scale seeded run.
Run as: python -m warden.generate_results
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from warden.agents import run_demo as run_agent_demo
from warden.harness import COSINE_ONLY, NO_GATING, WARDEN_CONFIG
from warden.metrics import (
    classify_cost_percentiles,
    confusion_matrix,
    false_facts_surviving,
    honest_wrongly_blocked,
    per_class_prf1,
)
from warden.results_gen import (
    CONFLICTRAG_WORST_F1_REFERENCE,
    CORPUS_LLM,
    CORPUS_TEMPLATE,
    ORACLE_LLM,
    ORACLE_STUB,
    WORST_F1_TARGET,
    build_metrics_table,
    build_renderer,
    corpus_fingerprint,
    run_all_configs,
)
from warden.workload import BOOTSTRAP, CLASSES

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CONFIG_LABELS = {NO_GATING: "No gating", COSINE_ONLY: "Cosine only", WARDEN_CONFIG: "WARDEN"}


def ensure_dir():
    RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- item 8 --
def write_writes_csv(dfs: dict[str, pd.DataFrame]):
    combined = pd.concat(dfs.values(), ignore_index=True)
    combined.to_csv(RESULTS_DIR / "writes.csv", index=False)
    return combined


# ---------------------------------------------------------------- item 1 --
def write_metrics_table(dfs, elapsed, corpus=CORPUS_TEMPLATE):
    table = build_metrics_table(dfs, elapsed)
    table.to_csv(RESULTS_DIR / "metrics_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axis("off")
    col_labels = list(table.columns)
    cell_text = table.values.tolist()
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        elif col == len(col_labels) - 1:
            status = cell_text[row - 1][-1]
            if status == "MET":
                cell.set_facecolor("#d4edda")
            elif status == "MISSED":
                cell.set_facecolor("#f8d7da")
    subtitle = ("M5 corpus: TEMPLATE (regex-separable -- see DECISIONS.md D17)"
                if corpus == CORPUS_TEMPLATE else "M5 corpus: LLM-generated (label-first)")
    ax.set_title(f"WARDEN metrics table\n{subtitle}", fontsize=13, weight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "metrics_table.png", dpi=150)
    plt.close(fig)
    return table


# ---------------------------------------------------------------- item 2 --
def write_confusion_matrix(dfs):
    cm = confusion_matrix(dfs[WARDEN_CONFIG])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("WARDEN four-class confusion matrix")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = cm.values[i, j]
            color = "white" if val > cm.values.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- item 3 --
def write_per_class_f1(dfs):
    prf1_cosine = per_class_prf1(dfs[COSINE_ONLY])
    prf1_warden = per_class_prf1(dfs[WARDEN_CONFIG])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(CLASSES))
    width = 0.35

    warden_f1 = [prf1_warden[c]["f1"] for c in CLASSES]
    cosine_f1 = [prf1_cosine[c]["f1"] for c in CLASSES]

    ax.bar(x - width / 2, cosine_f1, width, label="Cosine only", color="#95a5a6")
    ax.bar(x + width / 2, warden_f1, width, label="WARDEN", color="#2980b9")

    ax.axhline(WORST_F1_TARGET, color="#27ae60", linestyle="--", linewidth=1.5,
               label=f"Target ({WORST_F1_TARGET:.2f})")
    ax.axhline(CONFLICTRAG_WORST_F1_REFERENCE, color="#c0392b", linestyle=":", linewidth=1.5,
               label=f"ConflictRAG reported ({CONFLICTRAG_WORST_F1_REFERENCE:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1: Cosine only vs WARDEN")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "per_class_f1.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- item 4 --
def write_cost_breakdown(dfs):
    df = dfs[WARDEN_CONFIG]
    classified = df[df["predicted_class"] != "n/a"]
    fast = classified[~classified["escalated"]]["classify_ms"]
    escalated = classified[classified["escalated"]]["classify_ms"]

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Fast path", "Escalated"]
    p50s = [np.percentile(fast, 50), np.percentile(escalated, 50)]
    p95s = [np.percentile(fast, 95), np.percentile(escalated, 95)]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, p50s, width, label="p50", color="#2980b9")
    ax.bar(x + width / 2, p95s, width, label="p95", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("classify_ms")
    ax.set_yscale("log")
    ax.set_title("WARDEN classification cost: fast path vs escalated")
    for xi, (v50, v95) in enumerate(zip(p50s, p95s)):
        ax.text(xi - width / 2, v50, f"{v50:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + width / 2, v95, f"{v95:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cost_breakdown.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- item 5 --
def write_false_facts_surviving(dfs):
    configs = [NO_GATING, COSINE_ONLY, WARDEN_CONFIG]
    ffs = [false_facts_surviving(dfs[c]) * 100 for c in configs]
    hwb = [honest_wrongly_blocked(dfs[c]) * 100 for c in configs]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(configs))
    width = 0.35
    ax.bar(x - width / 2, ffs, width, label="False facts surviving", color="#c0392b")
    ax.bar(x + width / 2, hwb, width, label="Honest writes wrongly blocked", color="#7f8c8d")
    ax.axhline(5, color="#c0392b", linestyle="--", linewidth=1, alpha=0.6, label="False-facts target (5%)")
    ax.axhline(10, color="#7f8c8d", linestyle="--", linewidth=1, alpha=0.6, label="Wrongly-blocked target (10%)")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs])
    ax.set_ylabel("%")
    ax.set_title("Safety vs cost, overlaid across all three configs")
    ax.legend(fontsize=8)
    for xi, v in enumerate(ffs):
        ax.text(xi - width / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    for xi, v in enumerate(hwb):
        ax.text(xi + width / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "false_facts_surviving.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- item 6 --
def _render_scene(rows: list[dict], columns: list[str], title: str, txt_path: Path, png_path: Path):
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    widths = {c: min(w, 55) for c, w in widths.items()}

    def fmt_row(vals):
        parts = []
        for c in columns:
            s = str(vals[c])
            if len(s) > widths[c]:
                s = s[: widths[c] - 3] + "..."
            parts.append(s.ljust(widths[c]))
        return "  ".join(parts)

    header = fmt_row({c: c for c in columns})
    sep = "-" * len(header)
    lines = [title, sep] + [header, sep] + [fmt_row(r) for r in rows]
    text = "\n".join(lines)
    txt_path.write_text(text, encoding="utf-8")

    max_line_len = max(len(line) for line in lines)
    fig_w = min(20, max(8, max_line_len * 0.09))
    fig_h = max(1.2, 0.28 * len(lines) + 0.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.text(0.01, 0.99, text, family="monospace", fontsize=10, va="top", ha="left", transform=ax.transAxes)
    fig.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def write_scene_demo(dfs, n=10, corpus=CORPUS_TEMPLATE, seed=2026, n_writes=3000, num_keys=60):
    df = dfs[WARDEN_CONFIG]
    classified = df[df["true_class"] != BOOTSTRAP].copy()
    picks = []
    for cls in CLASSES:
        sub = classified[classified["true_class"] == cls]
        if len(sub):
            picks.append(sub.iloc[0])
    remaining = n - len(picks)
    extra = classified[~classified.index.isin([p.name for p in picks])].sample(
        n=min(remaining, len(classified)), random_state=0
    )
    chosen = pd.concat([pd.DataFrame(picks), extra]).head(n)

    # need the actual stored/incoming text -- re-run to capture, since the
    # log doesn't store text (by design: writes.csv is metrics-only)
    from warden.classifier import FastPathEscalator
    from warden.gate import WardenGate
    from warden.store import Store
    from warden.workload import WorkloadConfig, generate as gen

    events = gen(WorkloadConfig(n_writes=n_writes, num_keys=num_keys, seed=seed),
                 renderer=build_renderer(corpus))
    store = Store()
    gate = WardenGate(FastPathEscalator())
    text_by_write_id = {}
    for e in events:
        version_read = store.read(e.key).version if store.read(e.key) else 0
        stored_text = store.read(e.key).value if version_read > 0 else None
        if e.true_class == BOOTSTRAP or version_read == 0:
            store.write(e.key, e.incoming_text)
            continue
        result = gate.process(store, e.key, stored_text, e.incoming_text)
        text_by_write_id[e.write_id] = {
            "stored value": (stored_text or "")[:55],
            "incoming value": e.incoming_text[:55],
            "true class": e.true_class,
            "predicted class": result.predicted_class,
            "action": result.action,
            "cost ms": f"{result.classify_ms:.2f}",
        }

    rows = [text_by_write_id[wid] for wid in chosen["write_id"] if wid in text_by_write_id]
    columns = ["stored value", "incoming value", "true class", "predicted class", "action", "cost ms"]
    _render_scene(rows, columns, "WARDEN scene demo (sample writes)",
                  RESULTS_DIR / "scene_demo.txt", RESULTS_DIR / "scene_demo.png")


# ---------------------------------------------------------------- item 7 --
def write_scene_agents():
    logs = run_agent_demo()
    rows = [{
        "agent": log.agent, "claim": log.claim[:45], "predicted class": log.predicted_class,
        "action": log.action, "escalated": log.escalated,
    } for log in logs]
    columns = ["agent", "claim", "predicted class", "action", "escalated"]
    title = (
        "M7 agent demo (LangGraph agents through the gateway)\n"
        "SCRIPTED: agent claims below are hardcoded strings, not LLM-generated. "
        "No live LLM call anywhere in this run (no ANTHROPIC_API_KEY; see DECISIONS.md D7)."
    )
    _render_scene(rows, columns, title,
                  RESULTS_DIR / "scene_agents.txt", RESULTS_DIR / "scene_agents.png")


# ---------------------------------------------------------------- item 9 --
MODULE_TEST_FILES = [
    ("M1 Store", ["tests/test_m1_store.py"]),
    ("M2 Fact extractor", ["tests/test_m2_facts.py"]),
    ("M5 Workload generator", ["tests/test_m5_workload.py"]),
    ("M6 Evaluation harness", ["tests/test_m6_harness.py", "tests/test_m6_output_files.py"]),
    ("M3 Classifier", ["tests/test_m3_classifier.py"]),
    ("M4 Gate", ["tests/test_m4_gate.py"]),
    ("M7 Agent layer", ["tests/test_m7_agents.py"]),
]


def _run_pytest_files(paths: list[str]) -> tuple[int, int]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-q"],
        capture_output=True, text=True, cwd=str(RESULTS_DIR.parent),
    )
    out = proc.stdout + proc.stderr
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    return passed, failed


def write_module_status():
    rows = []
    for name, paths in MODULE_TEST_FILES:
        passed, failed = _run_pytest_files(paths)
        rows.append((name, passed, failed))

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.5))
    ax.axis("off")
    col_labels = ["Module", "Tests passing", "Tests failing", "Status"]
    cell_text = [[name, str(p), str(f), "COMPLETE" if f == 0 and p > 0 else "INCOMPLETE"] for name, p, f in rows]
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        elif col == 3:
            cell.set_facecolor("#d4edda" if cell_text[row - 1][3] == "COMPLETE" else "#f8d7da")
    ax.set_title("Module completion status", fontsize=14, weight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "module_status.png", dpi=150)
    plt.close(fig)
    return rows


def write_provenance(events, corpus, seed, n_writes, num_keys):
    """Ties every file in results/ to the exact corpus that produced it.
    The 12-column log schema is frozen, so provenance lives here rather than
    as a 13th column (DECISIONS.md D19)."""
    record = {
        "corpus": corpus,
        "corpus_fingerprint": corpus_fingerprint(events),
        "seed": seed,
        "n_writes": n_writes,
        "num_keys": num_keys,
        "unique_texts": len({e.incoming_text for e in events}),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (RESULTS_DIR / "corpus_provenance.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=[CORPUS_TEMPLATE, CORPUS_LLM], default=CORPUS_TEMPLATE,
                    help="which M5 corpus to evaluate against")
    ap.add_argument("--oracle", choices=[ORACLE_STUB, ORACLE_LLM], default=ORACLE_LLM,
                    help="M3 escalation adjudicator: llm (real, default) or stub (ablation)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-writes", type=int, default=3000)
    ap.add_argument("--num-keys", type=int, default=60)
    args = ap.parse_args()

    ensure_dir()
    print(f"Running all three configs at full scale (corpus={args.corpus})...")
    events, dfs, elapsed = run_all_configs(
        seed=args.seed, n_writes=args.n_writes, num_keys=args.num_keys,
        corpus=args.corpus, oracle=args.oracle)

    print("Writing corpus provenance...")
    record = write_provenance(events, args.corpus, args.seed, args.n_writes, args.num_keys)
    print(f"  corpus={record['corpus']} fingerprint={record['corpus_fingerprint']} "
          f"unique_texts={record['unique_texts']}")

    print("Writing writes.csv...")
    write_writes_csv(dfs)

    print("Building metrics table...")
    table = write_metrics_table(dfs, elapsed, corpus=args.corpus)
    print(table.to_string(index=False))

    print("Building confusion matrix...")
    write_confusion_matrix(dfs)

    print("Building per-class F1 chart...")
    write_per_class_f1(dfs)

    print("Building cost breakdown chart...")
    write_cost_breakdown(dfs)

    print("Building false-facts-surviving chart...")
    write_false_facts_surviving(dfs)

    print("Building scene demo...")
    write_scene_demo(dfs, corpus=args.corpus, seed=args.seed,
                     n_writes=args.n_writes, num_keys=args.num_keys)

    print("Building agent scene demo (M7)...")
    write_scene_agents()

    print("Running module status (this re-runs each module's tests)...")
    write_module_status()

    print("Done. See results/")


if __name__ == "__main__":
    main()
