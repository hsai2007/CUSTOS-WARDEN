"""Breaks down surviving false facts (WARDEN config) by true class and by
whether the surviving write was correctly classified. Answers a question
the frozen metrics table doesn't: false_facts_surviving (24.1%) and
false_via_correct_classification (24.0%) are nearly identical -- this
shows that's because almost every surviving false fact got in through a
write the classifier labeled correctly, not through a misclassification.

Reads results/writes.csv (already produced by generate_results.py) rather
than re-running anything, since the log has everything this needs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from warden.harness import WARDEN_CONFIG
from warden.workload import BOOTSTRAP, CLASSES

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SURVIVING_ACTIONS = frozenset({"apply", "merge"})


def build_table(warden_df: pd.DataFrame) -> pd.DataFrame:
    df = warden_df[warden_df["true_class"] != BOOTSTRAP].copy()
    df["is_false"] = df["is_false"].astype(bool)
    false_rows = df[df["is_false"]]

    rows = []
    for cls in CLASSES:
        cls_false = false_rows[false_rows["true_class"] == cls]
        n_false = len(cls_false)
        survived = cls_false[cls_false["action"].isin(SURVIVING_ACTIONS)]
        n_survived = len(survived)
        n_correct = int((survived["predicted_class"] == survived["true_class"]).sum())
        n_misclassified_survivor = n_survived - n_correct
        rows.append({
            "True class": cls,
            "False facts": n_false,
            "Survived": n_survived,
            "Survived %": f"{(n_survived / n_false * 100):.1f}%" if n_false else "n/a",
            "Survived via correct classification": n_correct,
            "Survived via misclassification": n_misclassified_survivor,
        })

    total_false = len(false_rows)
    total_survived = int(false_rows["action"].isin(SURVIVING_ACTIONS).sum())
    survived_rows = false_rows[false_rows["action"].isin(SURVIVING_ACTIONS)]
    total_correct = int((survived_rows["predicted_class"] == survived_rows["true_class"]).sum())
    rows.append({
        "True class": "TOTAL",
        "False facts": total_false,
        "Survived": total_survived,
        "Survived %": f"{(total_survived / total_false * 100):.1f}%" if total_false else "n/a",
        "Survived via correct classification": total_correct,
        "Survived via misclassification": total_survived - total_correct,
    })
    return pd.DataFrame(rows)


def render(table: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(table) + 1.8))
    ax.axis("off")
    col_labels = list(table.columns)
    cell_text = table.values.tolist()
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        elif cell_text[row - 1][0] == "TOTAL":
            cell.set_facecolor("#dfe6e9")
            cell.set_text_props(weight="bold")
    ax.set_title(
        "WARDEN: surviving false facts by true class and classification correctness\n"
        "(almost all survivors got in through a write correctly classified, not a misclassification)",
        fontsize=11, weight="bold", pad=20,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(RESULTS_DIR / "writes.csv")
    warden_df = df[df["config"] == WARDEN_CONFIG]
    table = build_table(warden_df)
    print(table.to_string(index=False))
    render(table, RESULTS_DIR / "survival_breakdown.png")
    print(f"\nWrote {RESULTS_DIR / 'survival_breakdown.png'}")


if __name__ == "__main__":
    main()
