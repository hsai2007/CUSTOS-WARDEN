"""Renders results/architecture.png - a layered system architecture diagram.

Not a flowchart: it shows the system's STRUCTURE (layers, components, ownership
and the boundary between them), not a sequence of steps. The central claim of
the layout is that WARDEN (the product) and the evaluation harness are separate
systems that touch at exactly one place -- the frozen 12-column write log.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

OWNER_A = "#cfe3f7"   # system under test
OWNER_B = "#fde4cf"   # evaluation harness
NEUTRAL = "#e6e6e6"
EDGE = "#2c3e50"


def box(ax, x, y, w, h, title, subtitle="", color=NEUTRAL, fontsize=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=color, edgecolor=EDGE, linewidth=1.3))
    ax.text(x + w / 2, y + h * (0.62 if subtitle else 0.5), title,
            ha="center", va="center", fontsize=fontsize, weight="bold", color="#1a1a1a")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.26, subtitle, ha="center", va="center",
                fontsize=fontsize - 2.2, color="#444", style="italic")


def arrow(ax, start, end, style="-|>", color=EDGE, lw=1.4, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13,
                                 color=color, linewidth=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def build():
    fig, ax = plt.subplots(figsize=(15.5, 10.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---------------------------------------------------------------- bands --
    ax.add_patch(FancyBboxPatch((1.5, 55), 97, 41, boxstyle="round,pad=0.4",
                                facecolor="#f7fbff", edgecolor="#9db8d2", linewidth=1.6))
    ax.text(3.2, 93.4, "SYSTEM UNDER TEST  —  WARDEN (this is what would ship)",
            fontsize=12, weight="bold", color="#1f4e79")

    ax.add_patch(FancyBboxPatch((1.5, 4), 97, 38, boxstyle="round,pad=0.4",
                                facecolor="#fffaf5", edgecolor="#d2b49d", linewidth=1.6))
    ax.text(3.2, 39.2, "EVALUATION HARNESS  —  measurement only (ships with nothing)",
            fontsize=12, weight="bold", color="#8a4b1f")

    # ------------------------------------------------------- product layer ---
    box(ax, 5, 83, 22, 7.5, "WardenGateway", "gateway.py  ·  submit(key, text)", OWNER_A)
    box(ax, 5, 71, 22, 8.5, "M4  Gate", "gate.py  ·  class → action", OWNER_A)
    box(ax, 5, 58, 22, 9.5, "M1  Store", "store.py  ·  versions + history", OWNER_A)

    box(ax, 34, 69, 30, 21, "", "", OWNER_A)
    ax.text(49, 88.0, "M3  Classifier", ha="center", fontsize=11, weight="bold", color="#1a1a1a")
    ax.text(49, 86.0, "classifier.py", ha="center", fontsize=8.5, style="italic", color="#444")
    box(ax, 36.5, 78.5, 12.5, 6, "Fast path", "facts + regex", "#ffffff", 8.5)
    box(ax, 50.5, 78.5, 12, 6, "Cosine", "embeddings", "#ffffff", 8.5)
    box(ax, 36.5, 71, 26, 6, "Escalation oracle", "StubOracle  |  LLMOracle", "#ffffff", 8.5)

    box(ax, 70, 82, 25, 7.5, "M2  Fact extractor", "facts.py  ·  dates/numbers/negation", OWNER_A)
    box(ax, 70, 72, 25, 7.5, "Embeddings", "embeddings.py  ·  MiniLM", OWNER_A)
    box(ax, 70, 62, 25, 7.5, "LLM client", "llm_client.py  ·  shared", NEUTRAL)

    box(ax, 34, 58, 30, 7.5, "M7  Agent layer", "agents.py · LangGraph · demo only", OWNER_B)

    # ------------------------------------------------------------- the seam --
    box(ax, 22, 45.5, 56, 7, "FROZEN WRITE LOG  —  results/writes.csv  (12 columns)",
        "ts · config · write_id · key · true_class · predicted_class · action · is_false · "
        "version_read · version_at_commit · classify_ms · escalated", "#d9ead3", 10.5)
    ax.text(79.5, 49, "the ONLY\ninterface", fontsize=8.5, style="italic", color="#38761d",
            ha="left", va="center", weight="bold")

    # ------------------------------------------------------- harness layer ---
    box(ax, 5, 27, 26, 9, "M5  Workload generator", "workload.py · labels FIRST", OWNER_B)
    box(ax, 5, 15.5, 12.5, 8, "Renderers", "template | LLM", OWNER_B, 8.5)
    box(ax, 18.5, 15.5, 12.5, 8, "Domains", "domains.py", OWNER_B, 8.5)
    box(ax, 5, 6, 26, 7, "Ground truth: true_class + is_false",
        "never visible to the classifier", "#f4cccc", 9)

    box(ax, 37, 27, 26, 9, "M6  Harness", "harness.py · runs 3 configs", OWNER_B)
    box(ax, 37, 15.5, 26, 8, "M6  Metrics", "metrics.py · reads log only", OWNER_B)
    box(ax, 37, 6, 26, 7, "Config ablations",
        "no gating | cosine only | WARDEN", "#ffffff", 9)

    box(ax, 69, 27, 26, 9, "Results generator", "generate_results.py", OWNER_B)
    box(ax, 69, 15.5, 26, 8, "9 output files", "tables · charts · scenes", OWNER_B)
    box(ax, 69, 6, 26, 7, "Provenance", "corpus_provenance.json", "#ffffff", 9)

    # ------------------------------------------------------------- arrows ----
    arrow(ax, (16, 83), (16, 79.5))          # gateway -> gate
    arrow(ax, (16, 71), (16, 67.5))          # gate -> store
    arrow(ax, (27, 75.5), (34, 77))          # gate -> classifier
    arrow(ax, (64, 84), (70, 85.5))          # classifier -> facts
    arrow(ax, (64, 78), (70, 76))            # classifier -> embeddings
    arrow(ax, (62.5, 72), (70, 67))          # oracle -> llm client
    arrow(ax, (36, 65.5), (24, 83), rad=0.18)   # agents -> gateway (public API only)
    ax.text(29.5, 70.5, "public API only", fontsize=7.5, style="italic", color="#666", rotation=52)

    arrow(ax, (31, 31.5), (37, 31.5))        # workload -> harness
    arrow(ax, (18, 27), (18, 23.5))          # workload -> renderers
    arrow(ax, (25, 27), (25, 23.5))          # workload -> domains
    arrow(ax, (50, 27), (50, 23.5))          # harness -> metrics
    arrow(ax, (63, 31.5), (69, 31.5))        # harness -> results
    arrow(ax, (82, 27), (82, 23.5))          # results -> files

    arrow(ax, (50, 45.5), (50, 36), color="#38761d", lw=2.0)     # log -> harness reads
    arrow(ax, (16, 58), (30, 52.5), color="#38761d", lw=2.0, rad=-0.15)  # store/product -> log
    arrow(ax, (18, 36), (30, 45.5), color="#8a4b1f", lw=1.6, ls=(0, (4, 2)), rad=0.15)
    ax.text(20.5, 41.5, "feeds writes", fontsize=7.5, style="italic", color="#8a4b1f")
    ax.text(51.5, 40.5, "every number\nderives from here", fontsize=8, style="italic",
            color="#38761d", ha="left")

    # -------------------------------------------------------------- legend ---
    ax.add_patch(FancyBboxPatch((66, 91.2), 29, 6.2, boxstyle="round,pad=0.25",
                                facecolor="white", edgecolor="#999", linewidth=1))
    ax.add_patch(FancyBboxPatch((67.3, 94.4), 2.2, 1.8, boxstyle="square,pad=0",
                                facecolor=OWNER_A, edgecolor=EDGE))
    ax.text(70.2, 95.3, "Owner A — system under test", fontsize=8.5, va="center")
    ax.add_patch(FancyBboxPatch((67.3, 92, ), 2.2, 1.8, boxstyle="square,pad=0",
                                facecolor=OWNER_B, edgecolor=EDGE))
    ax.text(70.2, 92.9, "Owner B — evaluation harness", fontsize=8.5, va="center")

    ax.set_title("WARDEN — System Architecture", fontsize=16, weight="bold", pad=14)
    fig.savefig(RESULTS_DIR / "architecture.png", dpi=170, bbox_inches="tight",
                facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {RESULTS_DIR / 'architecture.png'}")


if __name__ == "__main__":
    build()
