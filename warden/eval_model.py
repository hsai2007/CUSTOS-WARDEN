"""Evaluate the trained fast path (M8) on the seed-2026 LLM corpus.

    python -m warden.eval_model

* Makes ZERO API calls: the corpus is replayed from data/llm_workload_cache.json
  and a cache miss is an error, never a silent Groq call.
* Uses the same dev/test split as training (DECISIONS.md D29): writes are
  grouped by memory key, every second key is DEV, the rest are TEST. No
  seed-2026 write was ever trained on, and no TEST-half subject appears in
  the training writes.
* Writes to results/model_fastpath/ only -- the published results/ files
  produced by generate_results are left untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from warden.gate import WardenGate
from warden.harness import run_workload
from warden.llm_client import PROJECT_ROOT
from warden.llm_workload import LLMRenderer
from warden.metrics import (
    classify_cost_percentiles,
    false_facts_surviving,
    four_class_accuracy,
    honest_wrongly_blocked,
    per_class_prf1,
)
from warden.model_classifier import DEFAULT_MARGIN, DEFAULT_MODEL_DIR, ModelFastPath
from warden.workload import BOOTSTRAP, CLASSES, WorkloadConfig, generate

OUT_DIR = PROJECT_ROOT / "results" / "model_fastpath"


class CacheOnlyRenderer(LLMRenderer):
    """Replay the cached corpus; refuse to call any API."""

    def render(self, req, rng):
        hit = self.cache.get(f"{self.model_id}::{req.cache_key()}")
        if hit is None:
            raise RuntimeError("corpus write missing from cache -- refusing to call the API "
                               "(set WARDEN_LLM_PROVIDER=groq; the published corpus is cached under it)")
        return hit


def summarize(df: pd.DataFrame) -> dict:
    d = df[df["true_class"] != BOOTSTRAP]
    f1 = per_class_prf1(d)
    contra, upd = d[d.true_class == "contradiction"], d[d.true_class == "update"]
    return {
        "writes": int(len(d)),
        "accuracy": four_class_accuracy(d),
        **{f"{c}_f1": f1[c]["f1"] for c in CLASSES},
        "macro_f1": sum(f1[c]["f1"] for c in CLASSES) / len(CLASSES),
        "contradictions_let_through_as_update": float((contra.predicted_class == "update").mean()),
        "updates_wrongly_blocked_as_contradiction": float((upd.predicted_class == "contradiction").mean()),
        "false_facts_surviving": false_facts_surviving(d),
        "honest_writes_wrongly_blocked": honest_wrongly_blocked(d),
        "classify_ms_p50_p95": classify_cost_percentiles(d),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-writes", type=int, default=1000)
    ap.add_argument("--num-keys", type=int, default=40)
    args = ap.parse_args()

    os.environ.setdefault("WARDEN_LLM_PROVIDER", "groq")
    events = generate(WorkloadConfig(n_writes=args.n_writes, num_keys=args.num_keys, seed=args.seed),
                      renderer=CacheOnlyRenderer())

    clf = ModelFastPath(model_dir=args.model_dir, margin=args.margin)
    start = time.perf_counter()
    rows, _store = run_workload(events, "warden_model", gate=WardenGate(clf))
    elapsed = time.perf_counter() - start
    df = pd.DataFrame(rows)

    keys = sorted({e.key for e in events if e.true_class != BOOTSTRAP})
    dev_keys = set(keys[::2])
    report = {
        "model_dir": args.model_dir, "margin": args.margin, "seed": args.seed,
        "elapsed_s": round(elapsed, 1), "writes_per_s": round(len(df) / elapsed, 2),
        "all": summarize(df),
        "dev": summarize(df[df.key.isin(dev_keys)]),
        "test": summarize(df[~df.key.isin(dev_keys)]),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "writes.csv", index=False)
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    table = pd.DataFrame({k: report[k] for k in ("dev", "test", "all")})
    print(table.to_string())
    print(f"\n{report['writes_per_s']} writes/s on this machine | saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
