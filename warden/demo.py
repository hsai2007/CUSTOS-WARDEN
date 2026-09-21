"""Per-module live demo. One module at a time, projector-readable.

    python -m warden.demo m1     (and m2, m3, m4, m5, m6, m7, or `all`)

Every demo runs OFFLINE from cached data and finishes in seconds. Nothing
here calls a network API during class.
"""
from __future__ import annotations

import sys
import time

W = 78


def head(title: str, claim: str):
    print("\n" + "=" * W)
    print(f"  {title}")
    print(f"  CLAIM: {claim}")
    print("=" * W)


def demo_m1():
    head("M1 - STORE", "every write is kept; nothing is ever silently overwritten")
    from warden.store import Store

    s = Store()
    for v in ["ships Friday", "ships Friday at 3pm", "ships Monday"]:
        ver = s.write("launch", v)
        print(f"   write v{ver}: {v}")
    print(f"\n   read() -> v{s.read('launch').version}: {s.read('launch').value}")
    print("   full history still available:")
    for r in s.history("launch"):
        print(f"      v{r.version}: {r.value}")

    print("\n   Contrast - a plain dict (what most systems do):")
    d = {}
    for v in ["ships Friday", "ships Friday at 3pm", "ships Monday"]:
        d["launch"] = v
    print(f"      {d}   <-- the first two values are GONE")


def demo_m2():
    head("M2 - FACT EXTRACTOR", "pulls comparable facts that similarity alone cannot see")
    from datetime import date
    from warden import facts as F

    ref = date(2026, 9, 14)
    for text in ["The release ships on Friday.", "The release ships on next Friday."]:
        f = F.extract(text, reference_date=ref)
        print(f"   {text:<38} -> date={f.date}")
    print("   ^ near-identical sentences, SEVEN DAYS apart\n")

    for text in ["Revenue was four million dollars.", "Revenue was 4,000,000 dollars."]:
        print(f"   {text:<38} -> numbers={F.extract(text).numbers}")
    print("   ^ different spellings, SAME number\n")

    for text in ["We ship on Friday.", "We do not ship on Friday."]:
        print(f"   {text:<38} -> negated={F.extract(text).negated}")


def demo_m5():
    head("M5 - WORKLOAD GENERATOR", "ground truth is assigned BEFORE any text is written")
    from warden.workload import WorkloadConfig, generate, dump_human_readable

    events = generate(WorkloadConfig(n_writes=200, num_keys=20, seed=7))
    print(dump_human_readable(events, n=8))
    print("\n   Every row carries its own true label. The classifier never sees it.")


def demo_m5_fix():
    head("M5 - THE DEFECT WE FOUND AND FIXED", "the first corpus was solvable by two regexes")
    from warden.classifier import REFINEMENT_DETAIL_PATTERN, STRONG_MARKER_PATTERN
    from warden.llm_workload import LLMRenderer
    from warden.workload import BOOTSTRAP, WorkloadConfig, generate

    cfg = WorkloadConfig(n_writes=200, num_keys=20, seed=3)
    print(f"   {'':<15}{'TEMPLATE corpus':>24}{'LLM corpus':>20}")
    print(f"   {'class':<15}{'marker':>12}{'detail':>12}{'marker':>10}{'detail':>10}")
    tmpl = [e for e in generate(cfg) if e.true_class != BOOTSTRAP]
    try:
        llm = [e for e in generate(cfg, renderer=LLMRenderer()) if e.true_class != BOOTSTRAP]
    except Exception as exc:
        print(f"   (LLM corpus unavailable: {exc})")
        return
    for c in ("duplicate", "refinement", "update", "contradiction"):
        row = []
        for corpus in (tmpl, llm):
            s = [e for e in corpus if e.true_class == c]
            row.append(sum(bool(STRONG_MARKER_PATTERN.search(e.incoming_text)) for e in s) / len(s))
            row.append(sum(bool(REFINEMENT_DETAIL_PATTERN.search(e.incoming_text)) for e in s) / len(s))
        print(f"   {c:<15}{row[0]*100:>11.0f}%{row[1]*100:>11.0f}%{row[2]*100:>9.0f}%{row[3]*100:>9.0f}%")
    print("\n   TEMPLATE: marker=100% for update, 0% everywhere else -> one regex solves it.")
    print("   LLM:      contradictions now carry markers too -> the task is real.")


def demo_m3():
    head("M3 - CLASSIFIER", "facts decide first; only the ambiguous band costs an LLM call")
    from warden.classifier import FastPathEscalator

    clf = FastPathEscalator()
    cases = [
        ("The launch ships on Friday.", "The launch is shipping on Friday.", "same fact, reworded"),
        ("The launch ships on Friday.", "The launch ships on Friday, at 3pm PT.", "adds a detail"),
        ("The launch ships on Friday.", "The launch ships on Monday.", "conflicts, no reason"),
        ("The launch ships on Friday.", "The launch has been moved to Monday.", "claims a change"),
    ]
    print(f"   {'incoming':<46}{'predicted':<15}{'ms':>6}  esc")
    for stored, incoming, _note in cases:
        r = clf.classify(stored, incoming)
        print(f"   {incoming[:44]:<46}{r.predicted_class:<15}{r.classify_ms:>6.2f}  {'YES' if r.escalated else '-'}")
    print(f"\n   escalation rate this run: {clf.escalation_rate*100:.0f}% - the rest were free")


def demo_m4():
    head("M4 - GATE", "each class gets a different action; the store proves it")
    from warden.classifier import FastPathEscalator
    from warden.gate import WardenGate
    from warden.store import Store

    store, gate = Store(), WardenGate(FastPathEscalator())
    store.write("launch", "The launch ships on Friday.")
    writes = [
        "The launch is shipping on Friday.",
        "The launch ships on Friday, at 3pm PT.",
        "The launch has been moved to Monday.",
        "The launch ships on Thursday.",
    ]
    for text in writes:
        before = store.read("launch").version
        r = gate.process(store, "launch", store.read("launch").value, text)
        after = store.read("launch").version
        print(f"   {r.predicted_class:<14} -> {r.action:<9} v{before}->v{after}   {text[:40]}")
    print(f"\n   final stored value: {store.read('launch').value}")
    print(f"   history retained:   {len(store.history('launch'))} versions")


def demo_m6():
    head("M6 - EVALUATION HARNESS", "every published number comes from one frozen log")
    import pandas as pd
    from warden.harness import LOG_COLUMNS

    print(f"   frozen schema ({len(LOG_COLUMNS)} columns):")
    print(f"      {', '.join(LOG_COLUMNS)}\n")
    try:
        df = pd.read_csv("results/writes.csv")
        print(f"   results/writes.csv: {len(df)} rows across {df['config'].nunique()} configs")
        print(df.groupby("config").size().to_string().replace("\n", "\n      "))
        print("\n   sample rows:")
        print(df.head(3).to_string(index=False, max_colwidth=22))
    except FileNotFoundError:
        print("   (run `python -m warden.generate_results` first)")


def demo_m7():
    head("M7 - AGENT LAYER", "two LangGraph agents; one gets visibly blocked")
    from warden.agents import run_demo

    for log in run_demo():
        flag = "  <-- BLOCKED" if log.action == "blocked" else ""
        print(f"   {log.agent:<12} {log.predicted_class:<14} {log.action:<9} {log.claim[:36]}{flag}")
    print("\n   NOTE: agent claims are scripted, not LLM-generated (DECISIONS.md D16).")


DEMOS = {
    "m1": demo_m1, "m2": demo_m2, "m3": demo_m3, "m4": demo_m4,
    "m5": demo_m5, "m5fix": demo_m5_fix, "m6": demo_m6, "m7": demo_m7,
}
ORDER = ["m1", "m2", "m5", "m5fix", "m6", "m3", "m4", "m7"]


def main():
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    names = ORDER if which == "all" else [which]
    for n in names:
        if n not in DEMOS:
            print(f"unknown demo {n!r}; choose from: {', '.join(ORDER)} or 'all'")
            return
        t = time.perf_counter()
        DEMOS[n]()
        print(f"\n   [{n} completed in {time.perf_counter()-t:.1f}s]")


if __name__ == "__main__":
    main()
