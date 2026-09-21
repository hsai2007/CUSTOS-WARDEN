# WARDEN

**W**rite **A**dmission and **R**esolution of **D**ivergent **EN**tries — write-time conflict gating for shared agent memory.

When several agents write to one memory store, the naive behaviour is last-write-wins: every write silently overwrites whatever was there. WARDEN puts a gate in front of the store. Each incoming write is classified against the currently stored value and acted on accordingly:

| Class | Meaning | Action |
|---|---|---|
| `duplicate` | same fact, reworded | **drop** — store unchanged |
| `refinement` | same fact plus detail | **merge** — version increments |
| `update` | the fact legitimately changed | **apply** — old value kept in history |
| `contradiction` | incompatible, not a real change | **blocked** — write recorded, store unchanged |

`update` vs `contradiction` is the hard boundary, and the reason the system exists.

---

## Architecture: two tiers

The design principle is that **most writes should never cost an LLM call**.

```
incoming write
      │
      ▼
┌─────────────────────────────────────────┐
│ FAST PATH  — regex facts + embeddings   │   93% of writes · 0.12 ms
│                                         │
│ facts agree            → duplicate      │
│ facts agree + detail   → refinement     │
│ facts differ, no       → contradiction  │
│   change-language                       │
└──────────────┬──────────────────────────┘
               │ facts differ AND the claim
               │ carries change-language
               ▼
┌─────────────────────────────────────────┐
│ SLOW PATH  — LLM adjudicator            │   7% of writes · 1.6 s
│ "real supersession, or a flat conflict   │
│  wearing update clothing?"               │
└─────────────────────────────────────────┘
```

Only the genuinely ambiguous band escalates. A claim like *"The mobile release is now scheduled for Monday"* carries change-language but may be a fabrication — no regex can tell, so it goes to the model.

---

## Results

Measured on 1000 LLM-generated writes (seed 2026, corpus fingerprint `b0e0e312a5f64761`).

| Metric | Target | No gating | Cosine only | **WARDEN** | |
|---|---|---|---|---|---|
| False facts surviving in store | < 5% | 100% | 73.1% | **15.5%** | MISSED |
| Honest writes wrongly blocked | < 10% | 0% | 1.1% | **22.1%** | MISSED |
| Four-class accuracy | > 74.3% | n/a | 30.8% | **64.8%** | MISSED |
| Worst per-class F1 | > 0.70 | n/a | 0.1 | **0.3** | MISSED |
| Duplicate / Refinement F1 | report | n/a | 0.2 / 0.4 | **0.8 / 0.8** | |
| Update / Contradiction F1 | report | n/a | 0.2 / 0.1 | **0.3 / 0.5** | |
| Fast-path p50 / p95 | report | n/a | 19.9 / 27.8 ms | **0.12 / 0.26 ms** | |
| Escalated p50 / p95 | report | n/a | n/a | **1.6 s / 30 s** | |
| LLM escalation rate | < 15% | n/a | 0% | **7.1%** | MET |
| Throughput | > 100/s | 202457 | 23.0 | **1.8** | MISSED |

Targets come from [ConflictRAG (arXiv 2605.17301)](https://arxiv.org/abs/2605.17301). **No threshold was tuned to hit a target** — every number is reported as measured, and most of them miss.

### What works

WARDEN cuts false facts surviving from **100% → 15.5%** and more than doubles the embedding-similarity baseline's accuracy (30.8% → 64.8%). Duplicate and refinement detection both reach 0.8 F1. The fast path resolves 93% of writes in 0.12 ms.

### What doesn't, and why

**Update recall is poor (F1 0.3)** — 151 of 216 genuine updates are misread as contradictions. The fast path recognises supersession through a fixed phrase list (`moved to`, `pushed to`, `revised to`), but real writing says *"is now 23 due to a surge in ticket volume"*. This single weakness drives both headline misses: over-blocking updates is why 22.1% of honest writes are rejected, and also why false-facts-surviving looks as low as it does.

**Throughput and p95 miss by design constraints, not bugs.** A network adjudicator on 7% of writes cannot reach 100 writes/s sequentially. The fast-path numbers are the ones that describe the part of the system under our control.

---

## The benchmark was broken first

The first version of this project reported **95.9% accuracy**. That number was wrong, and finding out why is the most useful thing in this repository.

The synthetic corpus was generated from templates. Measured over 3000 writes:

| true class | has supersession marker | has trailing detail-clause |
|---|---|---|
| duplicate | 0% | 0% |
| refinement | 0% | **100%** |
| update | **100%** | 0% |
| contradiction | 0% | 0% |

Two regular expressions solved the entire task. Worse, **10 of the 13 phrases in the classifier's marker regex appeared verbatim in the generator's own templates** — the exam and the student were written by the same hand. And 29 sentences appeared with *both* `is_false=True` and `is_false=False`, so identical inputs carried opposite labels.

The consequence went further than an inflated score: **the project could not measure its own thesis.** All 724 escalated writes were `update`, so a stub returning the constant `"update"` scored 100% on the escalation band. A real LLM and a hardcoded string were provably indistinguishable.

**The fix** (`warden/renderers.py`, `warden/llm_workload.py`): keep the seeded RNG owning ground truth — `true_class`, `is_false` and the fact values are all decided *before* any text exists — and hand only the final rendering step to an LLM, with RNG-chosen surface styles. Contradictions are now styled to wear change-language ~45% of the time; genuine updates are styled to omit it ~33% of the time.

Result on the rebuilt corpus:

| | Template corpus | LLM corpus |
|---|---|---|
| Unique sentences | 45% | **74%** |
| Marker ⇒ update | **100% / 0%** | 19.4% update vs **20.9% contradiction** |
| Sentences labelled both true and false | **29** | **0** |
| Reported accuracy | 95.9% | **64.8%** |

The marker is now mildly *anti*-correlated with the class it was written to detect. The honest score is 64.8%.

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
```

To regenerate the corpus or run the LLM adjudicator you need an API key. Copy `.env.example` to `.env` and fill in one provider:

```
WARDEN_LLM_PROVIDER=groq
GROQ_API_KEY=your-key-here
```

Groq, Gemini, OpenRouter, Mistral and local Ollama are supported (any OpenAI-compatible endpoint). `.env` is gitignored — **never commit it**.

Generated text and adjudicator decisions are cached under `data/`, so **the published results reproduce offline with no key at all**.

## Running

```bash
pytest tests/ -q                     # 87 tests

python -m warden.demo all            # per-module live demo, offline, seconds
python -m warden.demo m5fix          # the benchmark defect, before vs after

python -m warden.generate_results --corpus llm --oracle llm \
    --n-writes 1000 --num-keys 40 --seed 2026
```

`--corpus {template,llm}` selects the workload; `--oracle {llm,stub}` selects the adjudicator. `stub` is the ablation arm — the policy *"just trust the change-language"* — and the stub-vs-llm gap is what isolates the value of escalation.

## Module map

| Module | File | Role |
|---|---|---|
| M1 | `store.py` | versioned in-memory store; full history, nothing overwritten |
| M2 | `facts.py` | extracts dates, numeric magnitudes, negation |
| M3 | `classifier.py` | fast path + LLM escalation; `CosineOnlyClassifier` ablation |
| M4 | `gate.py` | maps class → action, writes to the store |
| M5 | `workload.py`, `domains.py`, `renderers.py`, `llm_workload.py` | labelled synthetic workload |
| M6 | `harness.py`, `metrics.py`, `results_gen.py`, `generate_results.py` | evaluation and figures |
| M7 | `agents.py`, `gateway.py` | two LangGraph agents writing through the public API |

All published numbers derive from one frozen 12-column log, `results/writes.csv`:

```
ts, config, write_id, key, true_class, predicted_class, action,
is_false, version_read, version_at_commit, classify_ms, escalated
```

## Limitations

- **M7 agent claims are hardcoded**, not LLM-generated. The agents exercise the gateway API and demonstrate a visible block; they are not autonomous. Labelled directly on `results/scene_agents.png`.
- **The classifier is evaluated on the labelled pair**, not on live store contents. Because the gate blocks writes, the store diverges from the generator's ground truth, and labels defined against the generator's value become invalid. Pinning the input costs realism and buys a well-posed measurement (29.9% vs 64.8% on the same classifier — see DECISIONS.md D24).
- **Sequential writes only.** `version_read` / `version_at_commit` are logged for future concurrency work but nothing currently reads them.
- **Escalated latency is inflated** by free-tier rate limiting (1.6 s median, 30 s p95); a paid tier measures ~700 ms.

## DECISIONS.md

Every deviation, ambiguity and bug is logged in [`DECISIONS.md`](DECISIONS.md) — 24 entries, including four bugs that would each have silently corrupted published numbers:

- a reasoning model returning empty strings that poisoned **26% of a corpus**
- a negation regex missing typographic apostrophes, blocking **~60% of honest writes**
- an evaluation flaw where blocked writes invalidated later labels, costing **35 accuracy points**
- a fabricated `time.sleep(0.002)` that made two latency targets pass for a hardcoded string
