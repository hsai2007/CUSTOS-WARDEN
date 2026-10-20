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

## Architecture: trained fast path + optional LLM slow path

The design principle is that **most writes should never cost an LLM call**. Every write is classified by a small trained model on the local CPU. Only a *doubtful update* can leave the fast path.

```
incoming write + stored value (same key)
      │
      ▼
┌──────────────────────────────────────────────┐
│ FAST PATH — fine-tuned DeBERTa-v3-small      │   every write · ~200 ms CPU
│ cross-encoder, 4 output classes              │   no API key, no network
│                                              │
│ duplicate / refinement / contradiction       │
│   → final                                    │
│ update → must pass the update margin:        │
│   z(update) − z(contradiction) ≥ 3           │
└──────────────┬───────────────────────────────┘
               │ predicted update, but gap < 3  (doubtful)
               ▼
┌──────────────────────────────────────────────┐
│ SLOW PATH (optional) — LLM judge             │   ~1.7 s per call
│ no LLM attached → doubtful update is blocked │
└──────────────────────────────────────────────┘
```

**Why the margin?** Applying a false "update" writes a false fact that every later reader sees. Blocking a true update only costs a retry. So the gate is biased against `update`: the model must beat the contradiction score by 3 logits, otherwise the write is blocked (or escalated). The margin only acts when the model predicts `update`; duplicate and refinement decisions are never touched.

The previous fast path (regex fact extraction + embeddings, 0.12 ms) is still in `warden/classifier.py` as the baseline.

---

## Results

### Locked test set — 511 writes on topics never seen in training

| Metric | Regex + LLM (before) | **Trained model, margin 3** |
|---|---|---|
| Four-class accuracy | 64.2% | **85.3%** |
| Macro F1 | 0.574 | **0.822** |
| Duplicate / Refinement F1 | 0.760 / 0.778 | **0.947 / 0.935** |
| Update / Contradiction F1 | 0.234 / 0.524 | **0.663 / 0.741** |
| Contradictions let through as update | 12.0% | **8.4%** |
| True updates blocked | 73.3% | **42.6%** |

The baseline is replayed from its cached LLM decisions on the same 511 writes, so both systems see identical inputs.

### Full gate — all 1000 writes (seed 2026)

| Metric | Target | No gating | Cosine only | Regex + LLM | **Trained model** |
|---|---|---|---|---|---|
| Four-class accuracy | > 74.3% | n/a | 30.8% | 64.8% | **87.3%** — MET |
| Macro F1 | report | n/a | n/a | n/a | **0.844** |
| Worst per-class F1 | > 0.70 | n/a | 0.1 | 0.3 | **0.73** — MET |
| Honest writes wrongly blocked | < 10% | 0% | 1.1% | 22.1% | **9.2%** — MET |
| False facts surviving in store | < 5% | 100% | 73.1% | 15.5% | **23.3%** — MISSED |
| LLM calls per write | < 15% | 0% | 0% | 7.1% | **0%** — MET |
| Classify latency p50 / p95 | report | n/a | 19.9 / 27.8 ms | 0.12 ms / 30 s | **200 / 352 ms** |
| Throughput (sequential) | > 100/s | 202457 | 23.0 | 1.8 | **4.5** — MISSED |

Trained-model numbers come from `python -m warden.eval_model` (no LLM attached, doubtful updates blocked), written to `results/model_fastpath/metrics.json` and `writes.csv`. Targets come from [ConflictRAG (arXiv 2605.17301)](https://arxiv.org/abs/2605.17301); no threshold was tuned to hit a target.

### What improved

- Accuracy **64.8% → 87.3%**, and every per-class F1 is now above 0.70. Update F1 went from 0.3 to 0.73.
- Honest writes wrongly blocked **22.1% → 9.2%**. The old gate was safe mainly because it blocked most true updates.
- No network on the write path. Throughput is 2.5× the old pipeline because nothing waits for an API.

### What did not, and why

**False facts surviving rose from 15.5% to 23.3%.** Of the 193 false writes, 45 entered the store:

| How the false write got in | Count |
|---|---|
| A *false refinement* the model correctly labelled `refinement`, so it was merged | 25 |
| A contradiction the model called `update` | 12 |
| A false update applied as an update | 5 |
| A false refinement called `update` | 3 |

Most of the gap is false refinements: the class is right, but the added detail is wrong. No sentence-pair classifier can see that; it needs an outside fact source. The old gate scored lower on this metric partly because it blocked 22.1% of honest writes.

**Latency.** ~200 ms per write is far slower than the 0.12 ms regex. Batching and int8 weights are the next steps.

---

## The trained fast path

| | |
|---|---|
| Base model | `cross-encoder/nli-deberta-v3-small` (DeBERTa-v3-small, ~140M parameters, already trained for NLI) |
| Head | replaced with a 4-class head: duplicate, refinement, update, contradiction |
| Training data | 20,000 public pairs (5,000 per class) + ~2,400 labelled writes from the WARDEN generator |
| Hardware | trained on a free Kaggle notebook; runs on CPU |
| Weights | `models/slow_stage_v3b/` (567 MB, not in git — see below) |

**Public sources, mapped to the four classes with written rules:**

| Class | Sources |
|---|---|
| duplicate | PAWS, MRPC, VitaminC, IteraTeR |
| refinement | WikiAtomicEdits, MultiNLI |
| update | VitaminC revisions, WikiFactDiff |
| contradiction | VitaminC, synthetic VitaminC pairs, MultiNLI |

Rows are split by source page/document so no group appears in two splits.

**Shortcut audit.** A blind logistic regression on 7 surface features (length, word overlap, "?", …) should score ~25% on balanced data. It scored 83.5% on the first build; after cleaning (drop Quora, length band, overlap floor for contradictions, add PAWS/MRPC) it scores 70.9%. The rest is partly real: a refinement *is* longer.

**Domain shift.** Development accuracy (489 writes):

| Training data | Dev accuracy |
|---|---|
| Public corpora only | 45.0% |
| + PAWS and MRPC | 59.5% |
| *Regex + LLM baseline* | *65.2%* |
| + generator writes (1 batch) | 84.3% |
| + generator writes (2 batches) | **88.8%** |

Generator writes never use the evaluation seed. The generator has only 26 topics, so every training write on a test-half topic was removed (18 of the 19 test topics overlapped).

**Dev / test split.** The 1000 seed-2026 writes are split by memory key: 489 dev, 511 test. Every choice used dev. Test was scored twice: once without the margin (86.1% accuracy, but 51.8% of contradictions let through) and once with it.

**Margin sweep (dev).** Rule fixed before the sweep: smallest margin that lets through ≤ ~15% of contradictions.

| Margin | Accuracy | Contradictions let through | True updates blocked |
|---|---|---|---|
| 0 | 88.8% | 58.5% | 7.0% |
| 1 | 90.4% | 38.5% | 11.3% |
| 2 | 92.0% | 20.0% | 14.8% |
| **3** | **89.6%** | **13.8%** | **28.7%** |
| 4 | 86.9% | 3.1% | 46.1% |
| 5 | 81.8% | 0% | 69.6% |

Full reasoning: DECISIONS.md D25–D33.

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
pip install -r requirements.txt   # includes torch + transformers for the trained model
```

**Get the trained model.** The weights are 567 MB, above GitHub's file limit, so they are not in the repo. Download `slow_stage_v3b.zip` from **[ADD MODEL LINK HERE]** and extract it so this file exists:

```
models/slow_stage_v3b/config.json
```

The LLM slow path and corpus regeneration need an API key. Copy `.env.example` to `.env` and fill in one provider:

```
WARDEN_LLM_PROVIDER=groq
GROQ_API_KEY=your-key-here
```

Groq, Gemini, OpenRouter, Mistral and local Ollama are supported (any OpenAI-compatible endpoint). `.env` is gitignored — **never commit it**.

Generated text and LLM decisions are cached under `data/`, so **all published results and demos reproduce offline with no key at all**.

## Running

```bash
pytest tests/ -q                      # unit tests (the model test is skipped if models/ is missing)

python -m warden.eval_model           # trained model on all 1000 writes, offline, ~4 min on CPU
                                      # -> results/model_fastpath/metrics.json + writes.csv

python -m warden.demo m7              # two agents; every write judged by the trained model
python -m warden.demo m7llm           # same, plus the LLM slow path for doubtful updates
python -m warden.demo m1              # any single module: m1 m2 m3 m4 m5 m5fix m6 m7
python -m warden.demo all             # every module demo in order

python -m warden.generate_results --corpus llm --oracle llm \
    --n-writes 1000 --num-keys 40 --seed 2026     # regex + LLM baseline figures
```

`m7llm` prints, for each write, which path decided it and the update–contradiction gap. New LLM answers are saved to `data/llm_oracle_cache.json`, so after one online run the demo is fully offline.

## Module map

| Module | File | Role |
|---|---|---|
| M1 | `store.py` | versioned in-memory store; full history, nothing overwritten |
| M2 | `facts.py` | extracts dates, numeric magnitudes, negation |
| M3 | `classifier.py` | regex fast path + LLM escalation (baseline); `LLMOracle`; `CosineOnlyClassifier` ablation |
| M3 | `model_classifier.py` | **trained DeBERTa fast path with update margin** |
| M4 | `gate.py` | maps class → action, writes to the store |
| M5 | `workload.py`, `domains.py`, `renderers.py`, `llm_workload.py` | labelled synthetic workload |
| M6 | `harness.py`, `metrics.py`, `results_gen.py`, `generate_results.py`, `eval_model.py` | evaluation and figures |
| M7 | `agents.py`, `gateway.py` | two LangGraph agents writing through the public API |

All published numbers derive from frozen 12-column logs (`results/writes.csv`, `results/model_fastpath/writes.csv`):

```
ts, config, write_id, key, true_class, predicted_class, action,
is_false, version_read, version_at_commit, classify_ms, escalated
```

The figures in `results/*.png` are from the regex + LLM baseline.

## Limitations

- **One generator.** Training writes and test writes come from the same generator, so these are in-distribution numbers. The topic filter makes the test harder but does not remove this.
- **False refinements and plausible false updates.** A wrong detail, or a false change written in real change language, cannot be caught from the two sentences alone. This needs an outside fact source.
- **The margin moves between splits.** True updates blocked: 28.7% on dev, 42.6% on test.
- **M7 agent claims are hardcoded**, not LLM-generated. The agents exercise the gateway API; they are not autonomous.
- **The classifier is evaluated on the labelled pair**, not on live store contents (see DECISIONS.md D24).
- **Sequential writes only.** `version_read` / `version_at_commit` are logged for future concurrency work but nothing reads them yet.

## DECISIONS.md

Every deviation, ambiguity and bug is logged in [`DECISIONS.md`](DECISIONS.md). D1–D24 cover the regex + LLM system, including four bugs that would each have silently corrupted published numbers:

- a reasoning model returning empty strings that poisoned **26% of a corpus**
- a negation regex missing typographic apostrophes, blocking **~60% of honest writes**
- an evaluation flaw where blocked writes invalidated later labels, costing **35 accuracy points**
- a fabricated `time.sleep(0.002)` that made two latency targets pass for a hardcoded string

D25–D33 cover the trained fast path: data sources and mapping, the shortcut audit, domain shift, the topic filter, the margin rule, final results, and a cache-flush bug in the corpus generator.
