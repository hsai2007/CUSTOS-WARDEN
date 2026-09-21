# WARDEN — DECISIONS.md

Log of every deviation/interpretation made where the spec was ambiguous or silent. Newest at bottom.

## D1 — `is_false` is orthogonal to `true_class` (user decision)

Asked the user directly since this defines the headline metric. Decision: `is_false`
is independent of `true_class`. False writes are injected mostly as `contradiction`
(which is false by construction — see D2) but also as a small share of `refinement`
and `update` writes, so the store can be corrupted through a write a perfect
classifier would still label correctly. Added a 13th metrics-table row, "False
facts via correctly-classified write" (report only, no target), to make that
gap visible rather than hiding it inside the headline number.

## D2 — Every `contradiction` is `is_false=True` by construction

M5 tracks a hidden ground-truth trajectory per key. A `contradiction` write is
defined as "incompatible with the stored value, not a legitimate update" —
under this model that is only possible if the content is factually wrong, so
`is_false` is not independently sampled for the contradiction class; it is
always `True`. This gives `false_fraction` a floor equal to the contradiction
share of `class_distribution` (see D3).

## D3 — `false_fraction` has a floor and `WorkloadConfig` validates it

Because every contradiction is false (D2), the overall false rate can never go
below the contradiction class's share of `class_distribution`. `WorkloadConfig.__post_init__`
raises `ValueError` if `false_fraction < class_distribution["contradiction"]`.
Chosen defaults: `class_distribution = {duplicate: .30, refinement: .30, update: .25,
contradiction: .15}`, `false_fraction = .20` (.15 floor + .05 spread across
refinement/update via `false_non_contradiction_share`).

## D4 — Bootstrap writes are a 5th, unclassified case

The very first write to a brand-new key can't be a duplicate/refinement/update/
contradiction — there is nothing stored yet to compare against. M5 emits it with
`true_class="bootstrap"`; the harness (M6) and every gate short-circuit it to
`action="apply"`, `predicted_class="bootstrap"` without invoking the classifier.
This does NOT violate "M3 classifies into exactly the four classes, never
returns anything else" — M3 is never called for bootstrap writes at all, the
gate/harness intercepts them first. Bootstrap rows are excluded from all
4-class accuracy/F1/confusion-matrix computations in `warden/metrics.py`.

## D5 — Date resolution semantics for "next X" / "last X"

Not a real NLP system — a synthetic, deterministic date resolver so tests are
reproducible. Given a reference date: bare weekday = the next (or same-day)
occurrence going forward. `next X` = that occurrence + 7 days (guarantees "friday"
and "next friday" always differ by exactly one week). `last X` = the most recent
past occurrence (7 days back if reference date itself is that weekday).

## D6 — No git repository initialized

Per top-level instructions, commits are only made when the user asks. No git
history is being kept for this build unless requested.

## D7 — No Anthropic API key available; M3 escalation stub

`ANTHROPIC_API_KEY` is not set in this environment. Per SCOPE, the LLM escalation
path is stubbed deterministically (a seeded pseudo-oracle with tunable accuracy,
not a live Haiku call). Escalation RATE is still measured and reported; escalation
ACCURACY is reported but flagged as coming from a stub, not a real model, per spec.

## D8 — No-gating baseline checkpoint (run before M3, per spec)

Workload: `n_writes=2000, num_keys=50, seed=7`, default class distribution/false
fraction. Result: **false_facts_surviving = 1.0 (100%)**, **honest_wrongly_blocked = 0.0**.
This is expected under blind-overwrite semantics (nothing is ever blocked) and
confirms the premise: there is a full 100 points of false-fact leakage for
WARDEN to close. Proceeding to M3 per spec.

## D9 — Cosine similarity is a poor discriminator across all 4 classes (measured)

Before picking `CosineOnlyClassifier` thresholds, measured real cosine similarity
of (stored_text, incoming_text) pairs from an 800-write sample, grouped by
`true_class`: duplicate mean=0.824, refinement mean=0.773, update mean=0.823,
contradiction mean=0.804 — all four classes overlap in the same 0.77-0.82 band
with wide (p10-p90) spread. Thresholds (duplicate>=.90, refinement>=.75,
update>=.60, else contradiction) are round numbers spanning this observed
range, fixed BEFORE looking at any downstream accuracy metric — not tuned to
hit a target, per the project's explicit rule against threshold-tuning.

## D10 — Escalation band and the StubOracle's honest limit

`FastPathEscalator`'s ambiguous band is "fact mismatch + supersession marker
present" (e.g. "moved to", "increased to", "no longer"). In this synthetic
corpus that band is reached only by genuine or fabricated ("false") `update`
writes — contradictions never carry supersession language, so they always
fast-path. Real and false updates are generated from the SAME marker-phrase
pools (see warden/domains.py), so they are textually indistinguishable; the
`StubOracle` (no `ANTHROPIC_API_KEY` available, see D7) can only trust the
supersession language and predicts "update". This is not a shortcut: a real
LLM given only these two strings would face the same limit. It is exactly
the gap D1's extra metric ("false facts via correctly-classified write") is
meant to surface, not hide.

## D11 — Bug found and fixed: "not" in an update-marker template tripped negation detection

`DateDomain.UPDATE_MARKERS` originally included "is now scheduled for {new},
not {old}". M2's negation detector correctly flags standalone "not", so a
stored value produced by this template got `negated=True` even though
nothing was actually being negated (it's a date supersession, not a boolean
claim) — this silently broke `_facts_compatible()` for later honest
refinement writes on the same key, misrouting them into the escalation band.
Fixed by rewording the template to "is now scheduled for {new} instead of
{old}" (already-marker vocabulary, no negation trigger). Caught by
`test_escalation_only_fires_on_the_ambiguous_band...` in test_m3_classifier.py
failing with an unexpected `refinement` in the escalated set. Also removed
`STATUS_UPDATE_MARKERS`, a dead constant left over from an earlier draft of
`StatusDomain.update()` that never referenced it.

## D12 — Two more bugs found and fixed while calibrating M3 against real data

1. **StatusDomain update phrasing vs. negation detection.** `update()`/`false_update()`
   used flowery phrasing ("has now been cancelled", "has just been un-cancelled")
   that M2's negation regex doesn't recognize, while `duplicate()`/`refinement()`
   for the same boolean state DO use recognizable negation words ("not", "won't").
   This made an honest write correctly reflecting the post-update state look
   fact-incompatible with its own stored value. Fixed by rewording `update()`/
   `false_update()` to `"is no longer shipping this quarter; the plan was
   cancelled"` (True->False: "no longer" is both a supersession marker AND a
   real single-negative, so `negated=True` correctly) and `"has been reinstated
   and will ship this quarter after all"` (False->True: no negation word,
   `negated=False` correctly, avoiding a double-negative like "no longer
   cancelled" which would have flipped the polarity wrong). Added "reinstated"
   to `STRONG_MARKER_PATTERN`; removed the now-unmatched "un-cancelled" /
   literal-cancellation-sentence entries.

2. **`_looks_like_refinement` used whole-sentence length**, which breaks the
   moment a stored value is itself a long update-marker sentence (frequently
   longer than a plain refinement). Replaced with `REFINEMENT_DETAIL_PATTERN`,
   a `",\s*(pending|including|across|before|per|at|in|with|behind|for)\b"`
   check -- every refinement template in `warden/domains.py` introduces its
   added detail with exactly one of these words after a comma; duplicates
   never do. Also reworded the three `false_refinement()` methods (which
   previously used unmatched connector words like "moved"/"revised"/"though")
   to use "with ... noted/cited in an earlier draft", so a false refinement's
   fabricated detail is recognized as a refinement by the same pattern instead
   of silently falling back to "duplicate".

Measured effect on an 800-write seeded run: four-class accuracy 62.6% -> 91.6%,
worst per-class F1 0.4 -> 0.8, honest-wrongly-blocked 12.8% -> 7.6%.

## D13 — Emergent finding: false updates cause cascading corruption, not just one-time leakage

After D12's fixes, remaining `duplicate`/`refinement` writes misclassified as
`contradiction` were traced to a real, expected phenomenon rather than a bug:
when a `false_update` (fabricated content, dressed in legitimate supersession
language) is accepted by the gate, M5's ground truth for that key does NOT
change (by design -- only a genuine, non-false update mutates
`KeyState.attrs`). The next HONEST write is generated from the true,
unchanged ground truth and therefore legitimately conflicts with the now
falsely-updated store -- and gets misclassified as `contradiction` (and, if
the honest write is a duplicate/refinement, possibly wrongly blocked). A
single successful false update doesn't just corrupt one value; it poisons
the well for every subsequent honest write about that key until the truth
happens to be restated as another "update." This is a real property of the
system, not an artifact to patch away -- fixing it would require either
preventing false updates from ever being applied (the exact open problem
this project measures) or leaking ground-truth knowledge into the generator,
which would invalidate the evaluation. Reported as observed, not corrected.

## D14 — Final full-scale results (n=3000 writes, seed=2026, num_keys=60)

Generated via `python -m warden.generate_results`. All numbers are measured,
none tuned to hit a target (see the "do not tune" rule at the top of this
project's spec).

| Metric | Target | No gating | Cosine only | WARDEN | Status |
|---|---|---|---|---|---|
| False facts surviving in store | < 5% | 100.0% | 89.6% | 24.1% | MISSED |
| Honest writes wrongly blocked | < 10% | 0.0% | 4.0% | 6.2% | MET |
| Four-class accuracy | > 74.3% | n/a | 32.4% | 92.3% | MET |
| Worst per-class F1 | > 0.70 | n/a | 0.0 | 0.8 | MET |
| Duplicate / Refinement / Update / Contradiction F1 | report | n/a | .2/.4/.4/.0 | .9/1.0/1.0/.8 | n/a |
| Classification cost p50 / p95 | <10ms / <50ms | n/a | 37.7/114.7 | 0.16/2.70 | MET |
| LLM escalation rate | < 15% | n/a | 0.0% | 23.7% | MISSED |
| Throughput, writes/s | > 100 | 252367 | 17.9 | 1353.5 | MET |
| False facts via correctly-classified write | report | n/a | 12.6% | 24.0% | n/a |

**Two honest misses, both explained, neither papered over:**
- False facts surviving (24.1%, target <5%): dominated by false `update`
  writes (see D10) -- fabricated content dressed in genuine supersession
  language is undetectable by fact-gate OR LLM alike without external
  verification, and D13's cascading-corruption effect compounds it further.
  Still a 76-point improvement over no gating and 65-point improvement over
  cosine-only.
- LLM escalation rate (23.7%, target <15%): every `update`-class write
  (real or false) escalates by design (the ambiguous band is defined as
  "fact mismatch + supersession marker present," and updates are the only
  class that ever has one -- see D10). The rate is essentially bounded below
  by the workload's update-class share (25%) under this safety-conservative
  policy. Lowering it would mean either a riskier fast-path policy or a
  smaller update share in the workload -- neither done, since that would be
  tuning the input to chase the target rather than reporting what a fixed,
  reasonable workload measures.

6 of 8 gated targets MET; both misses are structural properties of the
approach, not measurement noise, and are the most interesting results in
the project.

## D15 — Surviving false facts are almost entirely correctly-classified, not misclassified

`false_facts_surviving` (24.1%) and `false_via_correct_classification` (24.0%)
are nearly equal by direct measurement, not coincidence: of the 144 false
writes that survived under WARDEN, **143 (99.3%) got in through a write the
classifier labeled correctly** (73/73 false refinements via correct `merge`,
70/71 false updates via correct `apply`); only 1 survived via misclassification
(a false update mistaken for a refinement, `merge`d instead of `apply`d).
Contradiction-labeled false writes are caught 100% of the time (0/446 survive).
Breakdown: `results/survival_breakdown.png`, built by `warden/survival_breakdown.py`
from the existing `writes.csv` (no re-run needed). This sharpens D14's point:
the 24.1% miss isn't a classifier accuracy problem at all -- it's the ceiling
of class-based gating itself, since a perfectly classified refinement/update
still gets its normal action with no way to check the *content* is true.

## D16 — M7 agent claims are scripted, not LLM-generated; labeled on the chart

Confirmed on request: `warden/agents.py`'s LangGraph node (`_make_submit_node`)
is a pass-through that forwards a claim already sitting in state to
`gateway.submit()` — it does not generate that claim. The 5 claims in
`run_demo()`'s `turns` list are hardcoded strings. Repo-wide grep confirms no
module calls the real Anthropic API anywhere (no `ANTHROPIC_API_KEY`, per D7);
M3's escalation path (which the agent demo's 4th turn does exercise, `escalated=True`)
uses `StubOracle`, not a live LLM. `scene_agents.png`/`.txt` now say this
plainly in the chart title itself. Doesn't change any metric — M7 already
never feeds results/ (see M7's module docstring and `test_no_results_module_imports_agents`).

## D17 — M5's template corpus was trivially separable; two published numbers are invalid

Found while considering whether to swap `StubOracle` for a real LLM. Two defects:

**(a) The four classes were perfectly separable by two regexes.** Measured over
the 3000-write corpus: a supersession marker appeared in 100.0% of `update`
writes and 0.0% of every other class; a trailing ", <preposition>" clause
appeared in 100.0% of `refinement` and 0.0% of everything else. So the task
reduced to two `re.search` calls plus one fact-comparison to split duplicate
from contradiction — which is exactly why those two were the only imperfect
classes (F1 0.9 / 0.8). Worse, 10 of the 13 phrases in `classifier.py`'s
`STRONG_MARKER_PATTERN` appear **verbatim** in `domains.py`'s own templates, and
`REFINEMENT_DETAIL_PATTERN`'s word list is exactly the set of leading words from
the refinement templates. M3 was written against M5's vocabulary.

**(b) 29 strings were simultaneously true and fabricated.** `false_update()`
drew from the same marker pool as `update()`, so identical sentences carried
opposite `is_false` labels (e.g. "The vendor contract renewal has shifted to
Sunday.").

**Consequence:** the 92.3% four-class accuracy and the 24.1% false-facts-surviving
figure in D14 do not support claims about the *approach* — only about this
corpus. 24.1% is not "the honest limit of class-based gating"; it is the limit
imposed by a corpus that hands identical inputs opposite labels. What survives
unaffected: the no-gating baseline (100%, true for any workload), the cosine-only
ablation (all four classes in a 0.77–0.82 similarity band), and the cost/throughput
measurements.

A consequence worth stating plainly: on this corpus the project **cannot measure
its own thesis**. The central claim is that escalating the ambiguous band to an
LLM beats fact-gating alone — but all 724 escalated writes are `update` (0 of 446
contradictions ever escalate), so `StubOracle`'s unconditional "update" scores
100% there and a real LLM is indistinguishable from it.

## D18 — Fix: label-first LLM rendering, and the RNG separation it forced

M5 was always label-first (the seeded RNG picks `true_class`, `is_false` and the
fact values, *then* renders). The fix isolates only that last step:

- `warden/renderers.py` — `RenderRequest` + `Renderer` protocol; `TemplateRenderer`
  reproduces the original `domain.<class>(...)` calls.
- `warden/llm_workload.py` — `LLMRenderer`: per-class prompts, on-disk cache
  (`data/llm_workload_cache.json`) so a seed reproduces byte-identically without
  re-calling the API, injectable client so plumbing is testable without a key.
- RNG-chosen **surface styles** (`STYLES`) are the actual mechanism that breaks
  (a): ~45% of contradictions are styled `assertive_change`/`corrective` (they DO
  wear change-language), and ~33% of genuine updates are styled `terse_restatement`
  (they do NOT). Honest and false variants share a style pool, so style never
  leaks `is_false`. Guarded by `test_separability_guard_*` — the regression test
  that would have caught D17.

**Forced change — published results no longer reproduce.** A test asserting
ground truth is renderer-independent failed: `TemplateRenderer` consumes the main
RNG (`rng.choice(forms)`) while `LLMRenderer` does not, so the renderer choice was
shifting which *key* each write targeted — making the two corpora incomparable.
Fixed by giving surface realization its own per-write stream (`render_rng(seed,
write_id)`), leaving the main stream to own only labels and structure. Because the
old corpus was only reproducible through that entanglement, removing it changes the
main stream's downstream draws: seed 2026 now yields a different (equally valid —
distribution and false-fraction still within tolerance) label sequence than the one
in `results/`. **`results/` is therefore stale and not currently regenerable.**
Deliberately NOT regenerated yet: doing so would just re-publish numbers from a
differently-seeded but equally flawed template corpus. Regenerate once, after the
LLM corpus exists.

**Still open:** the fix is inert without `ANTHROPIC_API_KEY`. Template rendering
remains the default and remains 100%/0% separable — verified after the refactor.

## D19 — Configs wired to either corpus; silent fallback made impossible

`run_all_configs(..., corpus=)` selects the M5 corpus, and
`python -m warden.generate_results --corpus {template,llm}` drives the whole
results build. Design constraints:

- **No silent fallback.** `build_renderer()` raises on an unknown corpus rather
  than defaulting, and `LLMRenderer` raises when the key is missing and the cache
  can't serve a request. Publishing template numbers under an LLM label would
  invalidate the comparison while looking completely normal in every output file
  — it is the one failure mode with no visible symptom, so it fails loudly
  instead. Guarded by `test_llm_corpus_without_key_or_cache_fails_loudly_not_silently`.
- **Provenance without touching the frozen schema.** The 12-column log schema is
  frozen, so corpus identity goes in `results/corpus_provenance.json` (corpus,
  SHA-256 fingerprint of all generated text, seed, sizes, unique-text count,
  timestamp) plus a subtitle on `metrics_table.png`. No 13th column.
- **Scene demo provenance bug fixed.** `write_scene_demo()` re-generated the
  workload with hardcoded template defaults, so an LLM-corpus run would have
  shown template sentences beside LLM metrics. It now takes the same corpus/seed
  /sizes as the run it illustrates.

`results/` regenerated at full scale on the template corpus so the repo is
self-consistent and reproducible under current code; `metrics_table.png` now
states "M5 corpus: TEMPLATE (regex-separable — see D17)" so the numbers can't be
mistaken for a clean result. Re-running with `--corpus llm` is a one-command
swap once a key exists.

Baseline for future diffs — template corpus, seed 2026, fingerprint `4e61909e`:
false-facts-surviving 25.5%, honest-wrongly-blocked 3.7%, four-class accuracy
95.9%, worst F1 0.9, escalation 23.7%, throughput 1273/s. (Shifted slightly from
D14's numbers because D18's RNG separation changed the label draw for this seed;
notably it got *more* flattering, consistent with D17.)

## D20 — Workload generation supports non-Anthropic providers, and that is preferred

M5 generation is a low-difficulty instruction-following task ("write one realistic
business sentence in this style"), so a free tier is adequate. More importantly, a
**non-Claude generator is methodologically preferable**: it removes the mild
circularity of one model family both writing the corpus and adjudicating M3
escalations. This is a weaker concern than D17's regex coupling, but it is free to
avoid, so it is avoided.

Implemented as a single OpenAI-compatible `/chat/completions` path, which covers
Groq, Gemini, OpenRouter, Mistral, Together and local Ollama. Uses `httpx`
(already a transitive dependency of `anthropic`) rather than adding an SDK.
Selected via `WARDEN_LLM_PROVIDER` / `WARDEN_LLM_MODEL` / the provider's key var,
all loadable from an untracked `.env`.

Two bugs caught by the tests written alongside it:
- **Cache namespacing.** Cache keys are now prefixed with the model id. Without
  this, switching provider mid-corpus would silently replay another model's
  sentences under the new label — the same invisible-provenance failure class as
  D19.
- **Offline replay.** `model_id` originally resolved through `self.client`, which
  built a client and demanded a key even on a pure cache hit, breaking the
  "a cached corpus replays with no key" guarantee that caching exists for. It now
  resolves the model string from config without constructing a client.

Note for the eventual M3 swap: if the *escalation* oracle also moves off the stub,
the model used there should be recorded separately — generator and adjudicator are
different roles and should not be assumed to be the same model.

## D21 — First LLM pilot was 26.5% contaminated; output validation added

Ran a 200-write pilot on Groq (`openai/gpt-oss-120b`). Four distinct failures,
all found by inspecting the corpus rather than by anything failing loudly:

1. **Stale model default.** `llama-3.3-70b-versatile` no longer exists on Groq —
   every call 404'd. Provider catalogues rot; the 404 path now tells the reader to
   list `GET <base_url>/models` and set `WARDEN_LLM_MODEL`.
2. **Rate-limit crash discarded all paid-for work.** The cache flushed only after
   the full run, so the first 429 threw away every sentence already generated.
   Now: `Retry-After`-aware backoff, and a checkpoint flush every 25 calls.
3. **26.5% of the corpus was EMPTY STRING (53/200), plus 1 refusal.** Root cause:
   `gpt-oss-120b` is a *reasoning* model. With `max_tokens=150` it spent the whole
   budget on internal reasoning (`reasoning_tokens: 148`), hit
   `finish_reason: length`, and returned empty `content`. Fixed with
   `reasoning_effort: "low"` (36 reasoning tokens instead of 179) plus
   `max_tokens=512`, with automatic fallback if a provider rejects the parameter.
4. **Nothing rejected any of it.** An empty string is a perfectly well-formed
   row in `writes.csv`. It would have produced real-looking metrics with no
   visible symptom — the same invisible-provenance failure class as D19/D20, and
   the third time it has appeared in this project. 29% of *false* writes were
   affected, which would have hit the headline metric hardest.

`validate_text()` now rejects empty, refusal (smart-quote-aware — the first
refusal regex missed `I’m sorry`) and over-long output; `LLMRenderer` retries
3× and then raises rather than caching anything unusable. Guarded by
`test_unusable_output_is_never_cached`. Contaminated entries were purged from the
cache (49 removed, 149 kept) rather than regenerating from scratch.

## D23 — Removed a fabricated cost number; cost now reported per tier

Raised by the user: the architecture must not be blurred to make targets pass.
Three changes, all in that direction.

**1. `StubOracle` no longer sleeps 2ms.** It previously did
`time.sleep(0.002)  # simulate real LLM call latency`. That was a fabricated
number whose only effect was to make two targets pass for a hardcoded string:

| Metric | With the fake sleep | Truth |
|---|---|---|
| Classification cost p95 | 2.76 ms — MET | stub costs ~0; a real oracle costs ~700 ms |
| Throughput | 1273/s — MET | ~5–15/s with a real oracle |

This violated the project's own rule ("Do not tune any threshold to hit a
target"). The stub now reports its real cost (~0), and the real oracle reports
its real latency (measured 650–917 ms on qwen/qwen3.8-27b).

**2. Cost is now also reported split by tier**, via `classify_cost_by_path()`.
A single blended p95 over a deliberately two-tier system describes neither tier
— it is a mix of ~0.1 ms and ~700 ms corresponding to no real operation. The
blended rows stay (nothing is hidden); the split rows show whether the cheap
tier is actually carrying the traffic it was designed for. When the blended p95
misses its target, the split makes the reason legible: the escalation *rate* is
the thing to optimise, not the fast path.

**3. Oracle is now selectable** — `--oracle {stub,llm}`. `StubOracle` is kept as
an **ablation arm**, not deleted: it represents the real policy "trust the
supersession language". The intended architecture (`LLMOracle`) is the `llm`
arm, adjudicating with `qwen/qwen3.8-27b` — deliberately a different model from
the corpus generator (`openai/gpt-oss-120b`) per D20. The stub→llm gap is the
measurement that isolates the value of escalation, and it is only obtainable by
running both; that is why the stub run happens first rather than being skipped.

`LLMOracle` caches each decision **together with its measured latency**, and
`FastPathEscalator` charges the recorded latency rather than wall-clock. Caching
the decision alone would make p50/p95/throughput collapse to ~0 ms on every
replay — another silently faked metric, the same failure class as D19/D20/D21.

**Also fixed:** `OpenAICompatClient` retried on HTTP status codes but not on
transport errors, so a single connection reset killed a multi-hour corpus run
outright (it did, at ~49%). It now retries `httpx.TransportError` with the same
backoff. The disk checkpoint from D21 meant no generated sentences were lost.

## D24 — Two bugs found by the first honest LLM-corpus run (29.9% → re-measuring)

The first run on the LLM corpus reported 29.9% accuracy and **58.8% of honest
writes blocked**. Both were largely artefacts, found by tracing individual rows
rather than accepting the numbers.

**(a) Negation missed on typographic apostrophes.** `facts.py`'s negation regex
matches ASCII `isn't`/`won't`; LLM text uses U+2019 (`isn’t`, `won’t`). Negation
therefore read as False on LLM-written negative claims, so
`"The offline mode will not ship this quarter"` vs
`"the offline mode isn’t slated to be released this quarter"` — a perfect
duplicate — registered as `negation True != False`, i.e. a fact conflict, and was
blocked. Fixed by folding typographic punctuation to ASCII once at the entry to
`extract()` (`_normalize`), with three regression tests. `raw_text` still
preserves the original.

*This was the second appearance of the same bug.* It hit the refusal detector in
D21, was fixed there, and `facts.py` was never checked for it.

**(b) The evaluation fed the classifier a stored value its label was never
defined against — the dominant cause.** M5 assigns `true_class` relative to *its
own* stored value, but the gate BLOCKS some writes. When it wrongly blocks a
genuine update, the store keeps the old value while M5's ground truth moves on;
from then on, every label for that key is defined against a value the store never
received. Observed consequence:

    label: duplicate
    STORED:   The infra budget is 250.
    INCOMING: The infra budget has been set to 280, replacing the earlier figure of 250.

That is an update, not a duplicate — the label is simply wrong. The effect is
**self-amplifying**: the more the classifier errs, the more writes are blocked,
the faster labels rot, so error compounds instead of averaging out. It stayed
invisible on the template corpus precisely because accuracy there was ~96%, so
almost nothing was wrongly blocked and the store never diverged.

Measured cost of the flaw, same classifier, same corpus:

| Classifier input | Accuracy |
|---|---|
| Whatever the store held (old behaviour) | 29.9% |
| The labelled pair (fixed) | **64.2%** |

Fixed in `harness.py`: classification is done on `e.stored_text`, the pair the
label was defined for. The store still records every action, version and history
entry, and the safety metrics still read from it — only the classifier's *input*
is pinned. This is standard practice for a classification benchmark (a label is a
property of an input pair), but it is a genuine limitation worth stating: a
deployed system would compare against live store contents, and would face the
same divergence without any label to invalidate.

**Surviving genuine finding (not a bug):** even on the labelled pair, `update`
recall is poor — 42/216, with 151 genuine updates read as contradictions. LLM-
written updates often express supersession in words `STRONG_MARKER_PATTERN` has
never seen. That is the real generalization gap D17 predicted, and it remains
after both fixes.

## D21 (cont.) — Test-isolation bug

**Test-isolation bug surfaced by this work:** once `.env` held a real key,
`resolve_client()`'s `load_dotenv()` re-populated variables that tests had just
deleted, so provider tests passed or failed depending on whether the developer had
a key on disk. `tests/test_m5_providers.py` now has an autouse fixture that
neutralizes `load_dotenv` and clears the relevant variables.
