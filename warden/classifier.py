"""M3: classifiers.

`FastPathEscalator` is WARDEN's classifier: a deterministic fact-gate fast
path (warden.facts + a supersession-marker regex) that only escalates to
an LLM oracle on the genuinely ambiguous band -- a fact mismatch dressed in
legitimate-sounding supersession language, which the fact gate alone cannot
resolve (see DECISIONS.md D9/D10: in this synthetic corpus a real update and
a fabricated ("false") update use identical phrasing, so no classifier --
stub or real -- can do better than trusting the supersession language here;
that is a deliberately visible limit of class-based gating, not a bug).

`CosineOnlyClassifier` is the ablation used for the "Cosine only" config:
thresholds on raw embedding similarity alone, no fact extraction, no
escalation. It exists to demonstrate rule 5 -- cosine similarity is a poor
discriminator between these four classes (measured overlap: all four
classes have mean pairwise cosine similarity in the same 0.77-0.82 band;
see DECISIONS.md D9).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from warden import facts as F
from warden.embeddings import cosine_sim
from warden.llm_client import (
    PROJECT_ROOT,
    UnusableResponse,
    _Cache,
    resolve_client,
    resolve_model_id,
)
from warden.workload import CLASSES

STRONG_MARKER_PATTERN = re.compile(
    r"(moved to|pushed to|shifted to|now scheduled|increased to|revised to|cut to|"
    r"following the|up from|down from|no longer|instead of|reinstated)",
    re.IGNORECASE,
)

# A trailing ", <preposition/participle> ..." clause is how every refinement
# template in warden.domains introduces its added true (or, for a false
# refinement, fabricated) detail. Duplicate paraphrases never have one.
REFINEMENT_DETAIL_PATTERN = re.compile(
    r",\s*(pending|including|across|before|per|at|in|with|behind|for)\b",
    re.IGNORECASE,
)


@dataclass
class ClassificationResult:
    predicted_class: str
    classify_ms: float
    escalated: bool


def _primary_number(facts: F.Facts) -> float | None:
    return facts.numbers[0] if facts.numbers else None


def _facts_compatible(stored: F.Facts, incoming: F.Facts) -> bool:
    """Two claims are compatible unless they EXPLICITLY assert different
    values for the same fact. Silence on a fact (one side doesn't mention
    a date/number at all) is not a conflict -- only a stated disagreement
    is. This also keeps a merged (concatenated) stored value's incidental
    secondary details from later reading as a false conflict against an
    unrelated write that simply doesn't repeat them."""
    stored_num = _primary_number(stored)
    incoming_num = _primary_number(incoming)

    date_ok = stored.date is None or incoming.date is None or stored.date == incoming.date
    numbers_ok = stored_num is None or incoming_num is None or stored_num == incoming_num
    negation_ok = stored.negated == incoming.negated
    return date_ok and numbers_ok and negation_ok


def _looks_like_refinement(stored: F.Facts, incoming: F.Facts) -> bool:
    if REFINEMENT_DETAIL_PATTERN.search(incoming.raw_text):
        return True
    more_numbers = len(incoming.numbers) > len(stored.numbers)
    more_days = len(incoming.day_tokens) > len(stored.day_tokens)
    return more_numbers or more_days


def _has_supersession_marker(text: str) -> bool:
    return bool(STRONG_MARKER_PATTERN.search(text))


class StubOracle:
    """Deterministic stand-in for a Claude Haiku escalation call (no
    ANTHROPIC_API_KEY in this environment -- see DECISIONS.md D7/D10).
    Decides ONLY from (stored_text, incoming_text); never sees is_false or
    true_class -- doing so would leak ground truth into the classifier and
    invalidate the evaluation.

    In this synthetic corpus, the escalation band is only ever reached via
    genuine or fabricated 'update' writes (contradictions never carry
    supersession language), and both use identical phrasing pools, so
    there is no legitimate textual signal to tell them apart. A truthful
    stub (and a real LLM facing the same inputs) can only trust the
    supersession language and predict 'update'. Per SCOPE, no accuracy
    claim is made for this stubbed path -- only its rate and cost are
    reported.
    """

    def decide(self, stored_text: str, incoming_text: str) -> str:
        # No artificial delay. An earlier version slept 2ms to "simulate LLM
        # latency", which fabricated the cost numbers and made the p95 and
        # throughput targets pass for a hardcoded string. This policy really
        # does cost ~0 -- report that, and let the real oracle report its
        # real ~700ms (DECISIONS.md D23).
        return "update"


ORACLE_SYSTEM = (
    "You adjudicate write conflicts for a shared agent memory store. "
    "You will see the value currently stored and a new incoming claim that "
    "disagrees with it on a fact. Decide which of two things the new claim is. "
    "Answer with EXACTLY ONE WORD: UPDATE or CONTRADICTION."
)

ORACLE_PROMPT = """Currently stored: "{stored}"
Incoming claim:  "{incoming}"

These disagree on a fact. Which is the incoming claim?

UPDATE - it reports a genuine change over time. It acknowledges or supersedes
the previous state, gives a reason or cause for the change, or otherwise reads
as a real revision of a fact that has moved on.

CONTRADICTION - it simply asserts a conflicting value with no legitimate basis
for a change. Confident change-sounding wording alone is NOT enough: a claim
that merely declares a different value, without grounding it in an actual
revision, is a contradiction however assertive it sounds.

Answer with exactly one word: UPDATE or CONTRADICTION."""


class LLMOracle:
    """M3 escalation adjudicator -- the real-LLM replacement for StubOracle
    that the architecture always intended (spec: "Claude Haiku via the
    Anthropic API for escalation").

    Sees ONLY (stored_text, incoming_text). Never sees `true_class` or
    `is_false`: that would leak ground truth into the system under test and
    invalidate the whole evaluation.

    Decisions are cached WITH their measured latency, so a replay reproduces
    the same classification AND the same honest cost numbers. Caching just the
    decision would make p50/p95/throughput collapse to ~0ms on re-runs -- a
    silently faked metric, the failure class of D19/D20/D21.
    """

    CACHE_PATH = PROJECT_ROOT / "data" / "llm_oracle_cache.json"
    VALID = ("update", "contradiction")
    ATTEMPTS = 3

    def __init__(self, client=None, cache=None, provider=None, model=None):
        self._client = client
        self._provider = provider
        self._model = model
        self.cache = cache if cache is not None else _Cache(self.CACHE_PATH)
        self.calls_made = 0

    @property
    def client(self):
        if self._client is None:
            self._client = resolve_client(self._provider, self._model)
        return self._client

    @property
    def model_id(self) -> str:
        if self._client is not None:
            return getattr(self._client, "model", "injected")
        return resolve_model_id(self._provider, self._model)

    @staticmethod
    def _parse(raw: str) -> str:
        word = re.sub(r"[^a-z]", "", (raw or "").strip().lower().split()[0]) if (raw or "").strip() else ""
        if word.startswith("update"):
            return "update"
        if word.startswith("contradiction"):
            return "contradiction"
        raise UnusableResponse(f"oracle returned {raw[:60]!r}, expected UPDATE or CONTRADICTION")

    def decide(self, stored_text: str, incoming_text: str) -> str:
        """Returns 'update' or 'contradiction'. Also records the latency that
        FastPathEscalator should charge for this call (see `last_latency_ms`)."""
        key = f"{self.model_id}::{stored_text}::{incoming_text}"
        hit = self.cache.get(key)
        if hit is not None:
            decision, latency_ms = json.loads(hit)
            self.last_latency_ms = latency_ms
            return decision

        prompt = ORACLE_PROMPT.format(stored=stored_text, incoming=incoming_text)
        last = None
        for _ in range(self.ATTEMPTS):
            start = time.perf_counter()
            # Generous budget: the adjudicator is a reasoning model, and a hard
            # case can spend several hundred tokens thinking before emitting its
            # one-word answer. Too tight a budget returns EMPTY content rather
            # than an error (the D21 failure, here on the product side).
            raw = self.client.complete(ORACLE_SYSTEM, prompt, max_tokens=768)
            latency_ms = (time.perf_counter() - start) * 1000
            self.calls_made += 1
            try:
                decision = self._parse(raw)
            except UnusableResponse as exc:
                last = exc
                continue
            self.last_latency_ms = latency_ms
            self.cache.put(key, json.dumps([decision, latency_ms]))
            if self.calls_made % 25 == 0:
                self.cache.flush()
            return decision
        raise UnusableResponse(f"oracle unusable {self.ATTEMPTS}x ({last})")


class FastPathEscalator:
    """WARDEN's classifier: fact-gate fast path + marker-gated escalation."""

    def __init__(self, oracle: StubOracle | None = None):
        self.oracle = oracle or StubOracle()
        self._escalated = 0
        self._total = 0

    def classify(self, stored_text: str, incoming_text: str) -> ClassificationResult:
        start = time.perf_counter()
        self._total += 1

        stored_facts = F.extract(stored_text)
        incoming_facts = F.extract(incoming_text)

        if _facts_compatible(stored_facts, incoming_facts):
            predicted = "refinement" if _looks_like_refinement(stored_facts, incoming_facts) else "duplicate"
            return ClassificationResult(predicted, (time.perf_counter() - start) * 1000, False)

        if not _has_supersession_marker(incoming_text):
            return ClassificationResult("contradiction", (time.perf_counter() - start) * 1000, False)

        self._escalated += 1
        predicted = self.oracle.decide(stored_text, incoming_text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # A cached oracle decision returns instantly, but the cost of this
        # classification is what the call ACTUALLY took when it was made.
        # Charging wall-clock here would make p50/p95/throughput collapse on
        # every replay -- a silently faked metric.
        recorded = getattr(self.oracle, "last_latency_ms", None)
        if recorded is not None:
            elapsed_ms = max(elapsed_ms, recorded)
        return ClassificationResult(predicted, elapsed_ms, True)

    @property
    def escalation_rate(self) -> float:
        return self._escalated / self._total if self._total else 0.0


class CosineOnlyClassifier:
    """Ablation: cosine similarity thresholds only, no fact extraction, no
    escalation. Thresholds are round numbers spanning the observed
    similarity range, fixed before looking at any downstream accuracy
    metric (see DECISIONS.md D9) -- never tuned to hit a target."""

    DUPLICATE_T = 0.90
    REFINEMENT_T = 0.75
    UPDATE_T = 0.60

    def classify(self, stored_text: str, incoming_text: str) -> ClassificationResult:
        start = time.perf_counter()
        sim = cosine_sim(stored_text, incoming_text)
        if sim >= self.DUPLICATE_T:
            predicted = "duplicate"
        elif sim >= self.REFINEMENT_T:
            predicted = "refinement"
        elif sim >= self.UPDATE_T:
            predicted = "update"
        else:
            predicted = "contradiction"
        return ClassificationResult(predicted, (time.perf_counter() - start) * 1000, False)
