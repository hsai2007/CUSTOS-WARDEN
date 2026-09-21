"""M5 label-first LLM rendering.

The seeded RNG still owns ground truth -- `true_class`, `is_false` and the
fact values are decided before anything here runs, and this module only
turns an already-labelled `RenderRequest` into a sentence. The model never
learns which label it is realizing beyond what the phrasing instruction
implies, and nothing it returns feeds back into the labels.

Why this exists: the template corpus made the four classes perfectly
separable by two regexes (supersession marker => update, 100%/0%; trailing
comma clause => refinement, 100%/0%), and made 29 strings simultaneously
true and fabricated. See DECISIONS.md D17.

Generated text is cached on disk by request, so a given seed reproduces a
byte-identical workload without re-calling the API.
"""
from __future__ import annotations

import random
from typing import Optional

from warden.llm_client import (
    CACHE_PATH,
    KNOWN_PROVIDERS,
    MODEL,
    PROJECT_ROOT,
    UnusableResponse,
    _Cache,
    load_dotenv,
    resolve_client,
    resolve_model_id,
    validate_text,
)
from warden.renderers import RenderRequest

SYSTEM = (
    "You write short, realistic internal-company status claims for a synthetic "
    "test corpus used to evaluate a memory-conflict detector. "
    "Output EXACTLY ONE sentence. No preamble, no quotation marks, no explanation, "
    "no hedging words like 'allegedly' or 'reportedly'. Write it the way a busy "
    "colleague would drop it into a channel."
)

STYLE_HINT = {
    "terse": "Keep it short and plain.",
    "verbose": "Use a slightly fuller, more conversational phrasing.",
    "reordered": "Lead with a different part of the sentence than the stored claim does.",
    "trailing_clause": "Attach the extra detail as a trailing clause after a comma.",
    "embedded": "Weave the extra detail into the middle of the sentence, NOT as a trailing comma clause.",
    "second_sentence": "Put the extra detail in its own short second clause joined by a semicolon or dash.",
    "explicit_supersession": "Explicitly signal that this supersedes the previous value.",
    "with_reason": "State the change and give a brief concrete reason for it.",
    "terse_restatement": (
        "State the new value plainly as the current fact. Do NOT use words like "
        "'moved', 'pushed', 'revised', 'updated', 'now', 'previously', or 'instead'."
    ),
    "flat": "State it plainly as a fact. Make no reference to anything having changed.",
    "assertive_change": (
        "Phrase it with confident change-language -- 'now', 'has moved to', 'updated to' -- "
        "even though no reason or prior state is given."
    ),
    "corrective": "Phrase it as a confident correction of the record, without giving a source or reason.",
    "plain": "State the fact plainly.",
}


def _describe(domain: str, attrs: dict) -> str:
    if domain == "ship_date":
        return f"ships on {str(attrs.get('day', '')).capitalize()}"
    if domain == "magnitude":
        return f"is {attrs.get('value')}"
    if domain == "status":
        return "will ship this quarter" if attrs.get("shipping") else "will not ship this quarter"
    return str(attrs)


def build_prompt(req: RenderRequest) -> str:
    subject = req.subject
    stored = req.stored_text or ""
    old = _describe(req.domain, req.stored_attrs)
    new = _describe(req.domain, req.new_attrs) if req.new_attrs else ""
    hint = STYLE_HINT.get(req.style, "")

    if req.true_class == "bootstrap":
        return f"Write the first claim recording this fact: {subject} {old}. {hint}"

    head = f'A memory store currently holds this claim: "{stored}"\n\n'

    if req.true_class == "duplicate":
        return head + (
            f"Write a new message that restates exactly the same fact ({subject} {old}) "
            f"in different words. Add no new information and change no value. {hint}"
        )

    if req.true_class == "refinement":
        if req.is_false:
            return head + (
                f"Write a new message that keeps the same core fact ({subject} {old}) and adds ONE "
                f"further specific detail. The added detail is wrong -- it refers to {new} -- but the "
                f"writer believes it and states it with the same confidence as the rest. {hint}"
            )
        return head + (
            f"Write a new message that keeps the same core fact ({subject} {old}) and adds ONE "
            f"further specific true detail (a time, a region, a caveat, an owner). "
            f"Do not change the existing value. {hint}"
        )

    if req.true_class == "update":
        if req.is_false:
            return head + (
                f"The writer claims the value changed to: {subject} {new}. In reality nothing changed "
                f"-- they are mistaken, and they have no actual basis for it. Write their claim exactly "
                f"as they would write it: confident and routine, giving no verifiable specifics. {hint}"
            )
        return head + (
            f"The value has genuinely changed: {subject} now {new} (it was {old}). "
            f"Write the message reporting that real change. {hint}"
        )

    return head + (
        f"Write a message asserting a conflicting value: {subject} {new}. This conflicts with the "
        f"stored claim and is not backed by any real change. {hint}"
    )


class LLMRenderer:
    """Renders an already-labelled RenderRequest via an LLM, with an on-disk
    cache so repeated runs of the same seed cost nothing and stay identical.

    `client` is injectable so the plumbing is testable without an API key.
    """

    def __init__(self, client=None, cache: Optional[_Cache] = None,
                 provider: Optional[str] = None, model: Optional[str] = None):
        self._client = client
        self._provider = provider
        self._model = model
        self.cache = cache if cache is not None else _Cache()
        self.calls_made = 0

    @property
    def client(self):
        if self._client is None:
            self._client = resolve_client(self._provider, self._model)
        return self._client

    @property
    def model_id(self) -> str:
        """Namespaces the cache, so switching provider/model never silently
        reuses another model's sentences (DECISIONS.md D20). Must NOT build a
        client -- a fully cached corpus has to replay with no key at all."""
        if self._client is not None:
            return getattr(self._client, "model", "injected")
        return resolve_model_id(self._provider, self._model)

    VALIDATION_ATTEMPTS = 3

    def _call(self, prompt: str) -> str:
        last = None
        for _ in range(self.VALIDATION_ATTEMPTS):
            text = self.client.complete(SYSTEM, prompt)
            self.calls_made += 1
            try:
                return validate_text(text)
            except UnusableResponse as exc:
                last = exc
        raise UnusableResponse(
            f"model returned unusable output {self.VALIDATION_ATTEMPTS}x in a row "
            f"({last}). Refusing to write it into the corpus."
        )

    FLUSH_EVERY = 25

    def render(self, req: RenderRequest, rng: random.Random) -> str:
        key = f"{self.model_id}::{req.cache_key()}"
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        text = self._call(build_prompt(req))
        self.cache.put(key, text)
        # checkpoint as we go: a rate-limit or network failure partway through
        # a long corpus must not throw away everything already paid for
        if self.calls_made % self.FLUSH_EVERY == 0:
            self.cache.flush()
        return text


def main():
    """Generate and cache the LLM corpus for a config. Sequential by
    necessity: a genuine update replaces the stored value, so write N's
    prompt depends on write N-1's generated text for the same key. (Chains
    for different keys are independent and could be parallelized -- not
    done yet, see DECISIONS.md D18.) Re-runs are free: everything is cached."""
    import argparse
    from warden.workload import WorkloadConfig, generate

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-writes", type=int, default=3000)
    ap.add_argument("--num-keys", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    renderer = LLMRenderer()
    before = len(renderer.cache)
    events = generate(
        WorkloadConfig(n_writes=args.n_writes, num_keys=args.num_keys, seed=args.seed),
        renderer=renderer,
    )
    renderer.cache.flush()
    print(f"{len(events)} writes | {renderer.calls_made} API calls | "
          f"cache {before} -> {len(renderer.cache)} entries")
    print(f"cache: {renderer.cache.path}")


if __name__ == "__main__":
    main()
