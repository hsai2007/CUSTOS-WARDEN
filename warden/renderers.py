"""Render boundary for M5.

M5 was always label-first: `workload.generate()` picks `true_class`,
`is_false` and the fact values from the seeded RNG, and only then turns
them into a sentence. This module isolates that last step so the text can
come from templates (the original behaviour) or from an LLM, without
touching the label logic that owns ground truth.

`TemplateRenderer` reproduces the original `domain.<class>(...)` calls
exactly, so an existing seed still yields a byte-identical workload.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Protocol

# Surface styles the RNG picks per write. They exist so the corpus's
# surface form is CONTROLLED and measurable rather than correlated 1:1
# with the class label -- the defect found in the template corpus, where
# a supersession marker appeared in 100% of updates and 0% of everything
# else (see DECISIONS.md D17).
STYLES: dict[str, list[str]] = {
    "duplicate": ["terse", "verbose", "reordered"],
    "refinement": ["trailing_clause", "embedded", "second_sentence"],
    # a third of genuine updates state the change WITHOUT explicit
    # supersession wording, so "has a marker" stops implying "is an update"
    "update": ["explicit_supersession", "with_reason", "terse_restatement"],
    # and nearly half of contradictions DO wear change-language, so
    # "has a marker" stops implying "is not a contradiction"
    "contradiction": ["flat", "assertive_change", "corrective"],
    "bootstrap": ["plain"],
}


@dataclass(frozen=True)
class RenderRequest:
    write_id: int
    domain: str
    subject: str
    true_class: str
    is_false: bool
    stored_text: Optional[str]
    stored_attrs: dict = field(default_factory=dict)
    new_attrs: Optional[dict] = None
    style: str = "plain"

    def cache_key(self) -> str:
        parts = [
            self.domain, self.subject, self.true_class, str(self.is_false),
            self.stored_text or "", repr(sorted(self.stored_attrs.items())),
            repr(sorted(self.new_attrs.items())) if self.new_attrs else "",
            self.style,
        ]
        return "|".join(parts)


class Renderer(Protocol):
    def render(self, req: RenderRequest, rng: random.Random) -> str:
        ...


def render_rng(seed: int, write_id: int) -> random.Random:
    """A per-write RNG stream for surface realization only. Keeping this
    separate from the main workload RNG is what makes ground truth
    renderer-independent: swapping TemplateRenderer for LLMRenderer must not
    shift which key a write targets or which class it is (DECISIONS.md D17)."""
    return random.Random(f"render:{seed}:{write_id}")


def pick_style(true_class: str, write_id: int, seed: int) -> str:
    """Choose a surface style from a stream independent of the main workload
    RNG, so adding styles does not shift the main RNG sequence (and thus
    does not change template-rendered workloads for an existing seed)."""
    styles = STYLES.get(true_class, ["plain"])
    return random.Random(f"style:{seed}:{write_id}").choice(styles)


class TemplateRenderer:
    """The original behaviour: render from warden.domains templates."""

    def __init__(self, domain_by_name: dict):
        self.domain_by_name = domain_by_name

    def render(self, req: RenderRequest, rng: random.Random) -> str:
        domain = self.domain_by_name[req.domain]

        if req.true_class == "bootstrap":
            return domain.base(req.subject, req.stored_attrs)
        if req.true_class == "duplicate":
            return domain.duplicate(req.subject, req.stored_attrs, rng)
        if req.true_class == "refinement":
            if req.is_false:
                return domain.false_refinement(req.subject, req.stored_attrs, req.new_attrs, rng)
            return domain.refinement(req.subject, req.stored_attrs, rng)
        if req.true_class == "update":
            if req.is_false:
                return domain.false_update(req.subject, req.stored_attrs, req.new_attrs, rng)
            return domain.update(req.subject, req.stored_attrs, req.new_attrs, rng)
        return domain.contradiction(req.subject, req.new_attrs, rng)
