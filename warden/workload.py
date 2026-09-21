"""M5: synthetic workload generator with ground-truth class labels.

Each key gets a hidden ground-truth trajectory (owned by this generator,
never exposed to the classifier). Every subsequent write is generated as
one of the four classes relative to that trajectory:

  duplicate    - restates the current true value in different words
  refinement   - current true value plus a real added detail
  update       - the ground truth genuinely changes, with explicit
                 supersession language, at a strictly later synthetic ts
                 than the write it supersedes (never a contradiction)
  contradiction - a flat, unmarked, incompatible value; by construction
                 this is always wrong relative to ground truth

`is_false` is tracked independently of `true_class` (see DECISIONS.md):
every contradiction is false by construction, and a small configurable
share of refinement/update writes are ALSO false -- fabricated content
dressed in legitimate-looking phrasing, so the store can be corrupted
even through a write a perfect classifier would label correctly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from warden.domains import DOMAINS, KeyState
from warden.renderers import RenderRequest, TemplateRenderer, pick_style, render_rng

CLASSES = ("duplicate", "refinement", "update", "contradiction")
BOOTSTRAP = "bootstrap"  # first write for a key; not one of the 4 classified outcomes


@dataclass
class WorkloadConfig:
    n_writes: int = 1000
    num_keys: int = 40
    seed: int = 0
    class_distribution: dict = field(default_factory=lambda: {
        "duplicate": 0.30, "refinement": 0.30, "update": 0.25, "contradiction": 0.15,
    })
    false_fraction: float = 0.20
    # share of the false budget (beyond the contradiction floor) spent on
    # refinement/update instead of contradiction
    false_non_contradiction_share: float = 0.25

    def __post_init__(self):
        total = sum(self.class_distribution.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"class_distribution must sum to 1.0, got {total}")
        for c in CLASSES:
            if c not in self.class_distribution:
                raise ValueError(f"class_distribution missing class {c!r}")
        if not (0.0 <= self.false_fraction <= 1.0):
            raise ValueError("false_fraction must be in [0, 1]")
        contra_share = self.class_distribution["contradiction"]
        if self.false_fraction < contra_share - 1e-9:
            raise ValueError(
                f"false_fraction ({self.false_fraction}) cannot be below the contradiction "
                f"class share ({contra_share}): every contradiction is false by construction, "
                "so that share is a floor on the achievable false_fraction."
            )


@dataclass
class WriteEvent:
    write_id: int
    ts: int
    key: str
    domain: str
    stored_text: Optional[str]
    incoming_text: str
    true_class: str
    is_false: bool
    version_read: int


def _mutate_attr(domain, rng: random.Random, base_attrs: dict) -> dict:
    """Return a copy of base_attrs with its (sole) attribute changed to a
    different value. Every domain in warden.domains carries exactly one
    ground-truth attribute, so this works generically across domains."""
    attr_name = next(iter(base_attrs))
    new_attrs = dict(base_attrs)
    new_attrs[attr_name] = domain.other_value(rng, base_attrs)
    return new_attrs


def _non_contradiction_false_probs(config: WorkloadConfig) -> dict[str, float]:
    dist = config.class_distribution
    contra_share = dist["contradiction"]
    extra_needed = max(0.0, config.false_fraction - contra_share)

    non_contra_classes = [c for c in ("refinement", "update") if dist.get(c, 0) > 0]
    probs = {"refinement": 0.0, "update": 0.0}
    if non_contra_classes and extra_needed > 0:
        mass = sum(dist[c] for c in non_contra_classes)
        for c in non_contra_classes:
            probs[c] = min(1.0, extra_needed * (dist[c] / mass) / dist[c])
    return probs


def _make_keys(config: WorkloadConfig, rng: random.Random) -> dict[str, KeyState]:
    key_states: dict[str, KeyState] = {}
    for i in range(config.num_keys):
        domain = DOMAINS[i % len(DOMAINS)]
        subject = rng.choice(domain.subjects)
        key = f"{domain.name}::{subject.replace(' ', '_')}::{i}"
        key_states[key] = KeyState(key=key, domain=domain.name, subject=subject, attrs=domain.new_attrs(rng))
    return key_states


def generate(config: WorkloadConfig, renderer=None) -> list[WriteEvent]:
    """Generate a labelled workload. `renderer` turns an already-labelled
    RenderRequest into text; it defaults to the original template renderer.
    Ground truth is decided here regardless of renderer -- see DECISIONS.md D17."""
    rng = random.Random(config.seed)
    domain_by_name = {d.name: d for d in DOMAINS}
    if renderer is None:
        renderer = TemplateRenderer(domain_by_name)
    key_states = _make_keys(config, rng)
    false_probs = _non_contradiction_false_probs(config)

    events: list[WriteEvent] = []
    stored_text: dict[str, str] = {}
    stored_version: dict[str, int] = {}
    write_id = 0
    ts = 0

    for key, state in key_states.items():
        text = renderer.render(RenderRequest(
            write_id=write_id, domain=state.domain, subject=state.subject,
            true_class=BOOTSTRAP, is_false=False, stored_text=None,
            stored_attrs=dict(state.attrs), style="plain",
        ), render_rng(config.seed, write_id))
        events.append(WriteEvent(
            write_id=write_id, ts=ts, key=key, domain=state.domain,
            stored_text=None, incoming_text=text, true_class=BOOTSTRAP,
            is_false=False, version_read=0,
        ))
        stored_text[key] = text
        stored_version[key] = 1
        write_id += 1
        ts += 1

    classes = list(config.class_distribution.keys())
    weights = [config.class_distribution[c] for c in classes]
    keys = list(key_states.keys())

    for _ in range(config.n_writes):
        true_class = rng.choices(classes, weights=weights, k=1)[0]
        key = rng.choice(keys)
        state = key_states[key]
        domain = domain_by_name[state.domain]

        # ground truth (label + values) is decided FIRST, in this block;
        # only then is it handed to the renderer to be turned into text.
        new_attrs = None
        if true_class == "duplicate":
            is_false = False

        elif true_class == "refinement":
            is_false = rng.random() < false_probs["refinement"]
            if is_false:
                new_attrs = _mutate_attr(domain, rng, state.attrs)

        elif true_class == "update":
            is_false = rng.random() < false_probs["update"]
            new_attrs = _mutate_attr(domain, rng, state.attrs)

        else:  # contradiction: always false by construction (see module docstring)
            is_false = True
            new_attrs = _mutate_attr(domain, rng, state.attrs)

        text = renderer.render(RenderRequest(
            write_id=write_id, domain=state.domain, subject=state.subject,
            true_class=true_class, is_false=is_false, stored_text=stored_text[key],
            stored_attrs=dict(state.attrs), new_attrs=new_attrs,
            style=pick_style(true_class, write_id, config.seed),
        ), render_rng(config.seed, write_id))

        if true_class == "update" and not is_false:
            state.attrs = new_attrs  # ground truth genuinely, temporally, changes

        events.append(WriteEvent(
            write_id=write_id, ts=ts, key=key, domain=state.domain,
            stored_text=stored_text[key], incoming_text=text, true_class=true_class,
            is_false=is_false, version_read=stored_version[key],
        ))

        if true_class == "update" and not is_false:
            stored_text[key] = text
            stored_version[key] += 1

        write_id += 1
        ts += 1

    return events


def dump_human_readable(events: list, n: int = 20) -> str:
    """Fixed-width, human-readable dump of the first n non-bootstrap events
    with their ground-truth labels, so the labeling can be eyeballed."""
    rows = [e for e in events if e.true_class != BOOTSTRAP][:n]
    header = f"{'id':>4}  {'class':<13} {'false':<6} {'key':<28} text"
    lines = [header, "-" * len(header)]
    for e in rows:
        text = e.incoming_text if len(e.incoming_text) <= 70 else e.incoming_text[:67] + "..."
        lines.append(f"{e.write_id:>4}  {e.true_class:<13} {str(e.is_false):<6} {e.key:<28} {text}")
    return "\n".join(lines)
