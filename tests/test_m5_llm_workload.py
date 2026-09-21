"""M5 label-first LLM rendering tests.

The separability guard here is the regression test for the defect in
DECISIONS.md D17: in the template corpus a supersession marker appeared in
100% of updates and 0% of every other class, so two regexes solved the
whole task. These tests assert the mechanism that breaks that correlation.
"""
import os
from collections import Counter

import pytest

from warden.llm_workload import LLMRenderer, _Cache, build_prompt
from warden.renderers import RenderRequest, STYLES, pick_style
from warden.workload import WorkloadConfig, generate, BOOTSTRAP


class FakeClient:
    """Implements the client interface (`complete`) without a key or network."""

    def __init__(self, model="fake-model"):
        self.prompts = []
        self.model = model

    def complete(self, system, prompt, max_tokens=150):
        self.prompts.append(prompt)
        return f"rendered#{len(self.prompts)}"


@pytest.fixture
def fake_renderer(tmp_path):
    cache = _Cache(tmp_path / "cache.json")
    return LLMRenderer(client=FakeClient(), cache=cache)


def test_ground_truth_is_identical_regardless_of_renderer(fake_renderer):
    """Label-first discipline: swapping the renderer must not move a single
    label, key or ordering -- only the text."""
    config = WorkloadConfig(n_writes=200, num_keys=20, seed=5)
    template_events = generate(config)
    llm_events = generate(config, renderer=fake_renderer)

    assert len(template_events) == len(llm_events)
    for t, l in zip(template_events, llm_events):
        assert (t.write_id, t.key, t.true_class, t.is_false) == (l.write_id, l.key, l.true_class, l.is_false)
    # ...and the text genuinely did change
    assert any(t.incoming_text != l.incoming_text for t, l in zip(template_events, llm_events))


def test_cache_prevents_a_second_api_call(fake_renderer):
    req = RenderRequest(write_id=0, domain="ship_date", subject="the launch",
                        true_class="duplicate", is_false=False, stored_text="It ships Friday.",
                        stored_attrs={"day": "friday"}, style="terse")
    import random
    rng = random.Random(0)

    first = fake_renderer.render(req, rng)
    calls_after_first = fake_renderer.calls_made
    second = fake_renderer.render(req, rng)

    assert first == second
    assert fake_renderer.calls_made == calls_after_first == 1


def test_style_is_carried_into_the_prompt(fake_renderer):
    import random
    req = RenderRequest(write_id=0, domain="ship_date", subject="the launch",
                        true_class="contradiction", is_false=True, stored_text="It ships Friday.",
                        stored_attrs={"day": "friday"}, new_attrs={"day": "monday"},
                        style="assertive_change")
    fake_renderer.render(req, random.Random(0))
    prompt = fake_renderer._client.prompts[0]
    assert "change-language" in prompt
    assert "Monday" in prompt


def test_separability_guard_contradictions_often_wear_change_language():
    """The core fix: 'has a supersession marker' must stop implying 'is an
    update'. A meaningful share of contradictions must be styled to use
    change-language, and a meaningful share of updates must not."""
    config = WorkloadConfig(n_writes=3000, num_keys=60, seed=2026)
    events = [e for e in generate(config) if e.true_class != BOOTSTRAP]

    styles = Counter()
    for e in events:
        styles[(e.true_class, pick_style(e.true_class, e.write_id, config.seed))] += 1

    contra_total = sum(v for (c, _), v in styles.items() if c == "contradiction")
    contra_marked = sum(v for (c, s), v in styles.items()
                        if c == "contradiction" and s in ("assertive_change", "corrective"))
    update_total = sum(v for (c, _), v in styles.items() if c == "update")
    update_unmarked = sum(v for (c, s), v in styles.items()
                          if c == "update" and s == "terse_restatement")

    assert 0.4 < contra_marked / contra_total < 0.9, contra_marked / contra_total
    assert 0.2 < update_unmarked / update_total < 0.5, update_unmarked / update_total


def test_every_class_has_more_than_one_surface_style():
    for cls in ("duplicate", "refinement", "update", "contradiction"):
        assert len(STYLES[cls]) > 1, f"{cls} would be single-styled and thus separable"


def test_false_and_honest_variants_share_a_style_pool():
    """A fabricated update must be able to draw the same styles as a genuine
    one, so style alone never reveals is_false."""
    config = WorkloadConfig(n_writes=2000, num_keys=40, seed=11)
    events = [e for e in generate(config) if e.true_class == "update"]
    honest = {pick_style("update", e.write_id, config.seed) for e in events if not e.is_false}
    false = {pick_style("update", e.write_id, config.seed) for e in events if e.is_false}
    assert false and false <= honest


def test_prompt_never_states_the_label_verbatim():
    """The model shouldn't be handed the class name as a word -- it renders a
    described relationship, not a label."""
    req = RenderRequest(write_id=1, domain="magnitude", subject="the budget",
                        true_class="contradiction", is_false=True, stored_text="The budget is 50.",
                        stored_attrs={"value": 50}, new_attrs={"value": 80}, style="flat")
    prompt = build_prompt(req).lower()
    for label in ("contradiction", "duplicate", "refinement", "true_class", "is_false"):
        assert label not in prompt


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs a real API key")
def test_real_llm_corpus_is_not_regex_separable():
    """The outcome version of the separability guard -- only runs with a key."""
    from warden.classifier import STRONG_MARKER_PATTERN

    config = WorkloadConfig(n_writes=200, num_keys=20, seed=3)
    renderer = LLMRenderer()
    events = [e for e in generate(config, renderer=renderer) if e.true_class != BOOTSTRAP]

    def marker_rate(cls):
        rows = [e for e in events if e.true_class == cls]
        return sum(bool(STRONG_MARKER_PATTERN.search(e.incoming_text)) for e in rows) / len(rows)

    # neither class may sit at a degenerate 0% / 100%
    assert 0.05 < marker_rate("contradiction") < 0.95
    assert 0.05 < marker_rate("update") < 0.95
