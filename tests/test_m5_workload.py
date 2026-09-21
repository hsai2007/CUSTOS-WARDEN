from collections import Counter

import pytest

from warden.workload import WorkloadConfig, generate, dump_human_readable, CLASSES, BOOTSTRAP


def make_config(**overrides):
    defaults = dict(n_writes=1000, num_keys=40, seed=0)
    defaults.update(overrides)
    return WorkloadConfig(**defaults)


def test_every_write_has_a_ground_truth_class_label():
    events = generate(make_config(n_writes=200))
    for e in events:
        assert e.true_class in CLASSES or e.true_class == BOOTSTRAP


def test_class_distribution_is_configurable_and_verified_over_1000_draws():
    dist = {"duplicate": 0.4, "refinement": 0.2, "update": 0.2, "contradiction": 0.2}
    config = make_config(n_writes=1000, class_distribution=dist, false_fraction=0.2)
    events = generate(config)
    sampled = [e for e in events if e.true_class != BOOTSTRAP]
    assert len(sampled) == 1000

    counts = Counter(e.true_class for e in sampled)
    for cls, target in dist.items():
        empirical = counts[cls] / len(sampled)
        assert abs(empirical - target) < 0.05, f"{cls}: empirical={empirical}, target={target}"


def test_false_fraction_is_configurable_and_lands_within_2_percent():
    config = make_config(n_writes=5000, false_fraction=0.30)
    events = generate(config)
    sampled = [e for e in events if e.true_class != BOOTSTRAP]
    empirical = sum(1 for e in sampled if e.is_false) / len(sampled)
    assert abs(empirical - 0.30) < 0.02, f"empirical false fraction {empirical}"


def test_same_seed_identical_different_seed_differs():
    a1 = generate(make_config(seed=42))
    a2 = generate(make_config(seed=42))
    b = generate(make_config(seed=43))

    a1_repr = [(e.key, e.true_class, e.incoming_text) for e in a1]
    a2_repr = [(e.key, e.true_class, e.incoming_text) for e in a2]
    b_repr = [(e.key, e.true_class, e.incoming_text) for e in b]

    assert a1_repr == a2_repr
    assert a1_repr != b_repr


def test_every_update_is_temporally_ordered_not_a_contradiction():
    events = generate(make_config(n_writes=500))
    by_key_ts = {}
    for e in events:
        by_key_ts.setdefault(e.key, []).append(e)

    for e in events:
        if e.true_class == "update" and not e.is_false:
            # a genuine update must be strictly later than the write it supersedes
            assert e.stored_text is not None
            prior = [p for p in by_key_ts[e.key] if p.ts < e.ts]
            assert prior, "genuine update must have a prior write to supersede"
            assert max(p.ts for p in prior) < e.ts


def test_contradiction_is_always_false_by_construction():
    events = generate(make_config(n_writes=1000))
    for e in events:
        if e.true_class == "contradiction":
            assert e.is_false is True


def test_false_fraction_floor_validation_rejects_impossible_config():
    with pytest.raises(ValueError):
        make_config(
            class_distribution={"duplicate": 0.25, "refinement": 0.25, "update": 0.25, "contradiction": 0.25},
            false_fraction=0.1,  # below the 0.25 contradiction floor
        )


def test_rule5_human_readable_dump_of_20_writes_is_inspectable():
    events = generate(make_config(n_writes=200))
    dump = dump_human_readable(events, n=20)
    lines = dump.splitlines()
    assert len(lines) == 22  # header + separator + 20 rows
    for cls in CLASSES:
        pass  # not all classes guaranteed in first 20, just check format below
    for line in lines[2:]:
        assert any(c in line for c in CLASSES)
