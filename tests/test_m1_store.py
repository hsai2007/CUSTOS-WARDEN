import random

from warden.store import Store


def test_write_then_read_round_trips():
    s = Store()
    s.write("k", "v1")
    rec = s.read("k")
    assert rec.value == "v1"
    assert rec.version == 1


def test_version_increments_by_exactly_one_per_write():
    s = Store()
    versions = [s.write("k", f"v{i}") for i in range(10)]
    assert versions == list(range(1, 11))
    assert s.read("k").version == 10


def test_history_retains_every_write_not_just_latest():
    s = Store()
    values = ["a", "b", "c", "d"]
    for v in values:
        s.write("k", v)
    hist = s.history("k")
    assert [r.value for r in hist] == values
    assert [r.version for r in hist] == [1, 2, 3, 4]
    # latest read must not silently drop older versions
    assert s.read("k").value == "d"


def test_reading_missing_key_does_not_crash():
    s = Store()
    assert s.read("nonexistent") is None
    assert s.history("nonexistent") == []


def test_independent_keys_have_independent_histories():
    s = Store()
    s.write("a", 1)
    s.write("b", 100)
    s.write("a", 2)
    assert [r.value for r in s.history("a")] == [1, 2]
    assert [r.value for r in s.history("b")] == [100]


def test_rule5_blind_overwrite_silently_destroys_previous_value_over_100_runs():
    """Rule 5 proof: a naive dict overwrite has no way to recover any value
    but the last one written, over 100 independent randomized runs. Store,
    by contrast, keeps every value that was ever written."""
    rng = random.Random(1234)
    for _ in range(100):
        n_writes = rng.randint(2, 8)
        values = [rng.randint(0, 10**9) for _ in range(n_writes)]

        naive_dict: dict[str, int] = {}
        store = Store()
        for v in values:
            naive_dict["k"] = v  # blind overwrite: destroys whatever was there
            store.write("k", v)

        # The naive dict only ever has ONE slot for "k" - every prior value
        # is gone. We prove destruction by showing the dict cannot distinguish
        # "only ever wrote the last value" from "wrote all of these values".
        alt_naive: dict[str, int] = {}
        alt_naive["k"] = values[-1]
        assert naive_dict == alt_naive  # indistinguishable: history is lost

        # Store never loses this distinction.
        recovered = [r.value for r in store.history("k")]
        assert recovered == values
        assert len(recovered) == n_writes
