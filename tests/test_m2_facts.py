from datetime import date

import pytest

from warden import facts as F


REF = date(2026, 9, 14)  # a Monday


def test_extracts_day_names():
    f = F.extract("Let's meet on Friday.", reference_date=REF)
    assert any("friday" in tok for tok in f.day_tokens)


def test_friday_and_next_friday_produce_different_dates():
    bare = F.extract("The release ships on Friday.", reference_date=REF)
    nxt = F.extract("The release ships on next Friday.", reference_date=REF)
    assert bare.date is not None and nxt.date is not None
    assert bare.date != nxt.date
    assert (nxt.date - bare.date).days == 7


def test_last_day_resolves_to_the_past():
    f = F.extract("We shipped it last Friday.", reference_date=REF)
    assert f.date is not None
    assert f.date < REF


def test_today_tomorrow_yesterday():
    assert F.extract("due today", reference_date=REF).date == REF
    assert F.extract("due tomorrow", reference_date=REF).date == date(2026, 9, 15)
    assert F.extract("due yesterday", reference_date=REF).date == date(2026, 9, 13)


def test_number_words_and_digits_produce_the_same_magnitude():
    words = F.extract("Revenue was four million dollars.")
    digits = F.extract("Revenue was 4,000,000 dollars.")
    assert 4_000_000.0 in words.numbers
    assert 4_000_000.0 in digits.numbers


def test_complex_number_words():
    f = F.extract("We hired twenty three people.")
    assert 23.0 in f.numbers

    f2 = F.extract("The budget is one hundred and five thousand.")
    assert 105_000.0 in f2.numbers


def test_negation_detected():
    positive = F.extract("We ship on Friday.")
    negative = F.extract("We do not ship on Friday.")
    assert positive.negated is False
    assert negative.negated is True


def test_negation_contractions():
    assert F.extract("It isn't ready.").negated is True
    assert F.extract("It won't ship.").negated is True
    assert F.extract("It is ready.").negated is False


def test_empty_fields_when_nothing_extractable():
    f = F.extract("The sky is blue and the grass is green.")
    assert f.is_empty()
    assert f.day_tokens == []
    assert f.date is None
    assert f.numbers == []
    assert f.negated is False


@pytest.mark.slow
def test_rule5_high_cosine_cannot_substitute_for_fact_extraction():
    """Rule 5 proof: two claims can be near-identical in embedding space
    (cosine > 0.90) yet disagree on an extracted fact (the resolved date),
    proving cosine similarity alone cannot be trusted to detect this
    disagreement -- structured fact extraction is required."""
    from warden.embeddings import cosine_sim

    a = "The release ships on Friday."
    b = "The release ships on next Friday."

    sim = cosine_sim(a, b)
    fa = F.extract(a, reference_date=REF)
    fb = F.extract(b, reference_date=REF)

    assert sim > 0.90, f"expected high cosine similarity, got {sim}"
    assert fa.date != fb.date, "dates must differ despite high cosine similarity"


def test_negation_detected_with_typographic_apostrophes():
    """LLM-written text uses U+2019, not ASCII '. Missing this made every
    negated duplicate look like a fact conflict, blocking ~60% of honest
    writes on the LLM corpus (DECISIONS.md D24)."""
    for text in ("It isn’t shipping.", "It won’t ship.",
                 "It doesn’t ship this quarter.", "It can’t ship."):
        assert F.extract(text).negated is True, text
    # and the ASCII forms still work, and positives stay positive
    assert F.extract("It isn't shipping.").negated is True
    assert F.extract("It is shipping.").negated is False
    assert F.extract("It’s shipping.").negated is False


def test_typographic_variants_compare_equal_to_ascii():
    smart = F.extract("The rollout isn’t slated for this quarter — it won’t ship.")
    ascii_ = F.extract("The rollout isn't slated for this quarter - it won't ship.")
    assert smart.negated == ascii_.negated is True
    assert smart.numbers == ascii_.numbers


def test_raw_text_preserves_original_punctuation():
    original = "It isn’t shipping."
    assert F.extract(original).raw_text == original
