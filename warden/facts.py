"""M2: fact extraction.

Pulls structured, comparable fields out of a short natural-language claim:
day names / resolved dates, numeric magnitudes (digits and number-words
unified), and negation. Two claims that *sound* alike (high cosine
similarity) can still disagree on these fields -- that disagreement is
what the M3 classifier's fact gate checks before trusting embedding
similarity alone (see the rule-5 proof in tests/test_m2_facts.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

WEEKDAYS = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
]
RELATIVE_DAYS = {"today": 0, "tomorrow": 1, "yesterday": -1}

_DAY_PATTERN = re.compile(
    r"\b(?P<modifier>next|last|this)?\s*"
    r"(?P<day>" + "|".join(WEEKDAYS + list(RELATIVE_DAYS)) + r")\b",
    re.IGNORECASE,
)

ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {
    "hundred": 100, "thousand": 1_000, "million": 1_000_000,
    "billion": 1_000_000_000, "trillion": 1_000_000_000_000,
}
_NUMBER_WORD_VOCAB = set(ONES) | set(TENS) | set(SCALES) | {"and"}

_DIGIT_NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")

_NEGATION_PATTERN = re.compile(
    r"\b(not|never|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|"
    r"wasn't|weren't|shan't|shouldn't|wouldn't|couldn't|without|no longer)\b",
    re.IGNORECASE,
)

_WORD_TOKEN = re.compile(r"[A-Za-z']+")


@dataclass
class Facts:
    day_tokens: list[str] = field(default_factory=list)   # raw tokens, e.g. ["next friday"]
    date: Optional[date] = None                            # resolved date of the FIRST day token found
    numbers: list[float] = field(default_factory=list)     # digits and number-words, unified
    negated: bool = False
    raw_text: str = ""

    def is_empty(self) -> bool:
        return not self.day_tokens and self.date is None and not self.numbers and not self.negated


def _resolve_date(day: str, modifier: Optional[str], reference: date) -> date:
    day = day.lower()
    if day in RELATIVE_DAYS:
        return reference + timedelta(days=RELATIVE_DAYS[day])

    target_wd = WEEKDAYS.index(day)
    ref_wd = reference.weekday()

    if modifier == "last":
        delta_back = (ref_wd - target_wd) % 7
        if delta_back == 0:
            delta_back = 7
        return reference - timedelta(days=delta_back)

    delta_forward = (target_wd - ref_wd) % 7  # 0 means "today is that weekday"
    bare = reference + timedelta(days=delta_forward)
    if modifier == "next":
        return bare + timedelta(days=7)
    return bare


def _extract_digit_numbers(text: str) -> list[float]:
    out = []
    for match in _DIGIT_NUMBER_PATTERN.finditer(text):
        raw = match.group().replace("$", "").replace(",", "").replace("%", "")
        if raw in ("", "-", "+"):
            continue
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def _words_to_number(tokens: list[str]) -> Optional[float]:
    current = 0
    result = 0
    found = False
    for word in tokens:
        word = word.lower()
        if word == "and":
            continue
        if word in ONES:
            current += ONES[word]
            found = True
        elif word in TENS:
            current += TENS[word]
            found = True
        elif word == "hundred":
            current = (current or 1) * 100
            found = True
        elif word in ("thousand", "million", "billion", "trillion"):
            scale = SCALES[word]
            current = (current or 1) * scale
            result += current
            current = 0
            found = True
    result += current
    return float(result) if found else None


def _extract_number_words(text: str) -> list[float]:
    tokens = _WORD_TOKEN.findall(text.lower())
    numbers = []
    run: list[str] = []

    def flush():
        if run:
            val = _words_to_number(run)
            if val is not None:
                numbers.append(val)
            run.clear()

    for tok in tokens:
        if tok in _NUMBER_WORD_VOCAB:
            run.append(tok)
        else:
            flush()
    flush()
    return numbers


def _normalize(text: str) -> str:
    """Fold typographic punctuation to ASCII before any regex runs.

    LLM-written text uses U+2019 ("isn't", "won't"), not the ASCII apostrophe
    the negation patterns are written with. Without this, negation silently
    fails to detect on such text: "isn't shipping" reads as NOT negated, so a
    duplicate of a negative claim looks like a fact conflict and gets blocked.
    That one mismatch was blocking ~60% of honest writes (DECISIONS.md D24)."""
    return (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("‑", "-").replace("–", "-").replace("—", "-")
                .replace(" ", " ").replace(" ", " "))


def extract(text: str, reference_date: Optional[date] = None) -> Facts:
    """Extract day names/dates, unified numeric magnitudes, and negation from `text`.

    `reference_date` anchors relative expressions ("friday", "next friday",
    "tomorrow"); it defaults to today. Passing an explicit reference makes
    extraction deterministic for tests.
    """
    if reference_date is None:
        reference_date = date.today()

    raw_text = text
    text = _normalize(text)

    day_tokens: list[str] = []
    resolved_date: Optional[date] = None
    for m in _DAY_PATTERN.finditer(text):
        modifier = m.group("modifier")
        day = m.group("day")
        token = f"{modifier or ''} {day}".strip().lower()
        day_tokens.append(token)
        if resolved_date is None:
            resolved_date = _resolve_date(day, modifier.lower() if modifier else None, reference_date)

    numbers = _extract_digit_numbers(text) + _extract_number_words(text)
    negated = bool(_NEGATION_PATTERN.search(text))

    return Facts(
        day_tokens=day_tokens,
        date=resolved_date,
        numbers=numbers,
        negated=negated,
        raw_text=raw_text,
    )
