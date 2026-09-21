"""Fact domains used by the M5 workload generator.

Each domain knows how to render a ground-truth attribute value as natural
language in the different rhetorical shapes a write can take: a duplicate
restatement, a refinement (added true detail), a legitimate update
(explicit supersession language, real value change), a flat contradiction
(no supersession language, incompatible value), and the two "false but
correctly shaped" adversarial variants requested for M5: a false update
(supersession language wrapped around a fabricated change) and a false
refinement (added detail that is actually wrong).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


DATE_SUBJECTS = [
    "the launch", "the API migration", "the mobile release", "the security audit",
    "the partner integration", "the vendor contract renewal", "the beta rollout",
    "the data migration", "the Q3 roadmap review", "the onboarding revamp",
]
NUMBER_SUBJECTS = [
    "the team", "the budget", "the headcount plan", "the marketing spend",
    "the infra budget", "the hiring plan", "the support queue", "the storage quota",
]
STATUS_SUBJECTS = [
    "Feature X", "the dark mode rollout", "the SSO integration", "the billing rewrite",
    "the export tool", "the referral program", "the offline mode", "the audit log feature",
]

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

UPDATE_MARKERS = [
    "has been moved to {new}",
    "is now scheduled for {new} instead of {old}",
    "was pushed to {new} after a scope change",
    "has shifted to {new}",
]
CONTRADICTION_TEMPLATES = [
    "{subject} ships on {value}.",
    "{subject} is scheduled for {value}.",
]

NUMBER_UPDATE_MARKERS = [
    "has increased to {new} as of this quarter",
    "was revised to {new} after the latest review",
    "is now {new}, up from {old}",
    "has been cut to {new} following the reforecast",
]


@dataclass
class KeyState:
    key: str
    domain: str
    subject: str
    attrs: dict[str, Any]


class DateDomain:
    name = "ship_date"
    subjects = DATE_SUBJECTS

    def new_attrs(self, rng: random.Random) -> dict:
        return {"day": rng.choice(WEEKDAYS)}

    def other_value(self, rng: random.Random, attrs: dict) -> str:
        choices = [d for d in WEEKDAYS if d != attrs["day"]]
        return rng.choice(choices)

    def base(self, subject: str, attrs: dict) -> str:
        return f"{subject.capitalize()} ships on {attrs['day'].capitalize()}."

    def duplicate(self, subject: str, attrs: dict, rng: random.Random) -> str:
        forms = [
            f"{subject.capitalize()} is shipping on {attrs['day'].capitalize()}.",
            f"We're shipping {subject} on {attrs['day'].capitalize()}.",
            f"{subject.capitalize()} will ship {attrs['day'].capitalize()}.",
        ]
        return rng.choice(forms)

    def refinement(self, subject: str, attrs: dict, rng: random.Random) -> str:
        detail = rng.choice(["at 3pm PT", "in the EU region first", "with a staged rollout", "pending final sign-off"])
        return f"{subject.capitalize()} ships on {attrs['day'].capitalize()}, {detail}."

    def update(self, subject: str, old_attrs: dict, new_attrs: dict, rng: random.Random) -> str:
        marker = rng.choice(UPDATE_MARKERS).format(new=new_attrs["day"].capitalize(), old=old_attrs["day"].capitalize())
        return f"{subject.capitalize()} {marker}."

    def contradiction(self, subject: str, fabricated_attrs: dict, rng: random.Random) -> str:
        tmpl = rng.choice(CONTRADICTION_TEMPLATES)
        return tmpl.format(subject=subject.capitalize(), value=fabricated_attrs["day"].capitalize())

    def false_update(self, subject: str, old_attrs: dict, fabricated_attrs: dict, rng: random.Random) -> str:
        marker = rng.choice(UPDATE_MARKERS).format(
            new=fabricated_attrs["day"].capitalize(), old=old_attrs["day"].capitalize()
        )
        return f"{subject.capitalize()} {marker}."

    def false_refinement(self, subject: str, attrs: dict, fabricated_attrs: dict, rng: random.Random) -> str:
        return f"{subject.capitalize()} ships on {attrs['day'].capitalize()}, with {fabricated_attrs['day'].capitalize()} noted in an earlier draft."


class NumberDomain:
    name = "magnitude"
    subjects = NUMBER_SUBJECTS

    def new_attrs(self, rng: random.Random) -> dict:
        return {"value": rng.choice([8, 12, 20, 35, 50, 75, 100, 150, 250, 400])}

    def other_value(self, rng: random.Random, attrs: dict) -> int:
        delta = rng.choice([-30, -15, -10, 10, 15, 30, 50])
        return max(1, attrs["value"] + delta)

    def base(self, subject: str, attrs: dict) -> str:
        return f"{subject.capitalize()} is {attrs['value']}."

    def duplicate(self, subject: str, attrs: dict, rng: random.Random) -> str:
        forms = [
            f"{subject.capitalize()} stands at {attrs['value']}.",
            f"{subject.capitalize()} is currently {attrs['value']}.",
        ]
        return rng.choice(forms)

    def refinement(self, subject: str, attrs: dict, rng: random.Random) -> str:
        detail = rng.choice(["including contractors", "across two regions", "before the Q4 review", "per the latest headcount report"])
        return f"{subject.capitalize()} is {attrs['value']}, {detail}."

    def update(self, subject: str, old_attrs: dict, new_attrs: dict, rng: random.Random) -> str:
        marker = rng.choice(NUMBER_UPDATE_MARKERS).format(new=new_attrs["value"], old=old_attrs["value"])
        return f"{subject.capitalize()} {marker}."

    def contradiction(self, subject: str, fabricated_attrs: dict, rng: random.Random) -> str:
        return f"{subject.capitalize()} is {fabricated_attrs['value']}."

    def false_update(self, subject: str, old_attrs: dict, fabricated_attrs: dict, rng: random.Random) -> str:
        marker = rng.choice(NUMBER_UPDATE_MARKERS).format(new=fabricated_attrs["value"], old=old_attrs["value"])
        return f"{subject.capitalize()} {marker}."

    def false_refinement(self, subject: str, attrs: dict, fabricated_attrs: dict, rng: random.Random) -> str:
        return f"{subject.capitalize()} is {attrs['value']}, with {fabricated_attrs['value']} cited in an earlier draft."


class StatusDomain:
    name = "status"
    subjects = STATUS_SUBJECTS

    def new_attrs(self, rng: random.Random) -> dict:
        return {"shipping": rng.choice([True, False])}

    def other_value(self, rng: random.Random, attrs: dict) -> bool:
        return not attrs["shipping"]

    def base(self, subject: str, attrs: dict) -> str:
        verb = "will ship this quarter" if attrs["shipping"] else "will not ship this quarter"
        return f"{subject} {verb}."

    def duplicate(self, subject: str, attrs: dict, rng: random.Random) -> str:
        if attrs["shipping"]:
            forms = [f"{subject} is shipping this quarter.", f"{subject} is on track to ship this quarter."]
        else:
            forms = [f"{subject} is not shipping this quarter.", f"{subject} won't ship this quarter."]
        return rng.choice(forms)

    def refinement(self, subject: str, attrs: dict, rng: random.Random) -> str:
        detail = rng.choice(["pending final QA sign-off", "for enterprise customers first", "behind a feature flag"])
        verb = "will ship this quarter" if attrs["shipping"] else "will not ship this quarter"
        return f"{subject} {verb}, {detail}."

    def _update_text(self, subject: str, was_shipping: bool, now_shipping: bool) -> str:
        # Phrasing is deliberately consistent with duplicate()/refinement()'s
        # own negation words ("not"/"no longer" vs none) so a genuine update
        # doesn't read as a false negation mismatch against later honest
        # writes about the same (now current) state. See DECISIONS.md D12.
        if was_shipping and not now_shipping:
            return f"{subject} is no longer shipping this quarter; the plan was cancelled."
        return f"{subject} has been reinstated and will ship this quarter after all."

    def update(self, subject: str, old_attrs: dict, new_attrs: dict, rng: random.Random) -> str:
        return self._update_text(subject, old_attrs["shipping"], new_attrs["shipping"])

    def contradiction(self, subject: str, fabricated_attrs: dict, rng: random.Random) -> str:
        verb = "will ship this quarter" if fabricated_attrs["shipping"] else "will not ship this quarter"
        return f"{subject} {verb}."

    def false_update(self, subject: str, old_attrs: dict, fabricated_attrs: dict, rng: random.Random) -> str:
        return self._update_text(subject, old_attrs["shipping"], fabricated_attrs["shipping"])

    def false_refinement(self, subject: str, attrs: dict, fabricated_attrs: dict, rng: random.Random) -> str:
        verb = "will ship this quarter" if attrs["shipping"] else "will not ship this quarter"
        return f"{subject} {verb}, with an earlier draft suggesting otherwise."


DOMAINS = [DateDomain(), NumberDomain(), StatusDomain()]
