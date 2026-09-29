"""Pre-registration. A hypothesis is written down, hashed and dated BEFORE anything is run.

WHY THIS IS CODE AND NOT A CONVENTION

"Write the rule before viewing results" is advice everyone agrees with and nobody follows,
because nothing stops you editing the rule afterwards. The split was made real by making the
holdout raise; this does the same for the hypothesis. A `Hypothesis` cannot be constructed
without every field, it is written to an append-only registry with a content hash, and any
later edit produces a different hash and shows up as a separate entry. What you claimed to
predict, and when, becomes a matter of record rather than memory.

THE NINE FIELDS

Every one is mandatory because leaving any of them open is how a vague idea passes for a
testable claim. From the research-sources note:

    "RSI below 30 looks bullish" is not a hypothesis; it has no market, no timeframe, no
    execution logic and nothing falsifiable.

    instrument    which contract, and which timeframe
    session       exact window WITH timezone -- "the open" is not a session
    setup         the entry condition, precise enough to code from this text alone
    direction     long, short, or both -- decided in advance, not discovered
    exit          stop, target, time limit. Definitive, because outcomes become labels
    execution     which price fills, and on which bar. Entry-at-close and entry-at-next-open
                  are different strategies
    costs         the CostModel that applies, including contract count
    invalidation  what observation would make you abandon this. Written BEFORE the result
    metric        what "works" means, numerically, and on which segment

`invalidation` is the field people skip and the one that does the most work. A hypothesis you
cannot lose is not a hypothesis, and deciding the failure condition after seeing the number is
how every disappointing result becomes "promising, needs more work".

WHAT THIS DELIBERATELY DOES NOT DO

It does not judge whether the idea is good. It records what was claimed, so the claim can be
held to. See the validation ladder for what happens next.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import pathlib


class PreRegistrationError(Exception):
    """Raised when a hypothesis is incomplete or an existing registration is contradicted."""


@dataclasses.dataclass(frozen=True)
class Hypothesis:
    """A falsifiable claim about a market, complete enough to code from."""

    name: str
    instrument: str  # e.g. "MES 10m"
    session: str  # e.g. "08:30-15:00 America/Chicago"
    setup: str  # the entry condition, in words, precise enough to implement
    direction: str  # "long" | "short" | "both"
    exit_rule: str  # stop, target, time limit
    execution: str  # which price, which bar
    costs: str  # e.g. "CostModel.for_prop('MES','average',contracts=1)"
    invalidation: str  # what result would make you abandon this
    metric: str  # what "works" means, numerically, and on which segment
    rationale: str = ""  # why this might be true. Optional, but a mechanism beats a pattern
    author: str = ""

    REQUIRED = (
        "name",
        "instrument",
        "session",
        "setup",
        "direction",
        "exit_rule",
        "execution",
        "costs",
        "invalidation",
        "metric",
    )

    def __post_init__(self) -> None:
        missing = [f for f in self.REQUIRED if not str(getattr(self, f)).strip()]
        if missing:
            raise PreRegistrationError(
                f"incomplete hypothesis, missing: {', '.join(missing)}. Every field is "
                f"mandatory -- leaving one open is how a vague idea passes for a testable claim."
            )
        if self.direction not in ("long", "short", "both"):
            raise PreRegistrationError(f"direction must be long/short/both, got {self.direction!r}")
        for f, least in (("setup", 40), ("invalidation", 30), ("metric", 20)):
            if len(str(getattr(self, f)).strip()) < least:
                raise PreRegistrationError(
                    f"'{f}' is too short to be meaningful ({len(str(getattr(self, f)))} chars, "
                    f"need {least}). Write what you would need to hand someone else."
                )

    @property
    def hash(self) -> str:
        """Content hash. Any edit to any field changes it, so a silent rewrite is impossible."""
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def register(self, path: str | pathlib.Path) -> str:
        """Append to the registry. Returns the hash. Re-registering an identical hypothesis is
        a no-op; registering a CHANGED one under the same name is recorded as a new entry with
        a new hash, sitting visibly next to the old one."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        prior = self.registry(p)
        same_name = [e for e in prior if e["name"] == self.name]
        if any(e["hash"] == self.hash for e in same_name):
            return self.hash
        entry = dict(
            hash=self.hash,
            registered=dt.datetime.now().isoformat(timespec="seconds"),
            revision_of=[e["hash"] for e in same_name] or None,
            **dataclasses.asdict(self),
        )
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return self.hash

    @staticmethod
    def registry(path: str | pathlib.Path) -> list[dict]:
        p = pathlib.Path(path)
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    @classmethod
    def revisions(cls, path: str | pathlib.Path, name: str) -> list[dict]:
        """Every version of a named hypothesis, oldest first. More than one is not misconduct --
        it is a record that the claim moved, which is exactly what should be visible."""
        return [e for e in cls.registry(path) if e["name"] == name]

    def describe(self) -> str:
        w = max(len(f) for f in self.REQUIRED)
        lines = [f"HYPOTHESIS  {self.name}   [{self.hash}]"]
        for f in self.REQUIRED:
            if f == "name":
                continue
            lines.append(f"  {f:<{w}}  {getattr(self, f)}")
        if self.rationale:
            lines.append(f"  {'rationale':<{w}}  {self.rationale}")
        return "\n".join(lines)
