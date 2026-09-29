"""Out-of-sample discipline, enforced rather than documented.

A holdout you can read by accident is not a holdout. Every rule below exists because this
project has already lost a slice: preset A (6xATR stop / 2xATR target) was chosen by looking at
test-slice performance, which spent that slice for that preset and nobody noticed until much
later. See "Known Errors and Corrections" in the vault.

THREE SEGMENTS, DECLARED BEFORE ANY CODE RUNS

    train      form the idea. Fit anything. Look as much as you like.
    validate   compare a SMALL number of pre-declared variants. Every look costs budget.
    holdout    frozen. Reading it is an EVENT, logged with a reason and a timestamp.

`holdout` is not reachable through the ordinary API. `Split.holdout` raises. You must call
`unlock_holdout(reason=...)`, which appends to an append-only ledger on disk. The ledger is the
point: it makes "how many times have we looked?" a question with an answer instead of a vibe.
Once the count is above one, the holdout is no longer a holdout and the report says so.

EMBARGO

Segments are separated by a gap of `embargo` bars, dropped from both sides. Without it, a trade
that opens near the end of train and closes inside validate has its outcome determined by bars
on both sides of the boundary -- the label straddles the split and leaks. Lopez de Prado calls
this purging and embargoing. Set it to at least the longest holding period the strategy allows,
which for us is one RTH session.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np


class HoldoutLocked(Exception):
    """Raised on any attempt to read the holdout without unlocking it first."""


@dataclass
class Split:
    """An immutable, pre-registered partition of a bar series."""

    n: int
    train_end: int
    validate_end: int
    embargo: int = 0
    name: str = "unnamed"
    ledger_path: pathlib.Path | None = None
    _unlocked: bool = field(default=False, repr=False)

    # ---------------------------------------------------------------- constructors
    @classmethod
    def by_fraction(
        cls,
        n: int,
        train: float = 0.50,
        validate: float = 0.25,
        *,
        embargo: int = 0,
        name: str = "unnamed",
        ledger: str | pathlib.Path | None = None,
    ) -> Split:
        assert 0 < train < 1, "train must be a fraction in (0, 1)"
        assert 0 < validate < 1, "validate must be a fraction in (0, 1)"
        assert train + validate < 1, "train + validate must leave room for a holdout"
        return cls(
            n=n,
            train_end=int(n * train),
            validate_end=int(n * (train + validate)),
            embargo=embargo,
            name=name,
            ledger_path=pathlib.Path(ledger) if ledger else None,
        )

    @classmethod
    def by_date(
        cls,
        dates: np.ndarray,
        train_end: str,
        validate_end: str,
        *,
        embargo: int = 0,
        name: str = "unnamed",
        ledger: str | pathlib.Path | None = None,
    ) -> Split:
        """Preferred over by_fraction: a date boundary is a decision you can defend, a
        fraction is one that moves silently when the data file grows."""
        d = np.asarray(dates, dtype="datetime64[ns]")
        return cls(
            n=len(d),
            train_end=int(np.searchsorted(d, np.datetime64(train_end))),
            validate_end=int(np.searchsorted(d, np.datetime64(validate_end))),
            embargo=embargo,
            name=name,
            ledger_path=pathlib.Path(ledger) if ledger else None,
        )

    # ---------------------------------------------------------------- segments
    @property
    def train(self) -> slice:
        return slice(0, max(0, self.train_end - self.embargo))

    @property
    def validate(self) -> slice:
        return slice(self.train_end, max(self.train_end, self.validate_end - self.embargo))

    @property
    def holdout(self) -> slice:
        if not self._unlocked:
            raise HoldoutLocked(
                f"The holdout for split '{self.name}' is frozen. If you genuinely intend to "
                f"spend it, call unlock_holdout(reason=...) -- which is logged and counted. "
                f"If you are tuning, iterating, or 'just checking', use .validate instead."
            )
        return slice(self.validate_end, self.n)

    @property
    def holdout_bounds(self) -> tuple[int, int]:
        """Size and position WITHOUT unlocking -- safe for reporting and plotting."""
        return self.validate_end, self.n

    # ---------------------------------------------------------------- the ledger
    def unlock_holdout(self, reason: str) -> Split:
        if not reason or len(reason) < 20:
            raise ValueError(
                "Give a real reason (>= 20 chars). It goes in the permanent ledger and is the "
                "only record of why this slice was spent."
            )
        self._unlocked = True
        entry = {
            "when": dt.datetime.now().isoformat(timespec="seconds"),
            "split": self.name,
            "reason": reason,
            "bars": [self.validate_end, self.n],
        }
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        return self

    def holdout_looks(self) -> list[dict]:
        """How many times this holdout has been opened, ever. More than one and it is spent."""
        if not self.ledger_path or not self.ledger_path.exists():
            return []
        out = []
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                if e.get("split") == self.name:
                    out.append(e)
        return out

    def is_holdout_spent(self) -> bool:
        return len(self.holdout_looks()) > 1

    # ---------------------------------------------------------------- reporting
    def describe(self, dates: np.ndarray | None = None) -> str:
        def span(s: slice) -> str:
            # An EMPTY segment is a normal state, not an error: it means the data does not
            # reach this boundary. MES hits it -- every built MES bar predates train_end, so
            # validate and holdout are empty until the bars are built. Say so rather than
            # indexing off the end of the array.
            if s.stop <= s.start:
                return "EMPTY - the data does not reach this segment"
            if dates is None:
                return f"bars {s.start}-{s.stop}"
            d = np.asarray(dates)
            lo = min(s.start, len(d) - 1)
            hi = min(s.stop - 1, len(d) - 1)
            return f"{str(d[lo])[:10]} -> {str(d[hi])[:10]}"

        h0, h1 = self.holdout_bounds
        hs = slice(h0, h1)
        looks = len(self.holdout_looks())
        lines = [
            f"Split '{self.name}'  {self.n} bars, embargo {self.embargo}",
            f"  train     {self.train.stop - self.train.start:>7} bars  {span(self.train)}",
            f"  validate  {self.validate.stop - self.validate.start:>7} bars  {span(self.validate)}",
            f"  holdout   {h1 - h0:>7} bars  {span(hs)}   "
            + (f"** SPENT: opened {looks}x **" if looks > 1 else f"(opened {looks}x)"),
        ]
        return "\n".join(lines)


# ===========================================================================================
# Walk-forward structure
# ===========================================================================================
@dataclass
class Fold:
    """One walk-forward fold: optimise on `opt`, then trade `test` without looking further."""

    i: int
    opt: slice
    test: slice

    @property
    def n_opt(self) -> int:
        return self.opt.stop - self.opt.start

    @property
    def n_test(self) -> int:
        return self.test.stop - self.test.start


@dataclass
class WalkForwardPlan:
    """The rolling optimise-then-trade scheme that runs INSIDE the validate segment.

    Three distinct roles, which are routinely conflated and must not be:

      development   (Split.train)     form the idea. Look freely. Nothing here is a result.
      walk-forward  (Split.validate)  re-optimise on a trailing window, trade the next block,
                                      never look ahead. The OOS blocks concatenated are the
                                      first honest performance estimate.
      holdout       (Split.holdout)   frozen. One look, ever, logged.

    `anchored=False` is a rolling window (fixed lookback, both ends move) -- the default,
    because it adapts to regime. `anchored=True` is an expanding window (start fixed, end
    moves), which uses more data but weights the distant past equally with last month.

    The optimisation window for early folds necessarily reaches back into development data.
    That is correct and is what walk-forward means: at that point in simulated time, the
    development period IS the past. What matters is that no TEST block ever overlaps another
    fold's optimisation window, which `folds()` guarantees by construction.
    """

    train_lookback: int
    step: int
    anchored: bool = False
    min_opt: int | None = None

    def folds(self, split: Split) -> list[Fold]:
        """Non-overlapping test blocks tiling the validate segment, each preceded by its own
        optimisation window."""
        v = split.validate
        out: list[Fold] = []
        i = v.start
        k = 0
        floor = self.min_opt or self.train_lookback // 2
        while i < v.stop:
            j = min(i + self.step, v.stop)
            opt_start = 0 if self.anchored else max(0, i - self.train_lookback)
            opt = slice(opt_start, max(opt_start, i - split.embargo))
            if opt.stop - opt.start >= floor:
                out.append(Fold(i=k, opt=opt, test=slice(i, j)))
                k += 1
            i = j
        return out

    def describe(self, split: Split, dates=None) -> str:
        f = self.folds(split)
        if not f:
            return "WalkForwardPlan: no usable folds (validate segment too short)"
        kind = "anchored/expanding" if self.anchored else "rolling"
        lines = [
            f"WalkForwardPlan: {len(f)} folds, {kind}, lookback {self.train_lookback} bars, "
            f"step {self.step} bars, embargo {split.embargo}",
            f"  optimisation windows: {f[0].n_opt} -> {f[-1].n_opt} bars",
            f"  test blocks: {sum(x.n_test for x in f)} bars total, "
            f"{f[0].n_test} per fold (last {f[-1].n_test})",
        ]
        if dates is not None:
            d = np.asarray(dates)
            lines.append(
                f"  spans {str(d[f[0].test.start])[:10]} -> "
                f"{str(d[min(f[-1].test.stop - 1, len(d) - 1)])[:10]}"
            )
        return "\n".join(lines)
