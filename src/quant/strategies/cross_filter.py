"""Momentum filters applied AT the cross. The user's hypothesis, stated precisely.

THE CLAIM

    "if we dont trade chop and trade when we have momentum and for say a ema cross we should
     have on average better entries"

A syllogism: chop makes crosses fail; if momentum is measurable then filtering by it should
raise average entry quality. The logic is sound and the claim is directly measurable -- "better
on average" is exactly a conditional expectation.

WHY THIS IS NOT HYPOTHESIS 7, AND NOT HYPOTHESIS 3

Hypothesis 7 measured the same conditioners on EVERY BAR inside an EMA state and rejected them.
That result does not transfer here, and the reason is structural rather than statistical: at a
cross bar the two averages have just met, so displacement over the preceding N bars is near zero
BY CONSTRUCTION. The conditioner distribution at crosses is a different distribution from the
conditioner distribution over all state-bars, so a null on one is not a null on the other. This
is the same trap hypothesis 3 hit with separation, which had a median of 0.035 ATR at the cross
bar and 0.231 three bars later.

Hypothesis 3 tested ONE filter (separation) as confirmation. This tests seven, including the
three from hypothesis 7 that are nearly collinear with each other, so the collinearity itself
becomes visible at the cross.

CONFIRMATION LAG IS THE POINT, NOT A PARAMETER

    "price starts to move and we get real market participation"

That describes a state AFTER the cross, not at it. So the entry is evaluated at cross + L bars
for L in {0, 3, 6}, the conditioner is measured at that bar from bars <= it, and the cross
direction must still hold -- a long is abandoned if the fast average has already fallen back
under. L = 0 is kept as the control precisely because it should be the degenerate case.

WHAT "BETTER ENTRIES" MEANS HERE

Three outcomes, not one. Forward return answers "is the mean better". MFE answers "does it give
a run", which is the actual question for a 5-15 minute hold. MAE answers "is the drawdown
smaller". A filter that improves entries should move at least one of them, and a filter that
only cuts trade count without moving any is a filter that costs sample and buys nothing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from ..strategy import ema
from .path_quality import BAR_SECONDS, COMPOSITE_WEIGHTS, SIGNED, _z, raw_conditioners

CONDITIONERS = ("disp", "ser", "persist", "rot", "speed", "relvol", "composite")
EMA_PAIRS = ((8, 21), (20, 50))
CONFIRM_LAGS = (0, 3, 6)
LOOKBACK = 20  # fixed, declared: one swept dimension fewer for the family to pay for
OUTCOMES = ("ret", "mfe", "mae")
CALIB_FRAC = 0.20


@dataclasses.dataclass(frozen=True)
class FilterSpec:
    """One filter configuration: which measure, how long after the cross, on which pair."""

    conditioner: str
    lag: int
    fast: int
    slow: int
    outcome: str

    @property
    def label(self) -> str:
        return f"{self.conditioner}/L{self.lag}/ema{self.fast}-{self.slow}/{self.outcome}"

    @property
    def family(self) -> str:
        return self.conditioner


def filter_family() -> list[FilterSpec]:
    """7 conditioners x 3 confirmation lags x 2 EMA pairs x 3 outcomes = 126 configurations."""
    return [
        FilterSpec(c, lag, f, s, o)
        for c in CONDITIONERS
        for lag in CONFIRM_LAGS
        for f, s in EMA_PAIRS
        for o in OUTCOMES
    ]


def confirmed_entries(close: np.ndarray, fast: int, slow: int, lag: int):
    """(entry mask, direction). An entry sits `lag` bars after a cross, if the cross still holds.

    Abandoning the entry when the relationship has already flipped back is what stops this from
    quietly becoming a different strategy: without the check, a long confirmed six bars after an
    upward cross could fire while the pair is already bearish again.
    """
    f, s = ema(close, fast), ema(close, slow)
    above = f > s
    ok = np.isfinite(f) & np.isfinite(s)
    cross_up = above & ~np.r_[False, above[:-1]] & ok
    cross_dn = ~above & np.r_[False, above[:-1]] & ok

    n = len(close)
    entry = np.zeros(n, bool)
    direction = np.zeros(n)
    src_up = np.zeros(n, bool)
    src_dn = np.zeros(n, bool)
    if lag:
        src_up[lag:] = cross_up[:-lag]
        src_dn[lag:] = cross_dn[:-lag]
    else:
        src_up, src_dn = cross_up.copy(), cross_dn.copy()

    hold_up = src_up & above & ok  # the cross must still be in force at the entry bar
    hold_dn = src_dn & ~above & ok
    entry[hold_up | hold_dn] = True
    direction[hold_up] = 1.0
    direction[hold_dn] = -1.0
    return entry, direction


def build_masks(
    df: pd.DataFrame,
    atr: np.ndarray,
    specs: list[FilterSpec],
    tf: str,
    calib: slice,
    eligible: np.ndarray,
):
    """(direction per (pair, lag), one (high-filter, low-filter) entry pair per spec).

    Terciles come from the calibration slice, computed over CONFIRMED ENTRIES only -- the
    conditioner's distribution at an entry is not its distribution over all bars, and using the
    latter would put nearly every entry in one bucket.
    """
    c = df["close"].to_numpy(dtype=float)
    raw = raw_conditioners(df, atr, LOOKBACK, BAR_SECONDS[tf], calib)

    ent: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for f, s, lag in {(sp.fast, sp.slow, sp.lag) for sp in specs}:
        ent[(f, s, lag)] = confirmed_entries(c, f, s, lag)

    masks: list[tuple[np.ndarray, np.ndarray]] = []
    for sp in specs:
        e, d = ent[(sp.fast, sp.slow, sp.lag)]
        if sp.conditioner == "composite":
            val = (
                COMPOSITE_WEIGHTS["disp"] * _z(raw["disp"] * d, calib)
                + COMPOSITE_WEIGHTS["ser"] * (raw["ser"] * d)
                + COMPOSITE_WEIGHTS["speed"] * (raw["speed"] * d)
                + COMPOSITE_WEIGHTS["persist"] * raw["persist"]
                + COMPOSITE_WEIGHTS["relvol"] * _z(raw["relvol"], calib)
            )
        else:
            val = raw[sp.conditioner]
            if sp.conditioner in SIGNED:
                val = val * d
        pool = val[calib][e[calib] & np.isfinite(val[calib])]
        if len(pool) < 150:
            empty = np.zeros(len(c), bool)
            masks.append((empty, empty))
            continue
        lo_t, hi_t = np.quantile(pool, [1 / 3, 2 / 3])
        base = e & eligible & np.isfinite(val)
        # HIGH filter first, so a positive t means "the filter improved the entry".
        masks.append((base & (val >= hi_t), base & (val <= lo_t)))
    return ent, masks
