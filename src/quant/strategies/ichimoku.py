"""The Ichimoku line family, as an INFORMATION question rather than a strategy.

===============================================================================
WHAT THE PASTED PINE SCRIPT ACTUALLY COMPUTES
===============================================================================

The script (HPotter, "Ichimoku2c") swaps the two cloud names relative to every standard
reference. Written out:

    script SenkouA = middleDonchian(52)                 <- this is standard SENKOU B
    script SenkouB = (Tenkan[26] + Kijun[26]) / 2       <- this is standard SENKOU A

Standard Ichimoku defines Senkou A = (Tenkan + Kijun)/2 and Senkou B = midDonchian(52), both
PLOTTED 26 bars into the future. The script gets to the same picture by a different route: it
plots `SenkouA[26]` (the value from 26 bars ago) at the current bar, and computes its "SenkouB"
from `Tenkan[26]`/`Kijun[26]` directly. Displacing the source backwards and plotting at the
current bar draws the same line as computing at the current bar and displacing the plot
forwards.

Two consequences:

  1. The names are swapped, so "SenkouA" in that script is the 52-period midpoint. That is the
     line the question is about, and it is the SLOWER of the two -- which is the more
     interesting one for a market-structure question anyway.
  2. Every line is CAUSAL. midDonchian(52) displaced back 26 bars uses bars t-77..t-26 and
     nothing after t. The forward displacement people associate with the Ichimoku cloud is a
     drawing convention, not a look-ahead. This was worth checking rather than assuming: a
     forward-shifted plot is exactly what look-ahead looks like on a chart.

Chikou is the one line that is genuinely just momentum: the script plots close with offset -26,
so a "Chikou cross" is close[t] vs close[t-26] -- the sign of the 26-bar change, nothing more.

===============================================================================
WHAT THIS MODULE IS FOR
===============================================================================

It does not define entries, exits or a P&L. It produces LINES and CROSS EVENTS so that the
prior question can be answered first: when price crosses one of these lines, does anything
measurable happen afterwards?

That ordering is deliberate. Four strategies have now been rejected, each one a variation on
trend-following via a moving average, and in every case the exit grid and the cost model did
real work before the answer arrived. A line that carries no directional information cannot be
rescued by an exit rule, so the cheap test comes first and the expensive one only runs if the
cheap one survives.

The family below is enumerated ONCE, in code, before any of it is run. Its size is the number
the permutation test has to charge for, so it is stated rather than grown.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from ..strategy import ema, mid_donchian

# Every mid-Donchian length any configuration needs. Computed once per series and shared, which
# is what makes a 2,400-cell family affordable.
MID_LENGTHS = (5, 9, 13, 20, 26, 40, 52, 60, 80, 120)

# The classical settings, for reference and for the zero-free-parameter reading of the result.
CLASSIC = dict(conversion=9, base=26, span_b=52, displacement=26)


@dataclasses.dataclass(frozen=True)
class LineSpec:
    """One line configuration. `kind` selects the formula, `params` its lengths."""

    kind: str
    params: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"{self.kind}({','.join(str(p) for p in self.params)})"

    @property
    def is_classic(self) -> bool:
        return self in CLASSIC_SPECS


def line_family() -> list[LineSpec]:
    """The pre-declared family. 40 line configurations across five formulas.

    mid      midDonchian(n), undisplaced. Tenkan is mid(9), Kijun is mid(26).
    middisp  midDonchian(n) displaced back d bars -- the script's "SenkouA", standard Senkou B.
    span     (midDonchian(t) + midDonchian(k))/2 displaced back d -- the script's "SenkouB",
             standard Senkou A.
    chikou   close[t] vs close[t-n]: the sign of the n-bar change.
    ema      the benchmark. Plain EMA crosses have already been rejected four times; if an
             Ichimoku line carries information, it has to beat this, not just beat zero.
    """
    out: list[LineSpec] = []
    out += [LineSpec("mid", (n,)) for n in MID_LENGTHS]
    out += [LineSpec("middisp", (n, d)) for n in (26, 52, 80, 120) for d in (13, 26, 52)]
    out += [
        LineSpec("span", (t, k, d))
        for t, k in ((9, 26), (5, 13), (13, 40), (20, 60))
        for d in (13, 26, 52)
    ]
    out += [LineSpec("chikou", (n,)) for n in (13, 26, 52)]
    out += [LineSpec("ema", (n,)) for n in (9, 26, 52)]
    return out


CLASSIC_SPECS = frozenset(
    {
        LineSpec("mid", (9,)),  # Tenkan
        LineSpec("mid", (26,)),  # Kijun
        LineSpec("middisp", (52, 26)),  # script "SenkouA" = standard Senkou B
        LineSpec("span", (9, 26, 26)),  # script "SenkouB" = standard Senkou A
        LineSpec("chikou", (26,)),  # Chikou
    }
)


def _displace(x: np.ndarray, d: int) -> np.ndarray:
    """x shifted d bars later, i.e. bar t carries the value computed at t-d. Strictly causal."""
    if d <= 0:
        return x
    out = np.full_like(x, np.nan)
    out[d:] = x[:-d]
    return out


def mid_bank(df: pd.DataFrame, lengths=MID_LENGTHS) -> dict[int, np.ndarray]:
    """midDonchian for every length the family needs, computed once."""
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    return {n: mid_donchian(h, lo, n) for n in lengths}


def build_line(spec: LineSpec, close: np.ndarray, bank: dict[int, np.ndarray]) -> np.ndarray:
    """The line's value at each bar, using only information available at that bar."""
    p = spec.params
    if spec.kind == "mid":
        return bank[p[0]]
    if spec.kind == "middisp":
        return _displace(bank[p[0]], p[1])
    if spec.kind == "span":
        return _displace((bank[p[0]] + bank[p[1]]) / 2.0, p[2])
    if spec.kind == "chikou":
        return _displace(close, p[0])
    if spec.kind == "ema":
        return ema(close, p[0])
    raise ValueError(f"unknown line kind {spec.kind!r}")


def cross_masks(close: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(up_cross, down_cross) at the bar whose CLOSE completes the cross.

    Both the current and previous bar's line must be finite, so a cross is never manufactured
    out of the warm-up NaNs. The event is known at the close of bar t and at no earlier time.
    """
    above = close > line
    ok = np.isfinite(line) & np.r_[False, np.isfinite(line[:-1])]
    prev = np.r_[False, above[:-1]]
    return (above & ~prev) & ok, (~above & prev) & ok
