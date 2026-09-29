"""HMA / TEMA / VWAP against EMA: the pair family, as an information question.

WHAT IS AND IS NOT NEW HERE

Two of the three proposed pairs are re-parameterisations of something already rejected. HMA and
TEMA are low-lag smoothers of the same closes an EMA smooths; HMA(21) crossing EMA(50) is
structurally a fast line crossing a slow line, which is the EMA(10)/EMA(29), EMA(8)/EMA(21),
EMA(20)/EMA(50) and KAMA(20)/EMA(50) family. Expecting a different answer from a third smoothing
kernel is the definition of a parameter tweak.

Two things make running it anyway the right call rather than a repeat:

  1. Every one of those four rejections was measured on 5-minute bars. The Ichimoku information
     test (2026-09-24) found that 5m and 10m sit at chance for BOTH symbols while 30m and 60m do
     not (p 0.006 / 0.010 / 0.007 / 0.003, 85-93% of cells positive). So the family has never
     actually been tested where the evidence says to look.
  2. VWAP is genuinely a different object. It is volume-weighted and SESSION-ANCHORED: it resets
     every day and is not a rolling filter at all. It is the first thing tested in this project
     that uses the volume column for anything.

So the honest expectation is that HMA and TEMA behave like the EMA control, and that if anything
separates from it, it is VWAP or the timeframe -- not the smoothing kernel. The EMA/EMA pairs are
included in the family for exactly that comparison.

DEFINITIONS

    WMA(n)   weights 1..n, newest heaviest
    HMA(n)   WMA( 2*WMA(close, n/2) - WMA(close, n), sqrt(n) )      Hull, low lag by design
    TEMA(n)  3*e1 - 3*e2 + e3  where e1=EMA(close,n), e2=EMA(e1,n), e3=EMA(e2,n)
    VWAP     cumulative sum(typical price * volume) / cumulative sum(volume), reset at the start
             of each futures session (17:00 America/Chicago, the same day-roll the permutation
             uses), typical price = (high + low + close) / 3

All are causal: bar i uses bars <= i. VWAP includes bar i's own volume and close, which is
legitimate because the cross is evaluated AT the close of bar i and can only be acted on after.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from ..strategy import ema, tema

FAST_LENGTHS = (9, 14, 21, 34, 55)
SLOW_LENGTHS = (21, 50, 100, 200)
VWAP_SLOW_LENGTHS = (9, 21, 50, 100, 200)
DAY_ROLL_HOURS = 7  # 17:00 America/Chicago starts the next session


def wma(x: np.ndarray, n: int) -> np.ndarray:
    """Linearly weighted moving average, newest bar weighted n, oldest weighted 1."""
    x = np.asarray(x, dtype=float)
    if n <= 1:
        return x.copy()
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        # np.convolve with the reversed kernel is a causal correlation: out[i] uses x[i-n+1..i].
        out[n - 1 :] = np.convolve(x, w[::-1], mode="valid")
    return out


def hma(x: np.ndarray, n: int) -> np.ndarray:
    """Hull moving average. Lower lag than an EMA of the same length, at the cost of overshoot."""
    half, root = max(int(n // 2), 1), max(int(math.sqrt(n)), 1)
    return wma(2.0 * wma(x, half) - wma(x, n), root)


def session_vwap(df: pd.DataFrame, day_roll_hours: int = DAY_ROLL_HOURS) -> np.ndarray:
    """Volume-weighted average price, anchored to each futures session open.

    Anchored, not rolling: the accumulation restarts every session, so this is the only line in
    the family whose value depends on where in the day you are.
    """
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    tp = (h + lo + c) / 3.0

    day = (df["dt"] + pd.Timedelta(hours=day_roll_hours)).dt.strftime("%Y-%m-%d").to_numpy()
    first = np.r_[True, day[1:] != day[:-1]]
    sess_no = np.cumsum(first) - 1
    starts = np.flatnonzero(first)

    cpv = np.cumsum(tp * v)
    cv = np.cumsum(v)
    prev = starts - 1
    off_pv = np.where(prev >= 0, cpv[np.maximum(prev, 0)], 0.0)
    off_v = np.where(prev >= 0, cv[np.maximum(prev, 0)], 0.0)

    num = cpv - off_pv[sess_no]
    den = cv - off_v[sess_no]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


@dataclasses.dataclass(frozen=True)
class PairSpec:
    """One crossover configuration: `fast` crossing `slow`."""

    fast_kind: str  # "hma" | "tema" | "vwap" | "ema"
    fast_n: int  # 0 for vwap, which has no length
    slow_n: int  # always an EMA

    @property
    def label(self) -> str:
        f = "vwap" if self.fast_kind == "vwap" else f"{self.fast_kind}{self.fast_n}"
        return f"{f}/ema{self.slow_n}"

    @property
    def family(self) -> str:
        return self.fast_kind


def pair_family() -> list[PairSpec]:
    """The pre-declared family: 61 crossover configurations.

    EMA/EMA pairs are restricted to fast < slow. The inverted pair is not a separate hypothesis --
    EMA(55) crossing above EMA(21) IS EMA(21) crossing below EMA(55), so it would contribute the
    same |t| twice and inflate the family count without adding a single new test.
    """
    out: list[PairSpec] = []
    for kind in ("hma", "tema"):
        out += [PairSpec(kind, f, s) for f in FAST_LENGTHS for s in SLOW_LENGTHS]
    out += [PairSpec("vwap", 0, s) for s in VWAP_SLOW_LENGTHS]
    out += [PairSpec("ema", f, s) for f in FAST_LENGTHS for s in SLOW_LENGTHS if f < s]
    return out


def build_pair(spec: PairSpec, df: pd.DataFrame, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """(fast line, slow line). `cache` is per-series and holds the shared EMA/VWAP computations."""
    c = df["close"].to_numpy(dtype=float)
    skey = ("ema", spec.slow_n)
    if skey not in cache:
        cache[skey] = ema(c, spec.slow_n)
    slow = cache[skey]
    key = (spec.fast_kind, spec.fast_n)
    if key not in cache:
        if spec.fast_kind == "hma":
            cache[key] = hma(c, spec.fast_n)
        elif spec.fast_kind == "tema":
            cache[key] = tema(c, spec.fast_n)
        elif spec.fast_kind == "ema":
            cache[key] = ema(c, spec.fast_n)
        elif spec.fast_kind == "vwap":
            cache[key] = session_vwap(df)
        else:
            raise ValueError(f"unknown fast kind {spec.fast_kind!r}")
    return cache[key], slow


def cross_masks(fast: np.ndarray, slow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fast crosses above slow, fast crosses below slow), at the bar whose CLOSE completes it.

    Both lines must be finite on this bar and the previous one, so a cross is never manufactured
    out of the warm-up NaNs.
    """
    above = fast > slow
    ok = (
        np.isfinite(fast)
        & np.isfinite(slow)
        & np.r_[False, np.isfinite(fast[:-1]) & np.isfinite(slow[:-1])]
    )
    prev = np.r_[False, above[:-1]]
    return (above & ~prev) & ok, (~above & prev) & ok


def build_features(
    df: pd.DataFrame,
    spec: PairSpec,
    rth: str = "08:30-15:00",
    atr_n: int = 14,
) -> dict:
    """Feature dict compatible with strategy.simulate(), for the gated stage-2 traded run.

    This exists so that whatever survives the information test goes through the SAME ladder
    every other strategy in this project went through -- the 45-cell exit grid, the permutation
    tests, PBO, the declared splits -- rather than getting a bespoke simulator written for it.
    A new simulator per strategy is how a project ends up with results that cannot be compared.
    """
    from ..strategy import true_range

    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    # R-unit is the PREVIOUS bar's ATR. A stop sized from the current bar's own range is
    # look-ahead, and is the convention used everywhere else here.
    atr = (
        pd.Series(true_range(h, l, c)).ewm(alpha=1 / atr_n, adjust=False).mean().shift(1).to_numpy()
    )

    fast, slow = build_pair(spec, df, {})
    el, es = cross_masks(fast, slow)

    a, b = rth.split("-")
    lo = int(a[:2]) * 60 + int(a[3:5])
    hi = int(b[:2]) * 60 + int(b[3:5])
    mins = (df["dt"].dt.hour.to_numpy() * 60 + df["dt"].dt.minute.to_numpy()).astype(int)
    inwin = (mins >= lo) & (mins < hi)

    n = len(c)
    return dict(
        o=o,
        h=h,
        l=l,
        c=c,
        atr=atr,
        el=el,
        es=es,
        ok_long=np.ones(n, bool),
        ok_short=np.ones(n, bool),
        st_bull=fast > slow,
        inwin=inwin,
        n=n,
        ts=df["ts"].to_numpy().astype(np.int64),
        split=n,
        fast=fast,
        slow=slow,
    )
