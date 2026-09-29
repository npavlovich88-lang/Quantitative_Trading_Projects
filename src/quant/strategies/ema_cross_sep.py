"""EMA cross with a separation filter: do not trade while the EMAs are close together.

THE IDEA

A moving-average crossover fires constantly in chop, because in chop the two averages sit on
top of each other and cross back and forth on noise. Each crossing pays a full round turn. The
proposed filter: only take the trade once the averages have actually pulled apart, on the
reasoning that separation is what distinguishes a developing trend from oscillation.

WHY THE FILTER CANNOT BE APPLIED AT THE CROSS ITSELF

Measured on the MES 5m train segment, 6,024 crosses: at the cross bar the normalised separation
|EMA20 - EMA50| / ATR(14) has a median of 0.035 ATR and a 90th percentile of 0.103. That is
near zero by construction -- a cross IS the moment the two lines touch. A threshold there
either admits everything or rejects everything; it cannot discriminate.

Three bars later the same measure spans p10 0.054, median 0.231, p90 0.488. That is a usable
range, so the filter is implemented as CONFIRMATION: after a cross, wait until separation
reaches the threshold, then enter -- provided the cross direction still holds and the wait has
not timed out.

WHAT THIS COSTS

Confirmation is not free. Waiting for separation means entering later and worse in the trades
that do run, in exchange for skipping the ones that never develop. Whether that trade is
worthwhile is the entire hypothesis, and it is exactly what the threshold sweep measures.

NORMALISATION

Separation is divided by ATR(14), shifted one bar. Raw point distance is not comparable across
MES at 2,500 in 2020 and 6,000 in 2026, nor across volatility regimes within either. This is
the same normalisation the project uses everywhere else, and the same reasoning neurotrader
gives for intramarket differencing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy import ema, true_range


def separation(df: pd.DataFrame, fast: int = 20, slow: int = 50, atr_n: int = 14) -> np.ndarray:
    """|EMA(fast) - EMA(slow)| / ATR(atr_n), with ATR shifted one bar."""
    c = df["close"].to_numpy(dtype=float)
    h, l = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    atr = (
        pd.Series(true_range(h, l, c)).ewm(alpha=1 / atr_n, adjust=False).mean().shift(1).to_numpy()
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(ema(c, fast) - ema(c, slow)) / atr


def build_features(
    df: pd.DataFrame,
    rth: str = "08:30-15:00",
    fast: int = 20,
    slow: int = 50,
    atr_n: int = 14,
    min_sep: float = 0.0,
    max_wait: int = 12,
) -> dict:
    """Feature dict compatible with strategy.simulate().

    min_sep = 0.0 reproduces the unfiltered crossover exactly, which is the control the
    threshold sweep is measured against.

    max_wait bounds the confirmation window. Without it a cross could be "pending" for days and
    fire on an unrelated move; 12 bars is one hour on 5m data. It is held FIXED rather than
    swept, to keep the search one-dimensional -- every extra swept parameter is another trial
    that PBO has to charge for.
    """
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    tr = true_range(h, l, c)
    atr = pd.Series(tr).ewm(alpha=1 / atr_n, adjust=False).mean().shift(1).to_numpy()

    f, s = ema(c, fast), ema(c, slow)
    above = f > s
    prev = np.r_[False, above[:-1]]
    cross_up, cross_dn = above & ~prev, ~above & prev
    with np.errstate(invalid="ignore", divide="ignore"):
        sep = np.abs(f - s) / atr

    n = len(c)
    el = np.zeros(n, bool)
    es = np.zeros(n, bool)
    if min_sep <= 0:
        el, es = cross_up.copy(), cross_dn.copy()
    else:
        # Walk forward from each cross until separation confirms, the direction flips, or the
        # wait times out. Strictly causal: bar i only ever consults bars <= i.
        pending_dir = 0
        waited = 0
        for i in range(n):
            if cross_up[i]:
                pending_dir, waited = 1, 0
            elif cross_dn[i]:
                pending_dir, waited = -1, 0
            elif pending_dir:
                waited += 1
                if waited > max_wait:
                    pending_dir = 0
            if pending_dir and np.isfinite(sep[i]) and sep[i] >= min_sep:
                still = above[i] if pending_dir > 0 else not above[i]
                if still:
                    if pending_dir > 0:
                        el[i] = True
                    else:
                        es[i] = True
                pending_dir = 0

    a, b = rth.split("-")
    lo = int(a[:2]) * 60 + int(a[3:5])
    hi = int(b[:2]) * 60 + int(b[3:5])
    mins = (df["dt"].dt.hour.to_numpy() * 60 + df["dt"].dt.minute.to_numpy()).astype(int)
    inwin = (mins >= lo) & (mins < hi)

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
        st_bull=above,
        inwin=inwin,
        n=n,
        ts=df["ts"].to_numpy().astype(np.int64),
        split=n,
        sep=sep,
    )
