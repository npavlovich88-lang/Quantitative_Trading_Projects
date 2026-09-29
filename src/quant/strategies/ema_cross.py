"""EMA(10) / EMA(29) crossover -- a deliberately ordinary strategy, used as a pipeline test.

The point is NOT to find an edge. It is to run something known-weak end to end and confirm the
framework says so. A validation protocol that passes a plain moving-average crossover on five
years of futures data is broken, and better to learn that on a throwaway than on a real idea.

neurotrader opens his methodology video with exactly this strategy for the same reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy import ema, true_range


def build_features(
    df: pd.DataFrame, rth: str = "08:30-15:00", fast: int = 10, slow: int = 29, atr_n: int = 14
) -> dict:
    """Feature dict compatible with strategy.simulate().

    Entry is the CROSS, not the state: going long every bar the fast EMA merely sits above the
    slow one is a different (and much more traded) strategy than entering when it crosses.
    """
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    tr = true_range(h, l, c)
    # R-unit is the PREVIOUS bar's ATR -- a stop sized from the current bar's own range is
    # look-ahead. Same convention as the rest of the project.
    atr = pd.Series(tr).ewm(alpha=1 / atr_n, adjust=False).mean().shift(1).to_numpy()

    f, s = ema(c, fast), ema(c, slow)
    above = f > s
    prev = np.r_[False, above[:-1]]
    cross_up = above & ~prev
    cross_dn = ~above & prev

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
        el=cross_up,
        es=cross_dn,
        ok_long=np.ones(n, bool),
        ok_short=np.ones(n, bool),  # no regime gate
        st_bull=above,  # unused unless exit="st_flip"
        inwin=inwin,
        n=n,
        ts=df["ts"].to_numpy().astype(np.int64),
        split=n,
    )
