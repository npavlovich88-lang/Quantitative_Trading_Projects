"""Momentum state quality as a CONDITIONER, not as a signal.

THE QUESTION THIS ASKS, AND WHY IT IS NOT THE ONE ALREADY ANSWERED SIX TIMES

Every hypothesis so far asked "does event X predict direction". The answer was no, consistently,
across 6,060 moving-average configurations. This asks a different question:

    WITHIN a directional state, does the QUALITY of the move predict what happens next?

Concretely: among all the bars where EMA(8) sits above EMA(21), compare the ones where price got
there efficiently against the ones where it rotated its way there. Same direction, same
instrument, same session -- only the path differs.

That framing matters for three reasons:

  1. It is a within-direction contrast, so unconditional drift cancels. No part of the answer can
     come from equity indices having gone up.
  2. It tests the claim the spec actually makes -- "high positive SER should outperform weak
     positive SER for long setups" -- rather than the claim we already disproved.
  3. It uses the EMA as PERMISSION rather than as a trigger, which is the one use of a moving
     average this project has never tested.

THE SIX CONDITIONERS, all causal over a lookback of N bars ending at t

    disp      (C_t - C_{t-N}) / ATR_N            net displacement in volatility units
    ser       (C_t - C_{t-N}) / sum|C_i - C_{i-1}|   signed efficiency ratio, -1..+1
    persist   |sum sign(r_i)| / N                 how many bars agree, 0..1
    rot       -(sign changes) / (N-1)             negated so higher is always cleaner
    speed     displacement per second, z-scored against the same time of day
    relvol    volume / expected volume for that time of day

DIRECTION ADJUSTMENT

`disp`, `ser` and `speed` are signed, so they are multiplied by the state's direction: inside a
bearish state a large NEGATIVE displacement is the supportive case. `persist`, `rot` and `relvol`
are unsigned and left alone. After this, "higher conditioner" means "more supportive of the trade
the state permits" for all six, which is what makes one family maximum meaningful across them.

THE COMPOSITE IS INCLUDED, AND IT COSTS NOTHING

The spec proposes 0.30*Z(disp) + 0.25*SER + 0.20*Z(speed) + 0.15*persist + 0.10*Z(relvol).
Those weights are PRE-SPECIFIED, not fitted, so testing them spends no selection-bias budget at
all -- it is one more cell in the family. Fitting them would be a continuous search over a
simplex that no permutation test can charge for, so the fitted version is not tested and will
not be.

CALIBRATION AND WHY IT IS A SLICE, NOT THE WHOLE SAMPLE

Terciles, z-score means and time-of-day baselines are all statistics of the data. Computing them
over the full sample and then applying them bar by bar is look-ahead: bar 500 would be labelled
"top tercile" using information from bar 400,000. So every threshold is computed on the FIRST
20% of the train segment and applied to the remaining 80%, which is also the only part evaluated.
Permutations recompute both the same way, so the null carries the identical construction.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from ..strategy import ema

CONDITIONERS = ("disp", "ser", "persist", "rot", "speed", "relvol", "composite")
SIGNED = frozenset({"disp", "ser", "speed"})
LOOKBACKS = (10, 20, 40)
EMA_STATES = ((8, 21), (20, 50))
SIDES = ("bull", "bear")
CALIB_FRAC = 0.20

# The spec's own weights, pre-specified and never fitted.
COMPOSITE_WEIGHTS = {"disp": 0.30, "ser": 0.25, "speed": 0.20, "persist": 0.15, "relvol": 0.10}

BAR_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "10m": 600,
    "15m": 900,
    "20m": 1200,
    "30m": 1800,
    "45m": 2700,
    "60m": 3600,
}


@dataclasses.dataclass(frozen=True)
class CondSpec:
    """One conditioning configuration: which measure, what lookback, which side of which state."""

    conditioner: str
    lookback: int
    fast: int
    slow: int
    side: str

    @property
    def label(self) -> str:
        return f"{self.conditioner}{self.lookback}/ema{self.fast}-{self.slow}/{self.side}"

    @property
    def family(self) -> str:
        return self.conditioner


def cond_family() -> list[CondSpec]:
    """The pre-declared family: 7 conditioners x 3 lookbacks x 2 EMA states x 2 sides = 84."""
    return [
        CondSpec(c, n, f, s, side)
        for c in CONDITIONERS
        for n in LOOKBACKS
        for f, s in EMA_STATES
        for side in SIDES
    ]


def _rolling_sum(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).sum().to_numpy()


def raw_conditioners(
    df: pd.DataFrame, atr: np.ndarray, n: int, bar_seconds: int, calib: slice
) -> dict[str, np.ndarray]:
    """The six raw measures over a lookback of n bars. Every one uses bars <= t only."""
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    r = np.r_[np.nan, np.diff(c)]
    net = c - np.r_[np.full(n, np.nan), c[:-n]]

    path = _rolling_sum(np.abs(r), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        disp = net / atr
        ser = np.where(path > 0, net / path, 0.0)

    sg = np.sign(r)
    persist = np.abs(_rolling_sum(sg, n)) / n
    flips = (sg != np.r_[np.nan, sg[:-1]]) & np.isfinite(sg) & np.isfinite(np.r_[np.nan, sg[:-1]])
    rot = -_rolling_sum(flips.astype(float), n) / max(n - 1, 1)

    speed = net / (n * bar_seconds)  # points per second; z-scoring makes the unit irrelevant

    # Time-of-day baselines, fitted on the calibration slice only.
    mins = (df["dt"].dt.hour.to_numpy() * 60 + df["dt"].dt.minute.to_numpy()).astype(int)
    speed_z = _deseasonalise(speed, mins, calib)
    relvol = _deseasonalise(v, mins, calib, ratio=True)

    return {
        "disp": disp,
        "ser": ser,
        "persist": persist,
        "rot": rot,
        "speed": speed_z,
        "relvol": relvol,
    }


def _deseasonalise(x: np.ndarray, mins: np.ndarray, calib: slice, ratio: bool = False):
    """z-score (or ratio) against the same minute-of-day, using the calibration slice only.

    A raw speed or volume mostly recovers the time of day: the RTH profile spans a factor of
    tens between the quiet hour and the open. Comparing a move against its OWN time of day is
    what makes "unusually fast" mean anything.
    """
    key = mins // 15  # 15-minute buckets: enough resolution, enough sample per bucket
    xc, kc = x[calib], key[calib]
    ok = np.isfinite(xc)
    out_mu = np.full(key.max() + 2, np.nan)
    out_sd = np.full(key.max() + 2, np.nan)
    for b in np.unique(kc[ok]):
        m = (kc == b) & ok
        if m.sum() >= 30:
            out_mu[b] = xc[m].mean()
            out_sd[b] = xc[m].std(ddof=1)
    mu, sd = out_mu[key], out_sd[key]
    with np.errstate(invalid="ignore", divide="ignore"):
        if ratio:
            return np.where(np.isfinite(mu) & (mu > 0), x / mu, np.nan)
        return np.where(np.isfinite(sd) & (sd > 0), (x - mu) / sd, np.nan)


def _z(x: np.ndarray, calib: slice) -> np.ndarray:
    xc = x[calib]
    xc = xc[np.isfinite(xc)]
    if len(xc) < 30 or xc.std(ddof=1) == 0:
        return np.full_like(x, np.nan)
    return (x - xc.mean()) / xc.std(ddof=1)


def build_masks(
    df: pd.DataFrame,
    atr: np.ndarray,
    specs: list[CondSpec],
    tf: str,
    calib: slice,
    eligible: np.ndarray,
) -> tuple[dict[tuple[int, int], np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """(direction array per EMA state, one (high-tercile, low-tercile) mask pair per spec).

    Terciles are computed on the calibration slice and applied outside it, so a bar is never
    labelled using statistics that include its own future.
    """
    c = df["close"].to_numpy(dtype=float)
    secs = BAR_SECONDS[tf]

    dirs: dict[tuple[int, int], np.ndarray] = {}
    states: dict[tuple[int, int], np.ndarray] = {}
    for f, s in {(sp.fast, sp.slow) for sp in specs}:
        above = ema(c, f) > ema(c, s)
        states[(f, s)] = above
        dirs[(f, s)] = np.where(above, 1.0, -1.0)

    raw = {n: raw_conditioners(df, atr, n, secs, calib) for n in {sp.lookback for sp in specs}}

    masks: list[tuple[np.ndarray, np.ndarray]] = []
    for sp in specs:
        above = states[(sp.fast, sp.slow)]
        in_state = above if sp.side == "bull" else ~above
        d = dirs[(sp.fast, sp.slow)]
        if sp.conditioner == "composite":
            r = raw[sp.lookback]
            val = (
                COMPOSITE_WEIGHTS["disp"] * _z(r["disp"] * d, calib)
                + COMPOSITE_WEIGHTS["ser"] * (r["ser"] * d)
                + COMPOSITE_WEIGHTS["speed"] * (r["speed"] * d)
                + COMPOSITE_WEIGHTS["persist"] * r["persist"]
                + COMPOSITE_WEIGHTS["relvol"] * _z(r["relvol"], calib)
            )
        else:
            val = raw[sp.lookback][sp.conditioner]
            if sp.conditioner in SIGNED:
                val = val * d
        pool = val[calib][in_state[calib] & np.isfinite(val[calib])]
        if len(pool) < 200:
            empty = np.zeros(len(c), bool)
            masks.append((empty, empty))
            continue
        lo_t, hi_t = np.quantile(pool, [1 / 3, 2 / 3])
        base = in_state & eligible & np.isfinite(val)
        masks.append((base & (val >= hi_t), base & (val <= lo_t)))
    return dirs, masks
