"""Reference levels, breaks, and acceptance. The first test in this project with a mechanism.

WHY LEVELS ARE A DIFFERENT KIND OF OBJECT

Every variable tested in hypotheses 1-9 was a smoothing of past closes. A moving average is a
number; nothing rests on it and no one defends it. A prior-day value-area edge, a session VWAP
or a swing high marks a price where contracts actually changed hands and where stops actually
sit. That is a reason for a level to matter, rather than a pattern that happens to have worked.

ACCEPTANCE IS ALSO STRUCTURALLY NEW

Everything so far was a point-in-time state or a trailing-window statistic. Acceptance is a
POST-EVENT measurement: price breaks a level, and then we watch whether it stays beyond it.
Rejection and acceptance are the two things a level can do, and they are only distinguishable
after the fact -- which is why the entry sits at the END of the acceptance window and every
outcome is measured from there. Nothing in the window can leak into the trade.

WHY THE LEVELS ARE DERIVED FROM BARS AND NOT FROM THE TICK PROFILES

The project holds tick-derived daily profiles (POC, VAH, VAL and the full price-volume array)
for 1,651 MES and 999 MNQ days. They are better estimates. They are also USELESS AS A NULL: the
permutation re-strings prices, so a level fixed at a real price sits somewhere the permuted path
may never visit, and the null breaks in the direction that flatters the real result.

So every level here is recomputed from the bars, including under permutation. Measured against
the tick profiles on RTH bars, the bar approximation places POC within 2.2% of the day's range
and the value-area edges within 4.5-6.8%. That is not an exact match and does not need to be:
the requirement is a level that is causal and permutable, not one that reproduces a better
estimator. The tick profiles remain available for a more precise second pass if anything shows.

Recorded while checking this, because it was not documented anywhere: THE TICK PROFILES ARE
RTH-ONLY. Bar-derived volume matches them at a ratio of exactly 1.000 when restricted to
08:30-15:00 America/Chicago, and 1.366 over the full session.

THE RANDOM CONTROL

One of the thirteen levels is a random line placed at a plausible distance with no structural
meaning at all. It is the matched control the framework spec never asked for, and it answers the
question underneath the whole idea: do "meaningful" levels beat arbitrary ones? If prior-day VAH
cannot outperform a randomly placed line at the same distance, then acceptance is measuring
drift and the premise is dead before any of the rest matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TICK = 0.25
DAY_ROLL_HOURS = 7
RTH_LO, RTH_HI = 8 * 60 + 30, 15 * 60
OPENING_RANGE_MIN = 30
SWING_K = 5  # a pivot is confirmed K bars after it prints, never before

LEVEL_TYPES = (
    "pd_poc",
    "pd_vah",
    "pd_val",
    "hvn",
    "lvn",
    "vwap",
    "vwap_u1",
    "vwap_d1",
    "or_hi",
    "or_lo",
    "swing_hi",
    "swing_lo",
    "rand",
)
ACCEPT_WINDOWS = (6, 12)
ACCEPT_MODES = ("time", "vol")
OUTCOMES = ("ret_r", "mfe_r", "mae_r")


def session_day(dt: pd.Series) -> np.ndarray:
    return (dt + pd.Timedelta(hours=DAY_ROLL_HOURS)).dt.strftime("%Y%m%d").to_numpy()


def _day_bounds(day: np.ndarray):
    """(start, stop) index pairs per session, computed in one pass.

    The obvious `for d in days: mask = day == d` is O(bars x days): 1,600 days over 309,507 MES
    5m bars is 500M comparisons and was the real cost of building levels, not the profile maths.
    Sessions are contiguous, so their boundaries can be found once.
    """
    first = np.r_[True, day[1:] != day[:-1]]
    starts = np.flatnonzero(first)
    stops = np.r_[starts[1:], len(day)]
    return list(zip(day[starts], starts, stops, strict=True))


def _day_profile(high, low, vol, tick=TICK):
    """Volume at price for one day, each bar's volume spread over its own range.

    Returns (poc, vah, val, hvn, lvn). HVN and LVN are the local volume peak and trough nearest
    the POC other than the POC itself -- the two structural features a profile has beyond its
    value area.

    Fully vectorised. The obvious version builds the ragged price index with a Python loop over
    bars and took 26s for one MES 5m series, which is 4+ hours once the permutation recomputes
    it 300 times. np.repeat with an offset trick and np.bincount do the same work in one pass.
    """
    lo = np.floor(low / tick).astype(np.int64)
    hi = np.ceil(high / tick).astype(np.int64)
    span = np.maximum(hi - lo + 1, 1)
    per = vol / span
    base = int(lo.min())
    starts = np.cumsum(span) - span
    offs = np.arange(int(span.sum())) - np.repeat(starts, span)
    idx = np.repeat(lo - base, span) + offs
    vols = np.bincount(idx, weights=np.repeat(per, span), minlength=int(hi.max()) - base + 1)
    prices = (base + np.arange(len(vols))) * tick
    if len(prices) < 5 or vols.sum() <= 0:
        return (np.nan,) * 5

    poc = float(prices[vols.argmax()])
    order = np.argsort(-vols)
    cum = np.cumsum(vols[order])
    k = int(np.searchsorted(cum, 0.70 * vols.sum())) + 1
    inva = prices[order[:k]]
    vah, val = float(inva.max()), float(inva.min())

    sm = pd.Series(vols).rolling(5, center=True, min_periods=1).mean().to_numpy()
    up = np.r_[False, (sm[1:-1] > sm[:-2]) & (sm[1:-1] >= sm[2:]), False]
    dn = np.r_[False, (sm[1:-1] < sm[:-2]) & (sm[1:-1] <= sm[2:]), False]
    pk = prices[up]
    tr = prices[dn]
    pk = pk[np.abs(pk - poc) > 2 * tick]
    hvn = float(pk[np.abs(pk - poc).argmin()]) if len(pk) else np.nan
    lvn = float(tr[np.abs(tr - poc).argmin()]) if len(tr) else np.nan
    return poc, vah, val, hvn, lvn


def prior_day_levels(df: pd.DataFrame, min_rth_bars: int = 8) -> dict[str, np.ndarray]:
    """POC, VAH, VAL, HVN and LVN of the PREVIOUS session, carried across every bar of this one.

    Built from RTH bars only, which is what the tick profiles turn out to use. Days without a
    usable previous profile carry the most recent one forward rather than being dropped -- 20% of
    bar-days have no tick profile, and those are holidays and half-days, so dropping them would
    remove a non-random slice of the sample.

    `min_rth_bars` is 8, not 20. RTH is 6.5 hours, which is only 13 bars at 30m: a threshold of
    20 silently produced ZERO profiles on every 30m series, and the run would have reported a
    clean null for five level types that were never computed.
    """
    day = session_day(df["dt"])
    mins = (df["dt"].dt.hour * 60 + df["dt"].dt.minute).to_numpy()
    rth = (mins >= RTH_LO) & (mins < RTH_HI)
    h, lo_, v = (df[k].to_numpy(float) for k in ("high", "low", "volume"))

    keys = ("pd_poc", "pd_vah", "pd_val", "hvn", "lvn")
    out = {k: np.full(len(df), np.nan) for k in keys}
    last: tuple | None = None
    for _d, i0, i1 in _day_bounds(day):
        if last is not None:
            for k, val in zip(keys, last, strict=True):
                out[k][i0:i1] = val
        m = rth[i0:i1]
        if m.sum() >= min_rth_bars:
            prof = _day_profile(h[i0:i1][m], lo_[i0:i1][m], v[i0:i1][m])
            if np.isfinite(prof[0]):
                last = prof
    return out


def session_vwap_bands(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Session-anchored VWAP and its +/- 1 sigma bands, both causal within the session."""
    h, lo_, c, v = (df[k].to_numpy(float) for k in ("high", "low", "close", "volume"))
    tp = (h + lo_ + c) / 3.0
    day = session_day(df["dt"])
    first = np.r_[True, day[1:] != day[:-1]]
    sess = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    prev = starts - 1

    def anchored(x):
        cs = np.cumsum(x)
        off = np.where(prev >= 0, cs[np.maximum(prev, 0)], 0.0)
        return cs - off[sess]

    cv = anchored(v)
    cpv = anchored(tp * v)
    cpv2 = anchored(tp * tp * v)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(cv > 0, cpv / cv, np.nan)
        var = np.where(cv > 0, cpv2 / cv - vwap**2, np.nan)
    sd = np.sqrt(np.maximum(var, 0.0))
    return {"vwap": vwap, "vwap_u1": vwap + sd, "vwap_d1": vwap - sd}


def opening_range(df: pd.DataFrame, minutes: int = OPENING_RANGE_MIN) -> dict[str, np.ndarray]:
    """High and low of the first `minutes` of RTH, NaN until that window has completed."""
    mins = (df["dt"].dt.hour * 60 + df["dt"].dt.minute).to_numpy()
    day = session_day(df["dt"])
    inor = (mins >= RTH_LO) & (mins < RTH_LO + minutes)
    after = mins >= RTH_LO + minutes
    h, lo_ = df["high"].to_numpy(float), df["low"].to_numpy(float)
    out = {"or_hi": np.full(len(df), np.nan), "or_lo": np.full(len(df), np.nan)}
    for _d, i0, i1 in _day_bounds(day):
        w = inor[i0:i1]
        # >= 1, not >= 2. A 30-minute opening range is exactly ONE bar on the 30m series, and a
        # threshold of 2 silently produced zero opening-range levels there.
        if w.sum() < 1:
            continue
        a_ = after[i0:i1]
        out["or_hi"][i0:i1][a_] = h[i0:i1][w].max()
        out["or_lo"][i0:i1][a_] = lo_[i0:i1][w].min()
    return out


def swing_levels(df: pd.DataFrame, k: int = SWING_K) -> dict[str, np.ndarray]:
    """Most recent CONFIRMED pivot high and low.

    A pivot at bar i needs k bars on each side, so it is only known at bar i + k. Publishing it
    at bar i would be a k-bar look-ahead and would make every swing level clairvoyant.
    """
    h, lo_ = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(h)
    ph = pd.Series(h).rolling(2 * k + 1, center=True).max().to_numpy()
    pl = pd.Series(lo_).rolling(2 * k + 1, center=True).min().to_numpy()
    is_hi = np.isclose(h, ph) & np.isfinite(ph)
    is_lo = np.isclose(lo_, pl) & np.isfinite(pl)

    out = {"swing_hi": np.full(n, np.nan), "swing_lo": np.full(n, np.nan)}
    for name, flag, src in (("swing_hi", is_hi, h), ("swing_lo", is_lo, lo_)):
        val = np.full(n, np.nan)
        val[k:] = np.where(flag[:-k], src[:-k], np.nan)  # published k bars late
        out[name] = pd.Series(val).ffill().to_numpy()
    return out


def random_levels(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """The CONTROL: one line per session, at a plausible distance with no structural meaning.

    Placed at the previous session's close plus a draw scaled by the previous session's range, so
    it lives at the same order of distance as a real level and differs only in having no reason
    to exist. If the real levels cannot beat this, the premise is dead.
    """
    day = session_day(df["dt"])
    c, h, lo_ = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    rng = np.random.default_rng(seed)
    out = np.full(len(df), np.nan)
    prev_close = prev_range = None
    for _d, i0, i1 in _day_bounds(day):
        if prev_close is not None and np.isfinite(prev_range) and prev_range > 0:
            out[i0:i1] = prev_close + rng.normal(0.0, 0.5) * prev_range
        prev_close = c[i1 - 1]
        prev_range = h[i0:i1].max() - lo_[i0:i1].min()
    return out


def all_levels(df: pd.DataFrame, seed: int = 0) -> dict[str, np.ndarray]:
    """Every level, per bar, all causal and all recomputable from the bars alone."""
    out: dict[str, np.ndarray] = {}
    out.update(prior_day_levels(df))
    out.update(session_vwap_bands(df))
    out.update(opening_range(df))
    out.update(swing_levels(df))
    out["rand"] = random_levels(df, seed)
    return {k: out[k] for k in LEVEL_TYPES}


def breaks_and_acceptance(
    df: pd.DataFrame, level: np.ndarray, window: int
) -> dict[str, np.ndarray]:
    """Break events, their direction, and how much of the next `window` bars held beyond.

    Acceptance is measured over bars t+1..t+window and the ENTRY sits at t+window, so nothing
    inside the measurement window is available to the trade. Two versions: the fraction of BARS
    that closed beyond the level, and the fraction of VOLUME that traded in those bars.
    """
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    ok = np.isfinite(level)
    above = c > level
    prev = np.r_[False, above[:-1]]
    brk = ok & np.r_[False, ok[:-1]] & (above != prev)
    direction = np.where(above, 1.0, -1.0)

    def fwd_mean(x):
        return pd.Series(x).rolling(window).mean().shift(-window).to_numpy()

    acc_above = fwd_mean(above.astype(float))
    vol_above = pd.Series(v * above).rolling(window).sum().shift(-window).to_numpy()
    vol_tot = pd.Series(v).rolling(window).sum().shift(-window).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        vacc_above = np.where(vol_tot > 0, vol_above / vol_tot, np.nan)

    acc_time = np.where(direction > 0, acc_above, 1.0 - acc_above)
    acc_vol = np.where(direction > 0, vacc_above, 1.0 - vacc_above)

    def lag(x):
        return np.r_[np.full(window, np.nan), x[:-window]] if window else x

    def lag_bool(x):
        return np.r_[np.zeros(window, bool), x[:-window]] if window else x

    return {
        "entry": lag_bool(brk),
        "direction": np.nan_to_num(lag(direction)),
        "acc_time": lag(acc_time),
        "acc_vol": lag(acc_vol),
    }
