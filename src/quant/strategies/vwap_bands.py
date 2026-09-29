"""Anchored VWAP with volume-weighted standard deviation bands, and the races between them.

WHAT THE USER ASKED, AND THE ONE CORRECTION IT NEEDS

    "if we reject off the -2 standard deviation does it hit the mean of vwap more often or go
     to +2 standard deviation"

As posed this is not a race, because from -2 sigma BOTH the mean and +2 sigma lie ABOVE price.
Reaching +2 sigma requires passing through the mean first, so "hits the mean more often" is true
by construction and measures nothing. The meaningful decomposition is three separate races:

    A  REVERSION   from -k sigma, does price reach the VWAP before reaching -(k+1) sigma?
    B  TRAVERSE    having reached the VWAP, does it go on to +k sigma before falling to -(k+1)?
    C  FULL        from -k sigma, does it reach +k sigma before -(k+1) sigma?

A is "does the band reject". B is "does the rejection carry". C is the whole trip. Mirrored for
the positive bands. All three are reported, and all three are measured against a permutation
null, because A in particular has a large purely geometric baseline: the VWAP sits k sigma away
and the next band sits one sigma away, so even a driftless walk resolves A toward the mean most
of the time. The null is what separates the geometry from the effect.

CAUSALITY -- the thing that makes or breaks this measurement

A VWAP band computed from bars up to and including bar i is partly a function of that same bar
price. Testing whether bar i touched that band is therefore circular: a large bar drags the band
toward itself and "touches" it by arithmetic. Every band here is computed from bars strictly
BEFORE the bar being tested (mu and sigma are shifted one bar within their anchor), and the
first MIN_BARS bars of each anchor are discarded because a volume-weighted variance over two
or three bars is not a dispersion estimate.

Race levels are FROZEN at the touch bar. A trader marks the level and trades to it; a live band
that drifts toward price would make "reached the mean" partly a statement about the band moving.

ANCHORS

    ny           resets 08:30 America/Chicago, the cash open, and runs to the next 08:30
    session      resets 17:00 America/Chicago, the CME session open
    continuous   never resets -- the "running non stop" case
    weekly       resets at the Sunday session open
    monthly      resets at the first session of the calendar month
    quarterly    resets at the first session of the calendar quarter

DEFINITIONS

Typical price is (high + low + close) / 3, as ta.vwap uses. The dispersion is the
volume-weighted standard deviation of typical price about the VWAP, accumulated from the anchor,
which is the standard construction the TradingView VWAP bands use:

    mu    = sum(v * tp) / sum(v)
    sigma = sqrt( sum(v * tp^2) / sum(v)  -  mu^2 )

accumulated with prices centred on the first bar of the anchor, so the subtraction is not a
difference of two numbers near 4e8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RTH_LO, RTH_HI = 8 * 60 + 30, 15 * 60
MIN_BARS = 5  # bars of an anchor discarded before its sigma is treated as an estimate
BANDS = (1, 2, 3, 4)
ANCHORS = ("ny", "session", "continuous", "weekly", "monthly", "quarterly")


# ------------------------------------------------------------------ anchors
def anchor_ids(dt: pd.Series, kind: str) -> np.ndarray:
    """Integer id per bar, one distinct value per anchor period. Depends only on timestamps,
    so it is identical between the real market and any permutation of it."""
    if kind == "continuous":
        return np.zeros(len(dt), dtype=np.int64)
    if kind == "ny":
        key = (dt - pd.Timedelta(hours=8, minutes=30)).dt.strftime("%Y%m%d")
    else:
        sday = dt + pd.Timedelta(hours=7)  # 17:00 CT -> next calendar day
        if kind == "session":
            key = sday.dt.strftime("%Y%m%d")
        elif kind == "weekly":
            iso = sday.dt.isocalendar()
            key = iso["year"].astype(str) + "W" + iso["week"].astype(str).str.zfill(2)
        elif kind == "monthly":
            key = sday.dt.strftime("%Y%m")
        elif kind == "quarterly":
            key = sday.dt.strftime("%Y") + "Q" + sday.dt.quarter.astype(str)
        else:
            raise ValueError("unknown anchor " + repr(kind))
    return pd.factorize(np.asarray(key), sort=False)[0].astype(np.int64)


def _group_start(gid: np.ndarray) -> np.ndarray:
    """For every bar, the index of the first bar of its anchor period."""
    first = np.r_[True, gid[1:] != gid[:-1]]
    starts = np.flatnonzero(first)
    return starts[np.cumsum(first) - 1]


# ------------------------------------------------------------------ the bands
def vwap_bands(df: pd.DataFrame, gid: np.ndarray, min_bars: int = MIN_BARS):
    """Causal (mu, sigma): the values at index i are computed from bars strictly before i.

    Returns NaN for the first min_bars bars of every anchor period.
    """
    h, lo, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    v = df["volume"].to_numpy(float)
    tp = (h + lo + c) / 3.0

    gs = _group_start(gid)
    base = tp[gs]  # centre each anchor on its own first typical price
    x = tp - base

    cv = np.cumsum(v)
    cxv = np.cumsum(x * v)
    cx2v = np.cumsum(x * x * v)
    off = np.where(gs > 0, gs - 1, 0)
    z = gs > 0
    sv = cv - np.where(z, cv[off], 0.0)
    sxv = cxv - np.where(z, cxv[off], 0.0)
    sx2v = cx2v - np.where(z, cx2v[off], 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        m = sxv / sv
        var = sx2v / sv - m * m
    mu = base + m
    sigma = np.sqrt(np.maximum(var, 0.0))

    # shift one bar WITHIN the anchor, then void the opening bars
    mu_c = np.r_[np.nan, mu[:-1]]
    sg_c = np.r_[np.nan, sigma[:-1]]
    age = np.arange(len(gid)) - gs
    bad = (age < min_bars) | ~np.isfinite(sg_c) | (sg_c <= 0)
    mu_c[bad] = np.nan
    sg_c[bad] = np.nan
    return mu_c, sg_c


def band_index(mu: np.ndarray, sigma: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Signed integer band zone of close: 0 inside +/-1 sigma, +2 between +2 and +3, etc.

    NaN-safe: returns 0 where the bands are undefined, which never counts as a point of
    interest because callers require abs(zone) >= 1.
    """
    with np.errstate(invalid="ignore"):
        z = (close - mu) / sigma
    z = np.where(np.isfinite(z), z, 0.0)
    return np.trunc(z).astype(int)


def first_touches(
    mu: np.ndarray,
    sigma: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    gid: np.ndarray,
    k: int,
    sign: int,
) -> np.ndarray:
    """Index of the FIRST bar of each anchor period whose range reaches the sign*k band.

    One event per band per sign per period. Taking every bar beyond the band instead would
    return a run of near-duplicate events from one excursion and inflate every sample size.
    """
    level = mu + sign * k * sigma
    hit = (high >= level) if sign > 0 else (low <= level)
    hit &= np.isfinite(level)
    if not hit.any():
        return np.empty(0, dtype=np.int64)
    idx = np.flatnonzero(hit)
    g = gid[idx]
    keep = np.r_[True, g[1:] != g[:-1]]
    return idx[keep]


# ------------------------------------------------------------------ the races
def _first_hit(mat: np.ndarray, level: np.ndarray, up: bool, valid: np.ndarray | None = None):
    """Column index of the first row-wise crossing, or H when there is none."""
    hit = (mat >= level[:, None]) if up else (mat <= level[:, None])
    if valid is not None:
        hit = hit & valid
    any_ = hit.any(axis=1)
    return np.where(any_, hit.argmax(axis=1), mat.shape[1]), any_


def race(
    df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    ev: np.ndarray,
    k: int,
    sign: int,
    horizon: int,
) -> dict:
    """Races A, B and C from each event in ev, levels frozen at the event bar.

    sign is the side of the band that was touched: -1 for the lower bands, +1 for the upper.
    An ambiguous bar -- one whose range covers both the target and the extension -- is charged
    to the EXTENSION, which biases every reported reversion rate DOWNWARD. ambig reports how
    often that mattered.
    """
    n = len(df)
    ev = ev[(ev + horizon) < n]
    out = dict(
        n=len(ev),
        nA=0,
        nB=0,
        nC=0,
        pA=np.nan,
        pB=np.nan,
        pC=np.nan,
        unres=np.nan,
        ambig=np.nan,
    )
    if len(ev) == 0:
        return out

    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    win = ev[:, None] + 1 + np.arange(horizon)[None, :]
    H, L = hi[win], lo[win]

    m, s = mu[ev], sigma[ev]
    tgt = m  # the VWAP itself
    ext = m + sign * (k + 1) * s  # one band further out, the adverse side
    opp = m - sign * k * s  # the mirror band

    up = sign < 0  # a lower-band touch reverts UPWARD toward the mean
    iT, okT = _first_hit(H if up else L, tgt, up)
    iE, okE = _first_hit(L if up else H, ext, not up)
    iO, okO = _first_hit(H if up else L, opp, up)

    # ---- race A: mean vs extension. Ties are charged to the extension.
    A_win = okT & (iT < iE)
    A_los = okE & (iE <= iT)
    A_res = A_win | A_los
    out["ambig"] = float((okT & okE & (iT == iE)).mean())
    out["unres"] = float((~A_res).mean())
    out["nA"] = int(A_res.sum())
    out["pA"] = float(A_win.sum() / max(A_res.sum(), 1))

    # ---- race B: having reached the mean, the mirror band vs the extension, from that bar on
    pos = np.arange(horizon)[None, :]
    after = pos > iT[:, None]
    iO2, okO2 = _first_hit(H if up else L, opp, up, valid=after)
    iE2, okE2 = _first_hit(L if up else H, ext, not up, valid=after)
    B_win = A_win & okO2 & (iO2 < iE2)
    B_los = A_win & okE2 & (iE2 <= iO2)
    nB = int((B_win | B_los).sum())
    out["nB"] = nB
    out["pB"] = float(B_win.sum() / nB) if nB else np.nan

    # ---- race C: mirror band vs extension, straight from the touch
    C_win = okO & (iO < iE)
    C_los = okE & (iE <= iO)
    nC = int((C_win | C_los).sum())
    out["nC"] = nC
    out["pC"] = float(C_win.sum() / nC) if nC else np.nan
    return out


def in_rth(dt: pd.Series) -> np.ndarray:
    m = dt.dt.hour.to_numpy() * 60 + dt.dt.minute.to_numpy()
    return (m >= RTH_LO) & (m < RTH_HI)
