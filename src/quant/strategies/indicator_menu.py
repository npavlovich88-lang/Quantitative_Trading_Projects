"""The structured candidate array: indicator families x mathematical variants x signal definitions.

WHY THIS FILE EXISTS IN THIS SHAPE

Twelve moving-average hypotheses were rejected here one at a time, and then scripts/
direction_ceiling.py bounded the WHOLE linear family at once: the best in-sample accuracy from
any causal linear filter of the last K returns is about +1.5 points over a null fitted the same
way on noise. Every SMA, EMA, WMA, HMA, DEMA, TEMA, ROC, MACD, PPO, TRIX and linear-regression
slope rule is one point inside that space, so all of them are settled.

What that bound does NOT cover is anything non-linear in returns. Sorting the candidate menu by
that line leaves a short list, and its members share one property:

    KAMA, VIDYA     weights depend on an efficiency ratio / volatility index
    RSI, CMO        up-magnitude against down-magnitude, so absolute values
    +DI / -DI       directional movement over true range
    TSI             double-smoothed momentum over double-smoothed |momentum|
    Donchian        max and min over a window
    Bollinger       rolling standard deviation
    Keltner         average true range

Every one of them normalises by volatility or range rather than filtering price. So the open
question is not WHICH filter, it is whether volatility normalisation carries information that
price filtering does not.

EMA, MACD and ROC are retained as NEGATIVE CONTROLS. They are known to be inside the bound, so
they have an expected score, and a harness that flatters them is broken.

SIGNAL DEFINITIONS

A line that hugs price does not make its crossing useful; it makes crossings frequent. So the
signal definition is a dimension in its own right, not an implementation detail. Six are defined,
each with a mirrored bearish case, and each evaluated identically.

    position     value on the bullish side of its reference
    fresh        was on the bearish side on the previous completed bar, bullish now
    buffered     crosses by at least BUFFER_ATR x ATR, so small fluctuations do not qualify
    slope        on the bullish side AND the reference rising over SLOPE_N bars
    sustained    on the bullish side for SUSTAIN_N consecutive completed bars
    aligned      a faster line on the bullish side of a slower one

Everything is computed from bars up to and including bar t and scored against the move from t to
t+N, so no value can read its own outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy import ema

SLOPE_N, SUSTAIN_N, BUFFER_ATR = 5, 3, 0.25

# name -> whether the linear-filter ceiling already bounds it
BOUNDED = {"ema": True, "macd": True, "roc": True}


# ------------------------------------------------------------------ helpers
def _atr(h, lo, c, n=14):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    tr[0] = h[0] - lo[0]
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def _rma(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


# ------------------------------------------------------------------ indicators
def kama(c, n=10, fast=2, slow=30):
    """Kaufman adaptive MA. The efficiency ratio makes the smoothing data-dependent, which is
    what puts it outside the linear bound."""
    ch = np.abs(c - np.r_[np.full(n, np.nan), c[:-n]])
    vol = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        er = np.where(vol > 0, ch / vol, 0.0)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    sc = np.nan_to_num(sc)
    out = np.empty_like(c)
    out[0] = c[0]
    for i in range(1, len(c)):
        out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])
    return out


def vidya(c, n=14, smooth=20):
    """Variable index dynamic average: smoothing scaled by a Chande momentum ratio."""
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).rolling(n).sum().to_numpy()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).rolling(n).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.abs((up - dn) / (up + dn))
    k = np.nan_to_num(k)
    a = 2.0 / (smooth + 1)
    out = np.empty_like(c)
    out[0] = c[0]
    for i in range(1, len(c)):
        out[i] = out[i - 1] + a * k[i] * (c[i] - out[i - 1])
    return out


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    g, l_ = _rma(np.where(d > 0, d, 0.0), n), _rma(np.where(d < 0, -d, 0.0), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(l_ > 0, 100 - 100 / (1 + g / l_), 100.0)


def cmo(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).rolling(n).sum().to_numpy()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).rolling(n).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((up + dn) > 0, 100 * (up - dn) / (up + dn), 0.0)


def di(h, lo, c, n=14):
    """+DI and -DI from Wilder's directional movement."""
    up, dn = np.diff(h, prepend=h[0]), -np.diff(lo, prepend=lo[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _atr(h, lo, c, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (
            np.where(tr > 0, 100 * _rma(pdm, n) / tr, 0.0),
            np.where(tr > 0, 100 * _rma(ndm, n) / tr, 0.0),
        )


def tsi(c, long_=25, short=13):
    m = np.diff(c, prepend=c[0])
    num = _rma(_rma(m, long_), short)
    den = _rma(_rma(np.abs(m), long_), short)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 100 * num / den, 0.0)


def donchian(h, lo, n=20):
    hh = pd.Series(h).rolling(n).max().to_numpy()
    ll = pd.Series(lo).rolling(n).min().to_numpy()
    return hh, ll, (hh + ll) / 2.0


def bollinger(c, n=20, k=2.0):
    m = pd.Series(c).rolling(n).mean().to_numpy()
    s = pd.Series(c).rolling(n).std().to_numpy()
    return m + k * s, m - k * s, m


def keltner(h, lo, c, n=20, k=2.0):
    m = ema(c, n)
    a = _atr(h, lo, c, n)
    return m + k * a, m - k * a, m


def macd(c, f=12, s=26, sig=9):
    line = ema(c, f) - ema(c, s)
    return line, ema(line, sig)


def roc(c, n=10):
    p = np.r_[np.full(n, np.nan), c[:-n]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(p > 0, 100 * (c - p) / p, 0.0)


# ------------------------------------------------------------------ the menu
def build(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (value, reference). Bullish means value > reference."""
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    z = np.zeros_like(c)
    p, n = di(h, lo, c)
    _du, _dl, dm = donchian(h, lo)
    bu, bl, bm = bollinger(c)
    ku, kl, km = keltner(h, lo, c)
    ml, ms = macd(c)
    return {
        # ---- open: non-linear in returns
        "kama": (c, kama(c)),
        "vidya": (c, vidya(c)),
        "rsi": (rsi(c), np.full_like(c, 50.0)),
        "cmo": (cmo(c), z),
        "di": (p, n),
        "tsi": (tsi(c), z),
        "donchian_mid": (c, dm),
        "bollinger_mid": (c, bm),
        "keltner_mid": (c, km),
        "bollinger_pctb": (c - bl, bu - c),  # bullish in the upper half of the band
        "keltner_pos": (c - kl, ku - c),
        # ---- negative controls: known to sit inside the linear bound
        "ema": (c, ema(c, 20)),
        "macd": (ml, ms),
        "roc": (roc(c), z),
    }


def signals(value: np.ndarray, ref: np.ndarray, atr: np.ndarray, fast_ref=None) -> dict:
    """Six signal definitions, +1 bullish / -1 bearish / 0 no opinion, all causal."""
    d = value - ref
    prev = np.r_[np.nan, d[:-1]]
    side = np.sign(d)
    out: dict[str, np.ndarray] = {}

    out["position"] = np.nan_to_num(side)

    fresh = np.zeros_like(d)
    fresh[(d > 0) & (prev <= 0)] = 1.0
    fresh[(d < 0) & (prev >= 0)] = -1.0
    out["fresh"] = fresh

    with np.errstate(invalid="ignore"):
        buf = np.where(atr > 0, d / atr, 0.0)
    b = np.zeros_like(d)
    b[(buf >= BUFFER_ATR) & (prev <= 0)] = 1.0
    b[(buf <= -BUFFER_ATR) & (prev >= 0)] = -1.0
    out["buffered"] = b

    rs = np.r_[np.full(SLOPE_N, np.nan), ref[SLOPE_N:] - ref[:-SLOPE_N]]
    sl = np.zeros_like(d)
    sl[(d > 0) & (rs > 0)] = 1.0
    sl[(d < 0) & (rs < 0)] = -1.0
    out["slope"] = np.nan_to_num(sl)

    pos = pd.Series(np.sign(np.nan_to_num(d)))
    up = pos.rolling(SUSTAIN_N).min().to_numpy()
    dnn = pos.rolling(SUSTAIN_N).max().to_numpy()
    su = np.zeros_like(d)
    su[up == 1.0] = 1.0
    su[dnn == -1.0] = -1.0
    out["sustained"] = su

    if fast_ref is None:
        fast_ref = ema(value, 5)
    out["aligned"] = np.nan_to_num(np.sign(fast_ref - ref))
    return out


def atr_for(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    return _atr(
        df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float), n
    )
