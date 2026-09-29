"""KAMA(20) x EMA(50) cross, fixed-point stops, and a trailing stop resolved at 1-minute.

WHY THIS ONE NEEDS A DIFFERENT SIMULATOR

Every strategy run so far used ATR-multiple exits evaluated on the signal timeframe. This one
specifies a TRAILING stop that activates after a fixed profit and then follows at a fixed
distance -- and a trailing stop evaluated only at 5-minute closes is not a trailing stop. Price
can travel far past the trail and come back inside a single 5m bar, and a 5m simulation would
never see it. On MES the trail distance is 2 ticks; a 5m bar routinely spans 20.

So the signal is computed on 5m bars and the TRADE is simulated on 1m bars. That is what
"intrabar" has to mean here: the entry is known at the 5m close (the cross needs the bar to
finish), but stops, targets and the trail are resolved at 1m. This is strictly more pessimistic
than 5m resolution, because finer bars find the stop more often.

It still is not tick-exact. Within a 1m bar we do not know the path, so the usual convention
applies: if stop and target are both inside a bar, assume the STOP filled first. With a trailing
stop the same rule extends to the trail.

KAMA -- Kaufman's Adaptive Moving Average

    ER  = |close - close[n]| / sum(|close - close[1]|, n)        efficiency ratio, 0..1
    SC  = (ER * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))^2        smoothing constant
    KAMA[i] = KAMA[i-1] + SC[i] * (close[i] - KAMA[i-1])

ER near 1 is a straight move, so KAMA tracks price closely; ER near 0 is chop, so KAMA almost
stops. That is the same intuition as the separation filter, built into the average itself --
which makes this a genuinely different idea from the EMA pair, not a relabelled one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy import ema, true_range

KAMA_FAST, KAMA_SLOW = 2, 30


def kama(
    close: np.ndarray, n: int = 20, fast: int = KAMA_FAST, slow: int = KAMA_SLOW
) -> np.ndarray:
    """Kaufman's Adaptive Moving Average. Causal by construction: bar i uses bars <= i only."""
    c = np.asarray(close, dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    change = np.abs(c - np.r_[np.full(n, np.nan), c[:-n]])
    vol = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        er = np.where(vol > 0, change / vol, 0.0)
    fsc, ssc = 2.0 / (fast + 1), 2.0 / (slow + 1)
    sc = (er * (fsc - ssc) + ssc) ** 2
    out[n] = c[n]
    for i in range(n + 1, len(c)):
        s = sc[i]
        out[i] = out[i - 1] + (s if np.isfinite(s) else ssc**2) * (c[i] - out[i - 1])
    return out


def signals(
    df5: pd.DataFrame,
    *,
    kama_n: int = 20,
    ema_n: int = 50,
    min_sep_atr: float = 0.035,
    atr_n: int = 14,
    rth: str = "08:30-15:00",
) -> pd.DataFrame:
    """Cross signals on the 5m frame, with the ATR-normalised separation confirmation.

    Returns the 5m frame plus `long_sig` / `short_sig`, true on the bar whose CLOSE confirms.
    """
    c = df5["close"].to_numpy(float)
    h, l = df5["high"].to_numpy(float), df5["low"].to_numpy(float)
    atr = (
        pd.Series(true_range(h, l, c)).ewm(alpha=1 / atr_n, adjust=False).mean().shift(1).to_numpy()
    )
    k, e = kama(c, kama_n), ema(c, ema_n)
    above = k > e
    prev = np.r_[False, above[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        sep = np.abs(k - e) / atr
    ok = np.isfinite(sep) & (sep >= min_sep_atr)

    a, b = rth.split("-")
    lo_m = int(a[:2]) * 60 + int(a[3:5])
    hi_m = int(b[:2]) * 60 + int(b[3:5])
    mins = df5["dt"].dt.hour.to_numpy() * 60 + df5["dt"].dt.minute.to_numpy()
    inwin = (mins >= lo_m) & (mins < hi_m)

    out = df5.copy()
    out["kama"], out["ema"], out["sep"], out["atr"] = k, e, sep, atr
    out["long_sig"] = (above & ~prev) & ok & inwin
    out["short_sig"] = (~above & prev) & ok & inwin
    return out


def simulate_1m(
    sig5: pd.DataFrame,
    df1: pd.DataFrame,
    *,
    stop_pts: float,
    tp_pts: float,
    trail_after: float,
    trail_dist: float,
    cost_pts: float,
    rth: str = "08:30-15:00",
) -> pd.DataFrame:
    """Signals from 5m, execution and exits resolved on 1m bars.

    Entry is the 5m signal bar's close, at the first 1m bar that starts at or after it -- the
    cross is only known once the 5m bar finishes, so entering earlier would be look-ahead.

    Exit precedence inside one 1m bar, worst-case for us:
        1. gap through the stop at the open
        2. stop (or trail, once armed) touched
        3. target touched
    Assuming the stop first when both are in range is the standard pessimistic convention; the
    alternative flatters every ambiguous bar.
    """
    o1 = df1["open"].to_numpy(float)
    h1, l1, c1 = (df1[k].to_numpy(float) for k in ("high", "low", "close"))
    t1 = df1["ts"].to_numpy(np.int64)
    a, b = rth.split("-")
    lo_m, hi_m = int(a[:2]) * 60 + int(a[3:5]), int(b[:2]) * 60 + int(b[3:5])
    mins1 = df1["dt"].dt.hour.to_numpy() * 60 + df1["dt"].dt.minute.to_numpy()
    in1 = (mins1 >= lo_m) & (mins1 < hi_m)
    last_rth = in1 & np.r_[~in1[1:], True]

    fire = sig5[sig5["long_sig"] | sig5["short_sig"]]
    starts = np.searchsorted(t1, fire["ts"].to_numpy(np.int64) + 300)  # 5m bar close -> next 1m
    rows = []
    busy_until = -1
    for (_, s), i0 in zip(fire.iterrows(), starts, strict=True):
        if i0 >= len(t1) or i0 <= busy_until:
            continue  # one position at a time
        d = 1 if s["long_sig"] else -1
        entry = c1[i0] if i0 < len(c1) else None
        if entry is None or not in1[i0]:
            continue
        stop = entry - d * stop_pts
        targ = entry + d * tp_pts
        best = entry
        armed = False
        exit_px = exit_i = reason = None
        worst = entry
        # Start at i0 + 1. We entered at the CLOSE of bar i0, so that bar's own open, high and
        # low already happened and cannot fill us. Starting at i0 manufactured instant stop-outs
        # from pre-entry price action -- it produced 477 'gap stops' in 833 MES trades.
        for i in range(i0 + 1, len(t1)):
            # ORDER MATTERS. Test this bar against the stop as it stood at the END of the
            # previous bar, THEN update the high-water mark and advance the trail.
            #
            # Doing it the other way -- raising the trail on bar i's high and then testing bar
            # i's low against it -- assumes the high happened before the low, which we cannot
            # know from OHLC. With a 0.5 pt trail and a 1m MES bar spanning several points it
            # stops out on nearly every bar: it produced 464 false exits in 833 trades and made
            # the trailing stop look worthless.
            if d > 0:
                if o1[i] <= stop:
                    exit_px, reason = o1[i], ("trail" if armed else "gap_stop")
                elif l1[i] <= stop:
                    exit_px, reason = stop, ("trail" if armed else "stop")
                elif h1[i] >= targ:
                    exit_px, reason = targ, "target"
            else:
                if o1[i] >= stop:
                    exit_px, reason = o1[i], ("trail" if armed else "gap_stop")
                elif h1[i] >= stop:
                    exit_px, reason = stop, ("trail" if armed else "stop")
                elif l1[i] <= targ:
                    exit_px, reason = targ, "target"

            if exit_px is None:
                best = max(best, h1[i]) if d > 0 else min(best, l1[i])
                worst = min(worst, l1[i]) if d > 0 else max(worst, h1[i])
                if not armed and (best - entry) * d >= trail_after:
                    armed = True
                if armed:
                    stop = (
                        max(stop, best - d * trail_dist)
                        if d > 0
                        else min(stop, best + d * trail_dist)
                    )

            if exit_px is None and last_rth[i]:
                exit_px, reason = c1[i], "session_close"
            if exit_px is not None:
                exit_i = i
                break
        if exit_px is None:
            exit_px, exit_i, reason = c1[-1], len(t1) - 1, "end"
        busy_until = exit_i
        rows.append(
            dict(
                entry_i=int(i0),
                exit_i=int(exit_i),
                direction=d,
                entry_px=float(entry),
                exit_px=float(exit_px),
                pnl_pts=float((exit_px - entry) * d - cost_pts),
                mfe=float((best - entry) * d),
                mae=float((entry - worst) * d),
                reason=reason,
                hold=int(exit_i - i0),
                armed=bool(armed),
            )
        )
    tr = pd.DataFrame(rows)
    if len(tr):
        tr["pnl_r"] = tr["pnl_pts"] / stop_pts  # R is the fixed stop, not ATR
    return tr
