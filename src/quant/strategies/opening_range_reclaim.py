"""Opening-range break, retest, and 2-minute EMA reclaim. The user's rule, stated precisely.

THE RULE AS GIVEN (long side; short is the mirror)

    1. Mark the 30-minute opening range on NQ.
    2. On the 10m chart, wait for a 10m candle to CLOSE outside the range.
    3. Drop to the 2m chart, 20 EMA on close.
    4. When price closes BELOW the 20 EMA and is near the top of the opening range high...
    5. ...and then bounces off the ORB high and reclaims, closing ABOVE the 20 EMA: ENTER.
    6. Stop: the swing low of what touched the ORB high.
    7. Target: the pre-market high.

WHY THIS IS STRUCTURALLY DIFFERENT FROM HYPOTHESES 1-12

Every one of those was a moving average, or a variable that turned out to be a moving average
wearing another name. This is session structure: an opening range, a breakout, a retest of the
broken level, and a target at a structural price rather than an ATR multiple. The 20 EMA appears
only as the trigger that times the reclaim, not as the signal.

It also has a property none of the others had: a VARIABLE reward. The target is a real price, so
the risk-reward ratio is whatever the session hands you. Given that wide fixed targets measured
WORSE than a coin flip (P(target) 0.051 against 0.167 at 1:5 on MES 5m), a structural target is
a materially different bet from a fixed multiple.

FIVE AMBIGUITIES, DECLARED RATHER THAN GUESSED

  1. "near the top of the opening range high" -- needs a tolerance. Parameterised as a multiple
     of ATR(14) on the 2m frame, swept over a small declared grid and charged for by the
     permutation test. This is the only genuinely free parameter in the rule.
  2. "pre-market high" -- taken as the high from the session open (17:00 America/Chicago) to
     08:30, i.e. the whole overnight. The narrower reading, a few hours before the cash open,
     gives a nearer target and is available as `premarket_from`.
  3. "the swing low of what touched the ORB high" -- the lowest low from the touch bar through
     the entry bar inclusive.
  4. The setup expires at the RTH close and takes at most one trade per side per session.
  5. If the target sits behind the entry -- for a long, a pre-market high already below the ORB
     high -- the setup is void. Measured: that is the case 41% of sessions on the long side.

MEASURED BEFORE BUILDING

MNQ train, 421 usable sessions (the 396-day data hole ends train in 2023-05):
    ORB range              median 105 pts, p10 57, p90 178
    price breaks the ORB   69.4% above, 68.4% below
    target live            58.7% long, 59.6% short
    distance to target     median 48 pts, p10 8, p90 148
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

RTH_OPEN, RTH_CLOSE = 8 * 60 + 30, 15 * 60
ORB_MINUTES = 30
DAY_ROLL_HOURS = 7
EMA_LEN = 20
ATR_LEN = 14
NEAR_ATR = (0.25, 0.50, 1.00)  # the declared tolerance grid for "near the ORB high"


@dataclasses.dataclass(frozen=True)
class Trade:
    session: str
    side: int  # +1 long, -1 short
    entry_i: int
    entry: float
    stop: float
    target: float
    exit_i: int
    exit: float
    reason: str
    r_multiple: float
    bars_held: int


def resample(df1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 1m bars to `minutes`. Bars are labelled by their START, as elsewhere here."""
    g = (df1["ts"] // (minutes * 60)) * (minutes * 60)
    out = df1.groupby(g).agg(
        ts=("ts", "first"),
        dt=("dt", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return out.reset_index(drop=True)


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def atr(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    tr[0] = h[0] - lo[0]
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().shift(1).to_numpy()


def session_of(dt: pd.Series) -> np.ndarray:
    return (dt + pd.Timedelta(hours=DAY_ROLL_HOURS)).dt.strftime("%Y%m%d").to_numpy()


def _bounds(key: np.ndarray):
    first = np.r_[True, key[1:] != key[:-1]]
    s = np.flatnonzero(first)
    return list(zip(key[s], s, np.r_[s[1:], len(key)], strict=True))


def find_trades(
    df1: pd.DataFrame,
    near_atr: float = 0.50,
    premarket_from: int | None = None,
    sides: tuple[int, ...] = (1, -1),
) -> list[Trade]:
    """Every trade the rule produces. `df1` must be 1-minute bars over the evaluated span.

    `premarket_from` restricts the pre-market window to minutes >= that value (e.g. 240 for
    04:00 onward); None uses the whole overnight from the session open.
    """
    d2 = resample(df1, 2)
    d10 = resample(df1, 10)

    m1 = (df1["dt"].dt.hour * 60 + df1["dt"].dt.minute).to_numpy()
    s1 = session_of(df1["dt"])
    h1, l1 = df1["high"].to_numpy(float), df1["low"].to_numpy(float)

    m2 = (d2["dt"].dt.hour * 60 + d2["dt"].dt.minute).to_numpy()
    s2 = session_of(d2["dt"])
    o2, h2, l2, c2 = (d2[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    e2 = ema(c2, EMA_LEN)
    a2 = atr(h2, l2, c2, ATR_LEN)

    m10 = (d10["dt"].dt.hour * 60 + d10["dt"].dt.minute).to_numpy()
    s10 = session_of(d10["dt"])
    c10 = d10["close"].to_numpy(float)

    idx2 = {k: (a, b) for k, a, b in _bounds(s2)}
    idx10 = {k: (a, b) for k, a, b in _bounds(s10)}
    out: list[Trade] = []

    for key, i0, i1 in _bounds(s1):
        if key not in idx2 or key not in idx10:
            continue
        w = slice(i0, i1)
        mm = m1[w]
        inorb = (mm >= RTH_OPEN) & (mm < RTH_OPEN + ORB_MINUTES)
        pre = mm < RTH_OPEN if premarket_from is None else (mm >= premarket_from) & (mm < RTH_OPEN)
        if inorb.sum() < ORB_MINUTES - 5 or pre.sum() < 60:
            continue
        orb_hi, orb_lo = h1[w][inorb].max(), l1[w][inorb].min()
        pre_hi, pre_lo = h1[w][pre].max(), l1[w][pre].min()

        a10, b10 = idx10[key]
        a2s, b2s = idx2[key]

        for side in sides:
            level = orb_hi if side > 0 else orb_lo
            target = pre_hi if side > 0 else pre_lo
            # Rule 5: a target behind the entry voids the setup.
            if (side > 0 and target <= level) or (side < 0 and target >= level):
                continue

            # Rule 2: the first 10m CLOSE outside the range, after the range completes.
            br = None
            for j in range(a10, b10):
                if m10[j] < RTH_OPEN + ORB_MINUTES or m10[j] >= RTH_CLOSE:
                    continue
                if (side > 0 and c10[j] > orb_hi) or (side < 0 and c10[j] < orb_lo):
                    br = m10[j] + 10  # the setup is armed once that 10m bar has CLOSED
                    break
            if br is None:
                continue

            # Rules 3-5 on the 2m frame.
            touch = None
            for j in range(a2s, b2s):
                if (
                    m2[j] < br
                    or m2[j] >= RTH_CLOSE
                    or not np.isfinite(e2[j])
                    or not np.isfinite(a2[j])
                ):
                    continue
                tol = near_atr * a2[j]
                if touch is None:
                    # closed on the wrong side of the EMA AND back near the broken level
                    below = c2[j] < e2[j] if side > 0 else c2[j] > e2[j]
                    near = abs((l2[j] if side > 0 else h2[j]) - level) <= tol
                    if below and near:
                        touch = j
                    continue
                # reclaim: a close back through the EMA in the trade's direction
                reclaim = c2[j] > e2[j] if side > 0 else c2[j] < e2[j]
                if not reclaim:
                    continue
                entry = c2[j]
                # Rule 6: swing extreme from the touch bar through the entry bar.
                seg = slice(touch, j + 1)
                stop = l2[seg].min() if side > 0 else h2[seg].max()
                if (side > 0 and stop >= entry) or (side < 0 and stop <= entry):
                    touch = None
                    continue
                if (side > 0 and target <= entry) or (side < 0 and target >= entry):
                    break  # target already taken out while we waited
                # resolve on the 2m frame, stop assumed first when both are inside one bar
                ex_i, ex_px, why = b2s - 1, c2[b2s - 1], "session_close"
                for k in range(j + 1, b2s):
                    if m2[k] >= RTH_CLOSE:
                        ex_i, ex_px, why = k, o2[k], "session_close"
                        break
                    hit_s = l2[k] <= stop if side > 0 else h2[k] >= stop
                    hit_t = h2[k] >= target if side > 0 else l2[k] <= target
                    if hit_s:
                        ex_i, ex_px, why = k, stop, "stop"
                        break
                    if hit_t:
                        ex_i, ex_px, why = k, target, "target"
                        break
                risk = abs(entry - stop)
                out.append(
                    Trade(
                        session=key,
                        side=side,
                        entry_i=j,
                        entry=float(entry),
                        stop=float(stop),
                        target=float(target),
                        exit_i=ex_i,
                        exit=float(ex_px),
                        reason=why,
                        r_multiple=float((ex_px - entry) * side / risk) if risk > 0 else np.nan,
                        bars_held=int(ex_i - j),
                    )
                )
                break  # one trade per side per session
    return out
