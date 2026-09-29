"""
Bar permutation, ported from neurotrader's get_permutation (video NLBXgSmRBgU),
with two additions that matter for a SESSION-BASED futures strategy.

HIS ALGORITHM (unchanged in spirit)
  1. work in log price
  2. split every bar into four numbers:
        gap  = log(open_t)  - log(close_{t-1})
        r_h  = log(high_t)  - log(open_t)
        r_l  = log(low_t)   - log(open_t)
        r_c  = log(close_t) - log(open_t)
  3. shuffle the (r_h, r_l, r_c) TRIPLES with one permutation  -> bar shapes move as a unit
     shuffle the gaps with a SEPARATE permutation             -> gaps move independently
  4. re-string cumulatively from the real bar at `start_index`
  5. exponentiate
  Result: the same marginal distribution of moves, the same total drift, the same first
  open and (statistically) the same end point -- but NO serial dependence.  Any edge that
  survives on this data came from the shape of the distribution, not from structure.

  Multi-market: pass a list of DataFrames and ONE pair of permutations is applied to all of
  them, so cross-market correlation is preserved.

TWO ADDITIONS FOR US
  * volume travels with the bar shape (same perm1).  He tests crypto OHLC only; our VWMA
    reads volume, and a bar's volume belongs to its shape, not to its clock position.
  * `groups`: permute WITHIN groups instead of globally.  Our strategy trades RTH only and
    is flat overnight.  A global shuffle drops quiet 3am bars into the middle of the RTH
    session and moves the overnight gap into 11:00, which destroys the time-of-day
    volatility profile -- something our strategy never claimed to exploit, so destroying it
    makes the null too easy to beat.  Grouping by (first-bar-of-session, in-RTH) keeps the
    session skeleton and the time-of-day vol profile, and destroys only the SEQUENCE of
    moves, which is exactly what is on trial.  groups=None reproduces his plain version.
"""

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]


def session_groups(df, rth_lo=8 * 60 + 30, rth_hi=15 * 60, day_roll_hours=7):
    """(is_first_bar_of_session, is_rth) -> integer group id, one per bar."""
    dt = df["dt"]
    day = (dt + pd.Timedelta(hours=day_roll_hours)).dt.strftime("%Y-%m-%d").to_numpy()
    first = np.r_[True, day[1:] != day[:-1]]
    mins = (dt.dt.hour.to_numpy() * 60 + dt.dt.minute.to_numpy()).astype(int)
    rth = (mins >= rth_lo) & (mins < rth_hi)
    return (first.astype(int) * 2 + rth.astype(int)).astype(np.int64)


def _grouped_perm(n_from_start, groups_slice, rng):
    """A permutation of range(n) that only maps positions onto positions of the same group."""
    idx = np.arange(n_from_start)
    if groups_slice is None:
        return rng.permutation(idx)
    out = idx.copy()
    for g in np.unique(groups_slice):
        m = np.flatnonzero(groups_slice == g)
        out[m] = rng.permutation(m)
    return out


def get_permutation(ohlc, start_index=0, seed=None, groups=None, permute_volume=True):
    """ohlc: a DataFrame or a list of DataFrames sharing one index (multi-market).
    Returns the same type, with bars after `start_index` permuted."""
    single = isinstance(ohlc, pd.DataFrame)
    mkts = [ohlc] if single else list(ohlc)
    n_bars = len(mkts[0])
    for m in mkts:
        assert len(m) == n_bars, "all markets must share one index"
    rng = np.random.default_rng(seed)

    perm_index = start_index + 1
    perm_n = n_bars - perm_index
    g = None if groups is None else np.asarray(groups)[perm_index:]
    perm1 = _grouped_perm(perm_n, g, rng)  # intrabar shapes (+ volume)
    perm2 = _grouped_perm(perm_n, g, rng)  # gaps

    out = []
    for m in mkts:
        lb = np.log(m[OHLC].to_numpy(dtype=float))
        start_bar = lb[start_index].copy()
        r_o = lb[:, 0] - np.r_[np.nan, lb[:-1, 3]]  # open vs previous close
        r_h = lb[:, 1] - lb[:, 0]
        r_l = lb[:, 2] - lb[:, 0]
        r_c = lb[:, 3] - lb[:, 0]

        po = r_o[perm_index:][perm2]
        ph, pl, pc = (a[perm_index:][perm1] for a in (r_h, r_l, r_c))

        bars = lb.copy()
        bars[start_index] = start_bar
        prev_close = start_bar[3]
        for k in range(perm_n):
            o = prev_close + po[k]
            i = perm_index + k
            bars[i, 0] = o
            bars[i, 1] = o + ph[k]
            bars[i, 2] = o + pl[k]
            bars[i, 3] = o + pc[k]
            prev_close = bars[i, 3]

        d = m.copy()
        d[OHLC] = np.exp(bars)
        if permute_volume and "volume" in d.columns:
            v = d["volume"].to_numpy(dtype=float).copy()
            v[perm_index:] = v[perm_index:][perm1]
            d["volume"] = v
        out.append(d)
    return out[0] if single else out


def get_permutation_fast(ohlc, start_index=0, seed=None, groups=None, permute_volume=True):
    """Vectorised equivalent of get_permutation, for permutation tests that need hundreds of
    draws over hundreds of thousands of bars.

    NOT bit-identical to the reference. The reference accumulates as `(prev + open_gap) + body`
    and this accumulates as `prev + (open_gap + body)`; float addition is not associative, so
    the two differ in the last bits. Agreement is asserted to 1e-12 in log space by
    `tests/test_permutation_fast.py`, which also checks that the derived cross events are
    identical -- the property that actually matters downstream.

    Use `get_permutation` when exactness matters and this when the loop is the bottleneck.
    """
    single = isinstance(ohlc, pd.DataFrame)
    mkts = [ohlc] if single else list(ohlc)
    n_bars = len(mkts[0])
    for m in mkts:
        assert len(m) == n_bars, "all markets must share one index"
    rng = np.random.default_rng(seed)

    perm_index = start_index + 1
    perm_n = n_bars - perm_index
    g = None if groups is None else np.asarray(groups)[perm_index:]
    perm1 = _grouped_perm(perm_n, g, rng)
    perm2 = _grouped_perm(perm_n, g, rng)

    out = []
    for m in mkts:
        lb = np.log(m[OHLC].to_numpy(dtype=float))
        start_bar = lb[start_index].copy()
        r_o = lb[:, 0] - np.r_[np.nan, lb[:-1, 3]]
        r_h = lb[:, 1] - lb[:, 0]
        r_l = lb[:, 2] - lb[:, 0]
        r_c = lb[:, 3] - lb[:, 0]

        po = r_o[perm_index:][perm2]
        ph, pl, pc = (a[perm_index:][perm1] for a in (r_h, r_l, r_c))

        c0 = start_bar[3]
        closes = c0 + np.cumsum(po + pc)
        opens = np.r_[c0, closes[:-1]] + po

        bars = lb.copy()
        bars[start_index] = start_bar
        bars[perm_index:, 0] = opens
        bars[perm_index:, 1] = opens + ph
        bars[perm_index:, 2] = opens + pl
        bars[perm_index:, 3] = closes

        d = m.copy()
        d[OHLC] = np.exp(bars)
        if permute_volume and "volume" in d.columns:
            v = d["volume"].to_numpy(dtype=float).copy()
            v[perm_index:] = v[perm_index:][perm1]
            d["volume"] = v
        out.append(d)
    return out[0] if single else out
