#!/usr/bin/env python3
"""
neurotrader's two permutation tests (github.com/neurotrader888/mcpt, video NLBXgSmRBgU),
applied to OUR strategy family.  Three deliberate departures from his code, each for a
reason stated here so the handoff can argue with them:

1. OBJECTIVE ON PER-BAR P&L, NOT PER-TRADE R.
   He computes profit factor on `signal * next_bar_log_return` -- tens of thousands of
   observations.  Masters ("Testing and Tuning Market Trading Systems", which neurotrader
   cites in his book video) argues analysing completed trade returns is a mistake, and
   neurotrader says he made that mistake for years.  Our strategy exits intrabar on stops
   and targets, so a naive position*return series would misprice every stop fill.  So we
   mark the open position to market bar by bar and use the ACTUAL fill price on the exit
   bar.  Result: a per-bar P&L series with ~130k observations that still respects stops.
   `pf_trade` is reported alongside, only so the numbers tie back to earlier work.

2. WHAT GETS RE-OPTIMISED IS THE EXIT GRID.
   The test is only honest if the permutation re-runs THE SEARCH WE ACTUALLY RAN.  We
   searched stop/target multiples; indicator lengths (9/52/14/20/12/14) were carried in
   fixed.  So the p-value covers the exit search and NOT any search over lengths.  If you
   ever tuned a length on this data, this p-value is optimistic and you must widen the grid.

3. GROUPED PERMUTATION (see permutation.py).
   Default keeps the session skeleton and the time-of-day volatility profile intact and
   destroys only the sequence of moves.  `--ungrouped` reproduces his plain version, which
   for an RTH-only strategy is an EASIER null (it randomises away the intraday vol profile
   the strategy never claimed to exploit).  Run both; report both.

Usage:
  python mcpt.py insample     --csv ... --point-value 2 --perms 1000
  python mcpt.py walkforward  --csv ... --point-value 2 --perms 100
"""

import argparse
import json
import time

import numpy as np

from . import strategy as S
from .permutation import get_permutation, session_groups

# the exit search we actually ran -- this, not the final cell, is what is on trial
GRID = [
    dict(exit="fixed", stop_atr=s, tp_atr=t)
    for s in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    for t in (None, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
] + [dict(exit="st_flip", stop_atr=s, tp_atr=None) for s in (None, 3.0, 6.0)]


# ------------------------------------------------------------------ objective
def bar_pnl(F, tr, cost_pts):
    """Per-bar P&L in points: mark the open position to market each bar, use the real fill
    price on the exit bar, charge the round turn on the exit bar."""
    c, n = F["c"], F["n"]
    out = np.zeros(n)
    for ei, xi, d, xp in zip(
        tr["entry_i"].to_numpy(),
        tr["exit_i"].to_numpy(),
        tr["direction"].to_numpy(),
        tr["exit_px"].to_numpy(),
        strict=True,
    ):
        ei, xi, d = int(ei), int(xi), int(d)
        if xi == ei:  # opened and closed on one bar
            out[xi] += d * (xp - c[ei]) - cost_pts
            continue
        out[ei + 1 : xi] += d * np.diff(c[ei:xi])  # entry at close -> first move is ei->ei+1
        out[xi] += d * (xp - c[xi - 1]) - cost_pts
    return out


def pf(x):
    w, l = x[x > 0].sum(), -x[x < 0].sum()
    return float(w / l) if l > 0 else np.nan


ARR = ("o", "h", "l", "c", "atr", "el", "es", "ok_long", "ok_short", "st_bull", "inwin", "ts")
WARM = 1500  # bars of indicator history kept before the window; max indicator
# lookback is 52+26=78 bars, so this is generous


def sub(F, a, b):
    """Slice already-computed features. Indicator values were computed on the FULL series,
    so slicing is exact -- no recomputation, no look-ahead."""
    G = {k: F[k][a:b] for k in ARR}
    G["n"] = b - a
    G["split"] = 0
    return G


def objective(F, cfg, cost_pts, lo=0, hi=None):
    """Profit factor of per-bar P&L over bars [lo, hi), counting only trades that ENTER
    in the window. Returns (pf, trades, bar_pnl aligned to [lo, hi))."""
    hi = F["n"] if hi is None else hi
    a = max(0, lo - WARM)
    G = sub(F, a, hi)
    tr = S.simulate(G, cfg["exit"], cfg["stop_atr"], cfg["tp_atr"], cost_pts)
    if len(tr):
        tr = tr[tr["entry_i"] >= lo - a]
    if len(tr) == 0:
        return np.nan, tr, np.zeros(hi - lo)
    b = bar_pnl(G, tr, cost_pts)
    return pf(b), tr, b[lo - a :]


def optimize(F, cost_pts, lo=0, hi=None, grid=GRID):
    best, best_pf, best_tr = None, -np.inf, None
    for cfg in grid:
        p, tr, _ = objective(F, cfg, cost_pts, lo, hi)
        if np.isfinite(p) and p > best_pf and len(tr) >= 20:
            best, best_pf, best_tr = cfg, p, tr
    return best, best_pf, best_tr


# ------------------------------------------------------------------ patched simulate
_SIM = S.simulate


def simulate_with_px(F, exit_mode, stop_atr, tp_atr, cost_pts):
    """S.simulate does not return the exit price; recover it from pnl_pts."""
    tr = _SIM(F, exit_mode, stop_atr, tp_atr, cost_pts)
    if len(tr):
        ep = F["c"][tr["entry_i"].to_numpy().astype(int)]
        tr["exit_px"] = ep + (tr["pnl_pts"].to_numpy() + cost_pts) * tr["direction"].to_numpy()
    return tr


S.simulate = simulate_with_px


# ------------------------------------------------------------------ in-sample MCPT
def insample(df, rth, cost_pts, n_perm, grouped, seed=0, log=print):
    F = S.build_features(df, rth)
    split = F["split"]
    cfg, real_pf, tr = optimize(F, cost_pts, 0, split)
    log(
        f"REAL in-sample best: {cfg}  PF(bar)={real_pf:.4f}  trades={len(tr)}  "
        f"expR={tr['pnl_r'].clip(-15, 15).mean():+.3f}  PF(trade)={pf(tr['pnl_r'].clip(-15, 15).to_numpy()):.3f}"
    )

    train = df.iloc[:split].reset_index(drop=True)
    groups = session_groups(train) if grouped else None
    better, perms = 1, []
    t0 = time.time()
    for i in range(1, n_perm):
        p = get_permutation(train, seed=seed + i, groups=groups)
        Fp = S.build_features(p, rth)
        Fp["split"] = len(p)
        _, ppf, _ = optimize(Fp, cost_pts, 0, len(p))
        if np.isfinite(ppf):
            perms.append(ppf)
            if ppf >= real_pf:
                better += 1
        if i % 25 == 0:
            el = time.time() - t0
            log(
                f"  {i}/{n_perm}  beaten {better - 1}x  p~{better / (i + 1):.3f}  "
                f"({el / i:.2f}s/perm, {(n_perm - i) * el / i / 60:.0f} min left)"
            )
    pval = better / n_perm
    return dict(
        test="insample",
        grouped=grouped,
        n_perm=n_perm,
        best=cfg,
        real_pf=real_pf,
        p_value=pval,
        perm_pf=perms,
        real_trades=len(tr),
    )


# ------------------------------------------------------------------ walk-forward
def walkforward(F, cost_pts, train_lookback, step, grid=GRID):
    """Re-optimise the exit grid on a trailing window, trade the next `step` bars with it.
    Blocks start flat, which is free for us: the strategy is flat at every RTH close."""
    n = F["n"]
    out = np.zeros(n)
    picks = []
    i = train_lookback
    while i < n:
        j = min(i + step, n)
        cfg, _, _ = optimize(F, cost_pts, i - train_lookback, i, grid)
        if cfg is not None:
            _, _, b = objective(F, cfg, cost_pts, i, j)
            out[i:j] = b
            picks.append((i, cfg))
        i = j
    return out, picks


def wf_mcpt(df, rth, cost_pts, n_perm, grouped, train_lookback, step, seed=0, log=print):
    F = S.build_features(df, rth)
    real_b, picks = walkforward(F, cost_pts, train_lookback, step)
    real_pf = pf(real_b[train_lookback:])
    log(
        f"REAL walk-forward PF(bar) = {real_pf:.4f} over {len(real_b) - train_lookback} bars, "
        f"{len(picks)} re-optimisations"
    )
    log("  picks: " + ", ".join(f"{c['exit'][:3]}{c['stop_atr']}/{c['tp_atr']}" for _, c in picks))

    groups = session_groups(df) if grouped else None
    better, perms = 1, []
    t0 = time.time()
    for i in range(1, n_perm):
        p = get_permutation(df, start_index=train_lookback, seed=seed + i, groups=groups)
        Fp = S.build_features(p, rth)
        b, _ = walkforward(Fp, cost_pts, train_lookback, step)
        ppf = pf(b[train_lookback:])
        if np.isfinite(ppf):
            perms.append(ppf)
            if ppf >= real_pf:
                better += 1
        if i % 5 == 0:
            el = time.time() - t0
            log(
                f"  {i}/{n_perm}  beaten {better - 1}x  p~{better / (i + 1):.3f}  "
                f"({el / i:.1f}s/perm, {(n_perm - i) * el / i / 60:.0f} min left)"
            )
    return dict(
        test="walkforward",
        grouped=grouped,
        n_perm=n_perm,
        real_pf=real_pf,
        p_value=better / n_perm,
        perm_pf=perms,
        picks=[[int(i), c] for i, c in picks],
    )


# ------------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["insample", "walkforward"])
    ap.add_argument("--csv", default=str(DATA_ROOT / "MNQ_OHLCV" / "MNQ_10m_full_session_ohlcv.csv"))
    ap.add_argument("--point-value", type=float, default=2.0)
    ap.add_argument("--cost-usd", type=float, default=5.0)
    ap.add_argument("--tz", default="America/Chicago")
    ap.add_argument("--rth", default="08:30-15:00")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--ungrouped", action="store_true", help="his plain global shuffle")
    ap.add_argument("--train-bars", type=int, default=52000, help="walk-forward trailing window")
    ap.add_argument("--step", type=int, default=2600, help="walk-forward re-opt interval")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = S.load_data(a.csv, a.tz)
    cost_pts = a.cost_usd / a.point_value
    print(
        f"{a.csv}: {len(df)} bars {df['dt'].iloc[0]} -> {df['dt'].iloc[-1]}, "
        f"cost {cost_pts:.2f} pts/round turn, grid = {len(GRID)} cells, "
        f"{'GROUPED' if not a.ungrouped else 'UNGROUPED (his plain version)'}"
    )
    if a.mode == "insample":
        res = insample(df, a.rth, cost_pts, a.perms, not a.ungrouped)
    else:
        res = wf_mcpt(df, a.rth, cost_pts, a.perms, not a.ungrouped, a.train_bars, a.step)
    q = np.percentile(res["perm_pf"], [50, 90, 95, 99]) if res["perm_pf"] else [np.nan] * 4
    print(
        f"\n{'=' * 70}\n{res['test'].upper()} MCPT  p-value = {res['p_value']:.4f}   "
        f"real PF {res['real_pf']:.4f}   permutation PF: median {q[0]:.3f}, "
        f"90th {q[1]:.3f}, 95th {q[2]:.3f}, 99th {q[3]:.3f}\n{'=' * 70}"
    )
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
