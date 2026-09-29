#!/usr/bin/env python3
"""What win rate and reward-to-risk actually satisfies the whole spec?

    python scripts/prop_shape_grid.py

THE SPEC

  1. 70-80% chance of passing each evaluation bought, no deadline
  2. once funded, reach the MAXIMUM payout, ideally repeatedly
  3. portable across firms

scripts/prop_spec_feasibility.py measured clauses 1 and 2 SEPARATELY and found that at a fixed
expectancy a low reward-to-risk shape passes far more often -- 98.7% at 0.33:1 against 39.5% at
3:1, both pinned at +0.10R. That conclusion was correct for the pass rate and WRONG for the spec
as a whole, because it never asked what the same shape does to the payout cap.

This grid asks both at once, and they pull in opposite directions:

  * PASSING is a barrier problem. Low reward-to-risk means low daily variance, and against a
    trailing floor with positive drift, low variance is unambiguously better. Low RR passes on
    less edge.
  * The MAX PAYOUT is a drift problem. Topstep's $6,000 Consistency cap only BINDS at a $12,000
    balance, from a funded account starting at $0. Reaching $12,000 inside a year needs annual
    drift, and a 0.25:1 strategy making $50 a win simply cannot generate it however high its win
    rate goes. It is 0.0% at every win rate tested, up to 94% and +0.155R.

So the shape that passes cheapest can never satisfy clause 2, and the spec is only satisfied in
the middle-to-high RR band where both clear at once.

PARAMETERS

R is the dollar risk per trade; cost is charged as a fraction of R on every trade, so the win
rates below are GROSS barrier win rates, the thing a backtest actually reports. Withdrawal
patience is 1.00 for the cap columns -- the cap can never bind under greedy withdrawal, which
scripts/prop_spec_feasibility.py establishes -- and 0.25 for the receipts column, which is where
total receipts peak.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant.prop import RULES, sim_eval, sim_funded

NB, NSIM = 20_000, 15_000
EVAL_RULE = "TOPSTEP_COMBINE_50K_NOACT_DLL1000"
RRS = (0.25, 0.33, 0.50, 1.00, 1.50, 2.00)
OFFSETS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.14)


def pool(w: float, rr: float, R: float, cost_frac: float, seed: int = 42):
    r = np.random.default_rng(seed)
    x = np.where(r.random(NB) < w, rr * R, -R) - cost_frac * R
    mae = np.where(x > 0, r.uniform(0.0, R, NB), R)
    return (
        x.reshape(-1, 1),
        np.abs(x).reshape(-1, 1),
        mae.reshape(-1, 1),
        np.ones((NB, 1), bool),
    )


def cell(w: float, rr: float, R: float, cost_frac: float, days: int = 252) -> dict:
    B = pool(w, rr, R, cost_frac)
    st, ed = sim_eval(
        B, 1, RULES[EVAL_RULE], NSIM, np.random.default_rng(7), max_days=400, mode="open"
    )
    p = float((st == 1).mean())
    f = sim_funded(
        B,
        1,
        "TOPSTEP_XFA_50K",
        NSIM,
        np.random.default_rng(9),
        days=days,
        dll_purchased=True,
        withdraw_at=1.0,
    )
    f25 = sim_funded(
        B,
        1,
        "TOPSTEP_XFA_50K",
        NSIM,
        np.random.default_rng(9),
        days=days,
        dll_purchased=True,
        withdraw_at=0.25,
    )
    return dict(
        w=w,
        rr=rr,
        edge=w * rr - (1 - w) - cost_frac,
        p_pass=p,
        med=float(np.median(ed[st == 1])) if (st == 1).any() else np.nan,
        cap1=float((f["n_capped"] >= 1).mean()),
        cap2=float((f["n_capped"] >= 2).mean()),
        rec25=float(f25["receipts"].mean()),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk", type=float, default=200.0)
    ap.add_argument("--cost-frac", type=float, default=0.02)
    a = ap.parse_args()

    print("=" * 96)
    print("  WIN RATE x REWARD-TO-RISK against the whole spec")
    print("=" * 96)
    print(f"  R = ${a.risk:.0f} risked a trade, cost {100 * a.cost_frac:.0f}% of R charged every")
    print("  trade, so win rates are GROSS. Topstep 50K + DLL, one funded year.")
    print("  cap columns use full withdrawal patience; receipts uses 25%, where receipts peak.\n")
    print(
        f"  {'RR':>7}{'win%':>7}{'edge/tr':>9}{'PASS':>8}{'med d':>7}"
        f"{'max x1':>8}{'max x2':>8}{'recpts':>9}"
    )
    rows = []
    for rr in RRS:
        w0 = 1.0 / (rr + 1.0)  # the win rate at which a costless bet breaks even
        for off in OFFSETS:
            w = round(w0 + off, 4)
            if w >= 0.985:
                continue
            c = cell(w, rr, a.risk, a.cost_frac)
            rows.append(c)
            print(
                f"  {rr:>6.2f}:1{100 * c['w']:>7.1f}{c['edge']:>+9.3f}{100 * c['p_pass']:>7.1f}%"
                f"{c['med']:>7.0f}{100 * c['cap1']:>7.1f}%{100 * c['cap2']:>7.1f}%"
                f"{c['rec25']:>9.0f}"
            )
        print()

    print("=" * 96)
    print("  THE TWO CLAUSES PULL APART")
    print("=" * 96)
    print(f"  {'RR':>7}{'cheapest edge for ~70% PASS':>30}{'cheapest edge for 50% max x1':>32}")
    for rr in RRS:
        s = [r for r in rows if r["rr"] == rr]
        pas = min((r for r in s if r["p_pass"] >= 0.70), key=lambda r: r["edge"], default=None)
        cap = min((r for r in s if r["cap1"] >= 0.50), key=lambda r: r["edge"], default=None)
        ps = f"{pas['edge']:+.3f}R at {100 * pas['w']:.1f}% win" if pas else "not reached"
        cs = f"{cap['edge']:+.3f}R at {100 * cap['w']:.1f}% win" if cap else "NEVER"
        print(f"  {rr:>6.2f}:1{ps:>30}{cs:>32}")
    print("\n  'NEVER' is not a sample-size artifact: at 0.25:1 the cap stays at 0.0% even at a")
    print("  94% win rate and +0.155R. A $50 win cannot compound to a $12,000 balance in a year.")

    print("\n" + "=" * 96)
    print("  CELLS THAT SATISFY THE WHOLE SPEC   pass >= 70% AND max payout twice >= 20%")
    print("=" * 96)
    ok = [r for r in rows if r["p_pass"] >= 0.70 and r["cap2"] >= 0.20]
    if not ok:
        print("  none in this grid")
    else:
        print(
            f"  {'RR':>7}{'win%':>7}{'edge/tr':>9}{'PASS':>8}{'med d':>7}"
            f"{'max x1':>8}{'max x2':>8}{'recpts':>9}"
        )
        for r in sorted(ok, key=lambda r: r["edge"]):
            print(
                f"  {r['rr']:>6.2f}:1{100 * r['w']:>7.1f}{r['edge']:>+9.3f}"
                f"{100 * r['p_pass']:>7.1f}%{r['med']:>7.0f}{100 * r['cap1']:>7.1f}%"
                f"{100 * r['cap2']:>7.1f}%{r['rec25']:>9.0f}"
            )
        cheap = min(ok, key=lambda r: r["edge"])
        print(
            f"\n  cheapest full-spec cell: {cheap['rr']:.2f}:1 at {100 * cheap['w']:.1f}% win, "
            f"{cheap['edge']:+.3f}R a trade, passing in a median {cheap['med']:.0f} days"
        )
    print("\n  For scale: the best cell ever measured in this project was +0.21R and it failed")
    print("  cross-market; the ORB rule was +0.10R on MNQ and negative on MES.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
