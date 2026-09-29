#!/usr/bin/env python3
"""Validate the prop-firm simulator against the closed-form pass probability, and map the ridge.

    python scripts/prop_validate.py

THE REFERENCE RESULT

Villahermosa (SSRN 7445798, Proposition 1): for a driftless participant with a FIXED floor,
continuous paths and an unbounded horizon, optional stopping gives

    P(pass) = L / (T + L)

independent of volatility. Topstep 50K and Lucid 50K Flex both have L = 2,000 and T = 3,000,
so the ceiling for both is 2,000/5,000 = 40.0%. If the simulator does not reproduce that under
those assumptions, it is wrong and nothing built on it means anything.

TWO ERRORS TO AVOID WHEN TESTING THIS, both of which were made first time round

  1. `lock = -1e9` does NOT fix the floor, it removes it. The simulator computes
     floor = min(maxeod - mll, lock), so pinning the floor at -mll requires lock = -mll.
  2. A "driftless" pool must be DEMEANED. Drawing 6,000 samples at sd 200 leaves a sample mean
     with standard error 2.58, and bootstrapping inherits it. A drift of -1.56 dollars a day --
     0.08% of the cushion -- moved the measured pass rate from 40.6% to 35.8%. The barrier
     probability is extraordinarily sensitive to drift, which is exactly where costs live.

WHAT THE RIDGE SECTION IS FOR

Lim (SSRN 7178078) finds contract value concentrates on a narrow ridge near 2% daily volatility,
on a USD 100,000 account. That figure cannot be transplanted to a 50K account whose MLL is 2,000
-- 4% of equity against roughly 10% in that paper -- so the ridge is recomputed here for the
actual rule sets.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant.prop import RULES, sim_eval

NSIM = 20_000
CEILING = 2000.0 / (3000.0 + 2000.0)


def demeaned_pool(sd: float, nb: int = 8000, seed: int = 1):
    """A pool with EXACTLY zero mean, so the bootstrap carries no drift."""
    r = np.random.default_rng(seed)
    x = r.normal(0.0, sd, nb)
    x -= x.mean()
    return (
        x.reshape(-1, 1),
        np.abs(x).reshape(-1, 1),
        np.abs(r.normal(0, sd * 0.7, (nb, 1))),
        np.ones((nb, 1), bool),
    )


def ceiling_check() -> None:
    print("=" * 88)
    print(f"  CEILING CHECK   L=2000  T=3000  ->  L/(T+L) = {100 * CEILING:.2f}%")
    print("=" * 88)
    print(f"  {'daily sd':>9}{'% cushion':>11}{'max days':>10}{'FIXED floor':>14}{'TRAILING':>11}")
    for sd, md in ((100.0, 12000), (200.0, 8000), (400.0, 4000), (800.0, 2000)):
        B = demeaned_pool(sd)
        fixed = dict(
            RULES["LUCIDFLEX_EVAL_50K"],
            consistency=None,
            dll=None,
            lock=-2000.0,
            breach="eod_close",
        )
        trail = dict(RULES["LUCIDFLEX_EVAL_50K"], consistency=None, dll=None)
        a, _ = sim_eval(B, 1, fixed, NSIM, np.random.default_rng(7), max_days=md, mode="closed")
        b, _ = sim_eval(B, 1, trail, NSIM, np.random.default_rng(7), max_days=md, mode="closed")
        print(
            f"  {sd:>9.0f}{100 * sd / 2000:>10.1f}%{md:>10,}"
            f"{100 * (a == 1).mean():>13.2f}%{100 * (b == 1).mean():>10.2f}%"
        )
    print("\n  FIXED floor must sit on 40.0% plus a little barrier overshoot.")
    print("  TRAILING is the real rule and must sit strictly below it.\n")


def drift_sensitivity() -> None:
    print("=" * 88)
    print("  DRIFT SENSITIVITY -- why cost per trade dominates everything")
    print("=" * 88)
    sd = 200.0
    r = np.random.default_rng(1)
    base = r.normal(0.0, sd, 8000)
    base -= base.mean()
    print(f"  {'mean $/day':>12}{'closed form':>13}{'simulated':>11}{'vs driftless':>14}")
    rows: list[tuple[float, float, float]] = []
    for m in (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0):
        x = base + m
        B = (
            x.reshape(-1, 1),
            np.abs(x).reshape(-1, 1),
            np.abs(r.normal(0, sd * 0.7, (8000, 1))),
            np.ones((8000, 1), bool),
        )
        rule = dict(
            RULES["LUCIDFLEX_EVAL_50K"],
            consistency=None,
            dll=None,
            lock=-2000.0,
            breach="eod_close",
        )
        s, _ = sim_eval(B, 1, rule, NSIM, np.random.default_rng(7), max_days=20000, mode="closed")
        got = 100 * (s == 1).mean()
        if abs(m) < 1e-9:
            pred = 100 * CEILING
        else:
            k = 2 * m / sd**2
            pred = 100 * (1 - np.exp(-k * 2000)) / (1 - np.exp(-k * 5000))
        rows.append((m, pred, got))
    # The reference is the driftless row, which is not the first row, so the comparison column
    # can only be filled once every row is in hand.
    ref = next(g for m, _p, g in rows if abs(m) < 1e-9)
    for m, pred, got in rows:
        print(f"  {m:>+12.1f}{pred:>12.2f}%{got:>10.2f}%{got - ref:>+12.1f}pp")
    print()


def ridge() -> None:
    print("=" * 88)
    print("  OUR RISK-SIZING RIDGE -- zero edge throughout, so this isolates SIZING alone")
    print("=" * 88)
    nb = 8000
    r = np.random.default_rng(3)
    shape = np.where(r.random(nb) < 0.40, 1.5, -1.0)  # a fair 1.5:1 bracket at a 40% hit rate
    shape -= shape.mean()  # exactly zero edge
    mae = np.abs(r.normal(0, 0.8, nb))
    print(
        f"  {'R/trade':>9}{'daily sd':>10}{'% of 50K':>10}{'% cushion':>11}"
        f"{'TOPSTEP':>10}{'LUCID':>9}{'spread':>9}"
    )
    for R in (50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0):
        pnl = (shape * R).reshape(-1, 1)
        B = (pnl, np.abs(pnl), (mae * R).reshape(-1, 1), np.ones((nb, 1), bool))
        got = []
        for key, mode in (("TOPSTEP_COMBINE_50K", "open"), ("LUCIDFLEX_EVAL_50K", "closed")):
            s, _ = sim_eval(
                B, 1, RULES[key], NSIM, np.random.default_rng(11), max_days=400, mode=mode
            )
            got.append(100 * (s == 1).mean())
        print(
            f"  {R:>9.0f}{pnl.std():>10.0f}{100 * pnl.std() / 50000:>9.2f}%"
            f"{100 * pnl.std() / 2000:>10.1f}%{got[0]:>9.1f}%{got[1]:>8.1f}%"
            f"{got[1] - got[0]:>+8.1f}pp"
        )
    print("\n  Topstep checks the MLL in REAL TIME including unrealised P&L, so large size is")
    print("  punished there and not on Lucid's end-of-day check. That is the whole spread.")
    print("  Below roughly 6% of the cushion per trade the pass rate collapses -- the failure")
    print("  mode is running out of days, which is Lim's low-volatility edge of the ridge.\n")


def main() -> int:
    ceiling_check()
    drift_sensitivity()
    ridge()
    print("=" * 88)
    print("  EXTERNAL CROSS-CHECK")
    print("=" * 88)
    print("  Topstep discloses that 16.8% of evaluations initiated in 2025 were completed")
    print("  (Lim, SSRN 7184138). Our zero-edge simulation gives 22-24% for Topstep, so the")
    print("  real population sits BELOW a zero-edge mechanical participant -- consistent with")
    print("  cost drag and with the drift sensitivity above, and a reason to trust the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
