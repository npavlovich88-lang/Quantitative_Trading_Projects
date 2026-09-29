#!/usr/bin/env python3
"""Can a Topstep 50K evaluation be passed in two days, and what does it cost to try?

    python scripts/prop_two_day_pass.py

WHY THIS RUN EXISTS

Two YouTube videos were put forward as making the case. Both are more careful than the genre
usually is, and they DISAGREE with each other on the thing that matters:

  * the first treats the account as a convex payoff -- downside capped at the challenge fee,
    upside realised -- and reports zero-expected-value pass rates of 37-40%, rising as the
    reward-to-risk falls, then claims about $8,600 net per account with no winning strategy;
  * the second walks 100 challenges to 6 payouts and shows the group LOSING money:
    100 x $150 = $15,000 of fees against 6 x $2,000 = $12,000 of payouts, a ratio of 0.80.

Both halves are checkable here. The first video's 37-40% brackets L/(T+L) = 40%, which is the
optional-stopping ceiling for a FIXED floor. Topstep's floor trails intraday in real time, which
is strictly worse, and this project measures 24% at zero edge with that floor modelled. The
second video names trailing drawdown as a major probability reducer, which is exactly the term
the first appears to have left out.

THE TWO-DAY QUESTION ITSELF

A two-day pass needs two days totalling the $3,000 target, so roughly $1,500 a day, against a
$2,000 cushion. That forces large size, and large size interacts with the consistency rule, whose
reading is still unresolved in this project:

    rising target   best day <= 55% of the profit target; exceeding it RAISES the target
    fixed cap       best day <= $1,650 outright

Both readings are run below, because they do not give the same answer -- at $1,650 a day one of
them says 35.3% and the other says the account cannot be passed at all.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant.prop import RULES, sim_eval

NB, NSIM = 20_000, 40_000
BASE = "TOPSTEP_COMBINE_50K_NOACT_DLL1000"


def pool(edge_R: float, R: float, rr: float = 1.0, seed: int = 42):
    w = (edge_R + 1.0) / (rr + 1.0)
    r = np.random.default_rng(seed)
    x = np.where(r.random(NB) < w, rr * R, -R)
    x = x - (x.mean() - edge_R * R)  # pin the sample mean so sampling noise cannot move the edge
    mae = np.where(x > 0, r.uniform(0.0, R, NB), R)
    return (
        x.reshape(-1, 1),
        np.abs(x).reshape(-1, 1),
        mae.reshape(-1, 1),
        np.ones((NB, 1), bool),
    )


def days(rule, B, mode: str = "open"):
    st, ed = sim_eval(B, 1, rule, NSIM, np.random.default_rng(7), max_days=400, mode=mode)
    pas = st == 1
    return pas, ed


def by_day(pas, ed, d: int) -> float:
    """Share of ALL attempts that had passed by day d."""
    return 100.0 * float((pas & (ed <= d)).sum()) / NSIM


def main() -> int:
    print("=" * 100)
    print("  HOW FAST CAN A TOPSTEP 50K BE PASSED?   target $3,000, MLL $2,000 trailing realtime")
    print("=" * 100)
    print(
        f"  {'R/trade':>9}{'edge':>8}{'P(pass)':>9}{'<=1d':>7}{'<=2d':>7}{'<=3d':>7}"
        f"{'<=5d':>7}{'<=10d':>7}{'med d':>7}"
    )
    for R in (400.0, 800.0, 1200.0, 1600.0, 2000.0):
        for e in (0.00, 0.20, 0.50):
            pas, ed = days(RULES[BASE], pool(e, R))
            med = float(np.median(ed[pas])) if pas.any() else np.nan
            print(
                f"  {R:>9.0f}{e:>+8.2f}{100 * pas.mean():>8.1f}%"
                f"{by_day(pas, ed, 1):>6.1f}%{by_day(pas, ed, 2):>6.1f}%"
                f"{by_day(pas, ed, 3):>6.1f}%{by_day(pas, ed, 5):>6.1f}%"
                f"{by_day(pas, ed, 10):>6.1f}%{med:>7.0f}"
            )
        print()

    print("=" * 100)
    print("  THE CONSISTENCY READING DECIDES WHETHER THE SIZE NEEDED IS EVEN LEGAL")
    print("=" * 100)
    print("  edge +0.20R, 1:1, one trade a day. A 2-day pass needs about $1,500 a day.\n")
    print(f"  {'R/trade':>9}{'rising-target reading':>26}{'fixed $1,650 cap':>24}")
    print(f"  {'':>9}{'pass':>10}{'<=2d':>8}{'<=3d':>8}{'pass':>11}{'<=2d':>7}{'<=3d':>7}")
    strict = dict(RULES[BASE], consistency=None, consistency_fixed=1650.0)
    for R in (1500.0, 1600.0, 1650.0, 1800.0, 2000.0):
        B = pool(0.20, R)
        out = []
        for rule in (RULES[BASE], strict):
            pas, ed = days(rule, B)
            out.append((100 * pas.mean(), by_day(pas, ed, 2), by_day(pas, ed, 3)))
        a, b = out
        print(
            f"  {R:>9.0f}{a[0]:>9.1f}%{a[1]:>7.1f}%{a[2]:>7.1f}%"
            f"{b[0]:>10.1f}%{b[1]:>6.1f}%{b[2]:>6.1f}%"
        )
    print("\n  Under the strict reading the window is a knife edge: each day must clear about")
    print("  $1,500 to total $3,000 in two, yet stay under $1,650 or the target moves away.")
    print("  That is a band roughly $150 wide, and it is the whole of the two-day strategy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
