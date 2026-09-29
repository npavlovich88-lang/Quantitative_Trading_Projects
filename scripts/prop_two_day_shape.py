#!/usr/bin/env python3
"""What sizing and reward-to-risk actually maximises a TWO-DAY pass?

    python scripts/prop_two_day_shape.py

THE ARITHMETIC THAT DRIVES EVERYTHING

A two-day pass needs the two days to total the $3,000 target, so a winning day must clear about
$1,500. With one trade a day that means a WIN of $1,500, hence a stop of R = 1500/rr. The lower
the reward-to-risk, the bigger the stop has to be -- and the stop is bounded by the $2,000
cushion, not by preference.

Measured separately: the two-day pass rate is almost exactly w**(2K), where w is the win rate
and K the trades per day. It is simply the probability of winning EVERY trade for two days.
That has two consequences, and both run against the usual advice:

  * fewer trades a day is strictly better. At 1:1 and a 60% win rate, one trade a day passes
    35.5% of the time in two days, two trades 12.8%, three trades 4.4%. Each extra trade is
    another coin that has to land.
  * a HIGHER win rate beats a higher reward-to-risk, so the optimum sits at LOW rr -- the
    opposite of the trend for the max payout, which needs rr >= 1:1 to compound at all.

WHERE IT STOPS

The floor is not the win rate, it is the cushion. Below about 0.75:1 the stop needed to make
$1,500 on a win exceeds $2,000, and the real-time maximum loss limit then closes the account on
the adverse excursion of trades that were going to WIN. That is what collapses the 0.50:1 row
from an apparent w**2 of 0.64 to a measured 26.4%.
"""

import sys

sys.path.insert(0, "src")
import numpy as np

from quant.prop import RULES, sim_eval

NB, NSIM = 20000, 30000
BASE = RULES["TOPSTEP_COMBINE_50K_NOACT_DLL1000"]


def run(edge, R, rr, K=1):
    w = (edge + 1.0) / (rr + 1.0)
    if not (0 < w < 0.999):
        return None
    r = np.random.default_rng(42)
    x = np.where(r.random((NB, K)) < w, rr * R, -R)
    x = x - (x.mean() - edge * R)
    B = (x, np.abs(x), np.where(x > 0, r.uniform(0, R, (NB, K)), R), np.ones((NB, K), bool))
    st, ed = sim_eval(B, 1, BASE, NSIM, np.random.default_rng(7), max_days=400, mode="open")
    pas = st == 1
    return dict(w=w, p=100 * pas.mean(), d2=100 * float((pas & (ed <= 2)).sum()) / NSIM)


print("  WHY LOW RR CANNOT DO 2 DAYS.  One trade a day. To total $3,000 in two days a WIN")
print("  must be >= $1,500, so R = 1500/rr. But a LOSS is R, against a $2,000 cushion.")
print()
print(
    f"  {'RR':>7}{'R needed':>10}{'loss vs cushion':>17}{'win%@+.2R':>11}"
    f"{'<=2d @0':>9}{'<=2d @+.2R':>12}{'overall @+.2R':>15}"
)
for rr in (0.50, 0.75, 0.85, 1.00, 1.25, 1.50, 2.00):
    R = 1550.0 / rr  # a win just clears the $1,500 a day needed
    note = "BUSTS on 1 loss" if R >= 2000 else f"{100 * R / 2000:.0f}% of cushion"
    a = run(0.0, R, rr)
    b = run(0.20, R, rr)
    if a is None or b is None:
        print(f"  {rr:>6.2f}:1{R:>10.0f}{note:>17}{'impossible':>11}")
        continue
    print(
        f"  {rr:>6.2f}:1{R:>10.0f}{note:>17}{100 * b['w']:>10.1f}%"
        f"{a['d2']:>8.1f}%{b['d2']:>11.1f}%{b['p']:>14.1f}%"
    )
print()
print("  Below about 0.85:1 the stop needed to make $1,500 on a win is bigger than the whole")
print("  $2,000 cushion, so a single loss ends the account before a second day exists.")
