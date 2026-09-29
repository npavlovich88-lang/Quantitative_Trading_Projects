#!/usr/bin/env python3
"""How much edge does the prop framework actually need to be worth buying?

    python scripts/prop_edge_threshold.py

Follows the Youngblood replication (scripts/replicate_youngblood.py). That run showed the paper's
zero-edge result does not survive a cost including the spread and the 8-lot book walk: the ratio
falls to 0.561. This asks the obvious next question -- what DOES clear 1.00 -- and the answer is
much lower than "a good edge".

The setup is held fixed at the paper's: Topstep 50K no-activation with the DLL bought, 8 MNQ
contracts, one trade a day, symmetric 100-tick barriers, full cost of $20.00 a trade. Only the
barrier win rate moves.

TWO OBJECTIVES THAT MUST NOT BE CONFUSED

Earlier work here measured the edge needed to pass an evaluation QUICKLY: a 70-80% pass inside
2-4 days is unreachable at any sizing or edge, because over two days the pass rate is 20.5%
whether the edge is 0.00R or +0.50R -- variance swamps everything.

This measures something different: expected value over REPEATED attempts, with no deadline. The
fee is the premium on an option whose payoff is the payouts, so the question is whether the
premium is cheaper than the option. That objective is reachable, and the threshold is small.

WHY THE THRESHOLD SITS AT A LOSING STRATEGY

The breakeven cell is a 52% barrier win rate, which is -$4.00 a trade and -0.010R. Standalone
that strategy LOSES money. It clears the prop framework anyway, because the framework caps the
downside at the account fee while leaving the payouts uncapped. That is the paper's thesis, and
it is correct; what the paper got wrong was the location of the threshold, by charging fees
without the spread.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant.costs import POINT_VALUE, CostModel
from quant.prop import RULES, sim_eval, sim_funded

NB, NSIM = 20_000, 40_000
BARRIER, CONTRACTS = 50.0, 8  # per contract: 100 ticks x $0.50; sim_eval scales by nc
COST_ATTEMPT = 7878.30 / (31.01 + 56.31)  # $90.22, the paper's own lifecycle accounting
RULE = RULES["TOPSTEP_COMBINE_50K_NOACT_DLL1000"]
WIN_RATES = (0.4998, 0.51, 0.52, 0.53, 0.54, 0.55, 0.57, 0.60)


def pool(w: float, cost: float, seed: int = 42):
    r = np.random.default_rng(seed)
    x = np.where(r.random(NB) < w, BARRIER, -BARRIER) - cost
    mae = np.where(x > 0, r.uniform(0.0, BARRIER, NB), BARRIER)
    return (
        x.reshape(-1, 1),
        np.abs(x).reshape(-1, 1),
        mae.reshape(-1, 1),
        np.ones((NB, 1), bool),
    )


def evaluate(w: float, cost: float) -> dict:
    B = pool(w, cost)
    st, _ = sim_eval(B, CONTRACTS, RULE, NSIM, np.random.default_rng(7), max_days=260, mode="open")
    p = float((st == 1).mean())
    f = sim_funded(
        B,
        CONTRACTS,
        "TOPSTEP_XFA_50K",
        NSIM,
        np.random.default_rng(9),
        days=252,
        dll_purchased=True,
    )
    pay = float(f["receipts"].mean())
    return dict(
        w=w,
        p_pass=p,
        payout=pay,
        n_pay=float(f["n_pay"].mean()),
        alive=float(f["alive"].mean()),
        ev_trade=(BARRIER * (2 * w - 1) - cost) * CONTRACTS,
        ev_R=(BARRIER * (2 * w - 1) - cost) / BARRIER,
        fees_per_pass=COST_ATTEMPT / p if p > 0 else np.inf,
        ratio=p * pay / COST_ATTEMPT if p > 0 else 0.0,
        net=pay - (COST_ATTEMPT / p if p > 0 else np.inf),
    )


def main() -> int:
    cost = CostModel.for_prop("MNQ", "topstep", contracts=8).round_turn_points * POINT_VALUE["MNQ"]

    print("=" * 104)
    print(f"  THE CHAIN at zero edge, full cost (${cost * CONTRACTS:.2f} a trade for 8 MNQ)")
    print("=" * 104)
    z = evaluate(0.4998, cost)
    print(
        f"  P(pass) per attempt            {100 * z['p_pass']:.1f}%"
        f"   ->  {1 / z['p_pass']:.2f} attempts to make one funded account"
    )
    print(
        f"  fees to manufacture one pass   {1 / z['p_pass']:.2f} x ${COST_ATTEMPT:.2f}"
        f"  =  ${z['fees_per_pass']:,.2f}"
    )
    print(f"  that funded account then pays  {z['n_pay']:.2f} payouts totalling ${z['payout']:.2f}")
    print(f"  net per funded account         ${z['net']:+,.2f}")
    print(f"  survival of the funded account {100 * z['alive']:.1f}%  (it always dies)")
    print("\n  So the failure is UPSTREAM of the payout. Payouts do arrive -- about one per")
    print("  funded account. They just do not cover the fees spent manufacturing the account.")

    print("\n" + "=" * 104)
    print("  SO HOW MUCH EDGE DOES IT TAKE?   same rules, same sizing, same cost")
    print("=" * 104)
    print(
        f"  {'win rate':>9}{'edge/trade':>12}{'edge in R':>11}{'P(pass)':>9}"
        f"{'payout/pass':>13}{'fees/pass':>11}{'ratio':>8}{'net/pass':>11}"
    )
    rows = [evaluate(w, cost) for w in WIN_RATES]
    for r in rows:
        print(
            f"  {100 * r['w']:>8.2f}%{r['ev_trade']:>+12.2f}{r['ev_R']:>+11.3f}"
            f"{100 * r['p_pass']:>8.1f}%{r['payout']:>13.2f}{r['fees_per_pass']:>11.2f}"
            f"{r['ratio']:>8.3f}{r['net']:>+11.2f}"
        )

    lo = max((r for r in rows if r["ratio"] < 1.0), key=lambda r: r["w"], default=None)
    hi = min((r for r in rows if r["ratio"] >= 1.0), key=lambda r: r["w"], default=None)
    if lo and hi:
        f = (1.0 - lo["ratio"]) / (hi["ratio"] - lo["ratio"])
        wb = lo["w"] + f * (hi["w"] - lo["w"])
        eb = lo["ev_R"] + f * (hi["ev_R"] - lo["ev_R"])
        print(f"\n  BREAKEVEN is at a {100 * wb:.2f}% barrier win rate, an edge of {eb:+.3f} R")
        print(f"  -- that is {100 * (wb - 0.50):+.2f} percentage points over a coin, and it is")
        print("  STILL A LOSING STRATEGY standalone. The framework pays for it because the")
        print("  downside is capped at the fee while the payouts are not.")

    print("\n  Scale check: 'edge in R' here is a 1:1 barrier, so +0.10R means a 55% win rate.")
    print("  The ORB rule measured +0.10R on MNQ and negative on MES, but at a MEDIAN RR of")
    print("  1.81 rather than 1:1, so the two are not the same object and cannot be read across.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
