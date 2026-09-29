#!/usr/bin/env python3
"""Is the stated goal reachable: 70-80% pass per eval, then the MAX payout, on any firm?

    python scripts/prop_spec_feasibility.py

THE SPEC BEING TESTED, as stated

  1. every evaluation bought passes with 70-80% probability -- no deadline this time
  2. whatever risk-reward the strategy needs, including reward smaller than risk
  3. once funded, reach the MAXIMUM payout at least once, ideally several times
  4. on almost any prop firm, so nothing tuned to one rule set

Each clause is measured separately, because they do not stand or fall together.

WHAT "MAXIMUM PAYOUT" MEANS HERE

Not "a payout". A payout is capped at the lesser of half the balance and the firm's ceiling, so
the ceiling only BINDS when the balance is at least twice it. On the Topstep 50K with the daily
loss limit bought, the Consistency ceiling is $6,000, which needs a $12,000 balance -- from a
funded account that starts at $0 with a $2,000 floor. sim_funded now counts those separately as
n_capped, so "reached the max payout" is measured rather than assumed.

THE CEILING THAT CLAUSE 1 HAS TO BEAT

For a driftless trader with a FIXED floor, optional stopping gives P(pass) = L/(T+L), which is
2,000/5,000 = 40% on both the Topstep and Lucid 50K. Both firms actually TRAIL the floor, which
is strictly worse -- this project measures 22-24% for Topstep at zero edge. So 70-80% is not
merely above the no-edge outcome, it is above the no-edge CEILING, and clause 1 therefore
requires real positive expectancy no matter what clause 2 does. That is the tension the run
below quantifies.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant.prop import RULES, sim_eval, sim_funded

NB, NSIM = 20_000, 30_000
FIRMS = (
    ("TOPSTEP", "TOPSTEP_COMBINE_50K_NOACT_DLL1000", "TOPSTEP_XFA_50K", "open", True),
    ("LUCID", "LUCIDFLEX_EVAL_50K", "LUCIDFLEX_FUNDED_50K", "closed", False),
)


def pool(edge_R: float, R: float, rr: float = 1.0, seed: int = 42):
    """One trade a day. Reward rr*R on a win, R on a loss, win rate set so the expectancy is
    exactly edge_R per trade. Returns dollars PER CONTRACT with nc=1 used throughout."""
    w = (edge_R + 1.0) / (rr + 1.0)
    if not (0.0 < w < 1.0):
        return None
    r = np.random.default_rng(seed)
    x = np.where(r.random(NB) < w, rr * R, -R)
    x = x - (x.mean() - edge_R * R)  # pin the sample mean so sampling noise cannot move the edge
    mae = np.where(x > 0, r.uniform(0.0, R, NB), R)
    return (
        x.reshape(-1, 1),
        np.abs(x).reshape(-1, 1),
        mae.reshape(-1, 1),
        np.ones((NB, 1), bool),
    ), w


def run(
    firm, edge_R: float, R: float, rr: float = 1.0, days: int = 252, withdraw_at: float = 0.0
) -> dict:
    name, ekey, fkey, mode, dll = firm
    got = pool(edge_R, R, rr)
    if got is None:
        return {}
    B, w = got
    st, ed = sim_eval(B, 1, RULES[ekey], NSIM, np.random.default_rng(7), max_days=400, mode=mode)
    p = float((st == 1).mean())
    f = sim_funded(
        B,
        1,
        fkey,
        NSIM,
        np.random.default_rng(9),
        days=days,
        mode=mode,
        dll_purchased=dll,
        withdraw_at=withdraw_at,
    )
    return dict(
        firm=name,
        edge_R=edge_R,
        R=R,
        rr=rr,
        win=w,
        p_pass=p,
        med_days=float(np.median(ed[st == 1])) if (st == 1).any() else np.nan,
        p_cap1=float((f["n_capped"] >= 1).mean()),
        p_cap2=float((f["n_capped"] >= 2).mean()),
        n_pay=float(f["n_pay"].mean()),
        receipts=float(f["receipts"].mean()),
        alive=float(f["alive"].mean()),
    )


def main() -> int:
    print("=" * 106)
    print("  CLAUSE 1   what edge does a 70-80% PASS need?   R = $200 a trade, 1:1")
    print("=" * 106)
    print(f"  {'edge/trade':>11}{'win rate':>10}", end="")
    for n, *_ in FIRMS:
        print(f"{n + ' pass':>15}{n + ' med d':>14}", end="")
    print()
    for e in (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        line = f"  {e:>+11.2f}R"
        wr = None
        for firm in FIRMS:
            r = run(firm, e, 200.0)
            wr = r["win"]
            line += f"{100 * r['p_pass']:>14.1f}%{r['med_days']:>14.0f}"
        print(f"  {e:>+9.2f}R{100 * wr:>9.1f}%" + line[len(f"  {e:>+11.2f}R") :])

    print("\n" + "=" * 106)
    print("  CLAUSE 2   does the RISK-REWARD SHAPE matter, holding expectancy fixed at +0.10R?")
    print("=" * 106)
    print(f"  {'reward:risk':>12}{'win rate':>10}", end="")
    for n, *_ in FIRMS:
        print(f"{n + ' pass':>15}{n + ' cap1':>14}", end="")
    print()
    for rr in (0.33, 0.50, 1.00, 2.00, 3.00):
        line = ""
        wr = None
        for firm in FIRMS:
            r = run(firm, 0.10, 200.0, rr)
            wr = r["win"]
            line += f"{100 * r['p_pass']:>14.1f}%{100 * r['p_cap1']:>13.1f}%"
        print(f"  {rr:>11.2f}:1{100 * wr:>9.1f}%" + line)
    print("\n  Expectancy is pinned, so any movement across these rows is the SHAPE alone.")

    print("\n" + "=" * 106)
    print("  CLAUSE 3   reaching the MAX payout: the cap must BIND, not just be eligible")
    print("=" * 106)
    print("  Topstep 50K + DLL: Consistency cap $6,000, so it needs a $12,000 balance from $0.")
    print("  Lucid 50K Flex:    cap $2,000,          so it needs a  $4,000 balance from $0.\n")
    print(f"  {'edge/trade':>11}", end="")
    for n, *_ in FIRMS:
        print(f"{n + ' payouts':>16}{n + ' max x1':>15}{n + ' max x2':>15}", end="")
    print()
    for e in (0.00, 0.10, 0.20, 0.30, 0.40, 0.60):
        line = f"  {e:>+10.2f}R"
        for firm in FIRMS:
            r = run(firm, e, 200.0, days=252)
            line += f"{r['n_pay']:>16.2f}{100 * r['p_cap1']:>14.1f}%{100 * r['p_cap2']:>14.1f}%"
        print(line)
    print("\n  One funded year. 'max x1' is the share of funded accounts that ever draw a CAPPED")
    print("  payout; 'max x2' the share that do it twice.")

    print("\n" + "=" * 106)
    print("  CLAUSE 4   does one edge carry across both firms, or is it rule-specific?")
    print("=" * 106)
    print(
        f"  {'edge/trade':>11}{'TOPSTEP pass':>15}{'LUCID pass':>13}{'gap':>9}"
        f"{'TOPSTEP max x1':>17}{'LUCID max x1':>15}{'gap':>9}"
    )
    for e in (0.10, 0.20, 0.30, 0.40):
        a = run(FIRMS[0], e, 200.0)
        b = run(FIRMS[1], e, 200.0)
        print(
            f"  {e:>+10.2f}R{100 * a['p_pass']:>14.1f}%{100 * b['p_pass']:>12.1f}%"
            f"{100 * (b['p_pass'] - a['p_pass']):>+9.1f}"
            f"{100 * a['p_cap1']:>16.1f}%{100 * b['p_cap1']:>14.1f}%"
            f"{100 * (b['p_cap1'] - a['p_cap1']):>+9.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
