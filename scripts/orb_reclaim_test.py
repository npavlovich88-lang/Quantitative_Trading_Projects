#!/usr/bin/env python3
"""Hypothesis 13. Opening-range break, retest, 2m EMA reclaim. Full ladder.

    python scripts/orb_reclaim_test.py --perms 200

THE RULE, as given: mark the 30-minute opening range; wait for a 10m candle to CLOSE outside it;
drop to 2m with a 20 EMA; when price closes on the wrong side of the EMA while back near the
broken level, and then reclaims by closing through the EMA, enter; stop at the swing extreme of
the retest; target the pre-market high. Mirror for shorts.

WHY THIS ONE IS DIFFERENT FROM HYPOTHESES 1-12

All twelve were moving averages, or variables that turned out to be moving averages under
another name. This is session structure: an opening range, a break, a retest of the broken
level, and a target at a real price rather than an ATR multiple. The 20 EMA times the trigger;
it is not the signal.

It also has a variable reward, which matters given what was measured earlier: fixed WIDE targets
are reached LESS often than a coin flip on this data (0.051 against 0.167 at 1:5 on MES 5m). A
structural target is a different bet from a fixed multiple, and the measured RR here has a
median of 1.81 with a p90 of 6.01.

THE FAMILY

One genuinely free parameter -- the tolerance for "near the ORB high" -- swept over a declared
three-point grid, times two symbols. Nine cells is a small family by this project's standards,
which keeps the noise ceiling low. The statistic is net points per session, and the null is the
grouped bar permutation recomputed end to end: the permuted market gets its own opening range,
its own pre-market high and its own EMA, which is the right null for a rule built on levels.

MES IS THE CROSS-MARKET CHECK, NOT AN EXTRA CHANCE

The rule was specified for NQ. MES is included because identical split dates across two
instruments is the confirmation test this project uses, and because MNQ train holds only 421
usable sessions -- the 396-day data hole ends MNQ train in 2023-05. MES has roughly four times
the sessions and four regimes.

SEGMENTS

TRAIN ONLY. Neither holdout is touched and neither validate segment is read.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant import strategy as S
from quant.costs import POINT_VALUE, CostModel
from quant.data_splits import DATA, make_split
from quant.hypothesis import Hypothesis
from quant.permutation import get_permutation_fast, session_groups
from quant.strategies.opening_range_reclaim import NEAR_ATR, find_trades

SYMBOLS = ("MNQ", "MES")


def log(m: str = "") -> None:
    print(m, flush=True)


def evaluate(df: pd.DataFrame, cost: float, near: float) -> dict:
    tr = find_trades(df, near_atr=near)
    if not tr:
        return dict(n=0, net=0.0, per_session=0.0, win=np.nan, meanR=np.nan, tgt=np.nan)
    d = pd.DataFrame([vars(x) for x in tr])
    pts = (d["exit"] - d["entry"]) * d["side"] - cost
    n_sess = d["session"].nunique()
    return dict(
        n=len(d),
        net=float(pts.sum()),
        per_session=float(pts.sum() / max(n_sess, 1)),
        win=float((pts > 0).mean()),
        meanR=float(d["r_multiple"].mean()),
        tgt=float((d["reason"] == "target").mean()),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/orb_reclaim")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 92)
    log("  OPENING-RANGE BREAK, RETEST, 2m EMA RECLAIM")
    log("=" * 92)
    log(
        f"  family: {len(NEAR_ATR)} tolerances x {len(SYMBOLS)} symbols = "
        f"{len(NEAR_ATR) * len(SYMBOLS)} cells"
    )
    log(f"  permutations: {a.perms}   segment: TRAIN only   holdouts untouched\n")

    h = Hypothesis(
        name="opening_range_reclaim",
        instrument="MNQ (specified) and MES (cross-market check), 1m bars aggregated to 2m/10m",
        session="30-minute opening range from 08:30 America/Chicago; entries 09:00-15:00; "
        "pre-market high taken from the session open at 17:00 to 08:30",
        setup="A 10m bar must CLOSE outside the 30-minute opening range. Then on 2m bars with a "
        "20 EMA of close: price closes on the wrong side of the EMA while its extreme is within "
        "`near` x ATR(14) of the broken opening-range level, and then closes back through the "
        "EMA in the direction of the break. That reclaim bar's close is the entry.",
        direction="both",
        exit_rule="stop at the swing extreme from the retest bar through the entry bar; target "
        "at the pre-market high (long) or low (short); flat at the RTH close. Stop assumed first "
        "when stop and target both fall inside one 2m bar.",
        execution="every input is known at the close of the bar that uses it: the 10m break arms "
        "only after that bar closes, ATR(14) is shifted one bar, and the exit scan starts on the "
        "bar AFTER entry. A target already behind the entry voids the setup.",
        costs=f"CostModel.for_prop per symbol, average firm, 1 contract: "
        f"MNQ {CostModel.for_prop('MNQ', 'average', contracts=1).round_turn_points:.3f} pts, "
        f"MES {CostModel.for_prop('MES', 'average', contracts=1).round_turn_points:.3f} pts",
        invalidation="abandon if the family-corrected permutation p-value of the best cell is "
        ">= 0.05. A pass carried by one tolerance on one symbol goes to a single pre-committed "
        "validate look, not to a parameter tweak.",
        metric="net points per session after costs, best cell over the 6-cell family, against "
        "the same maximum on each permuted market; train segment only",
        rationale="Twelve prior hypotheses tested moving averages and variables that reduced to "
        "moving averages. This is session structure with a structural target: a level that a "
        "prior session actually printed, retested after a break. The measured reward is variable "
        "with a long right tail (RR median 1.81, p90 6.01), which is a different bet from the "
        "fixed wide targets already measured to be reached LESS often than a coin flip. The "
        "median hold is 12 minutes, inside the stated 5-15 minute window.",
        author="research@example.invalid",
    )
    hh = h.register("out/hypotheses.jsonl")
    log(f"[pre-registered] {hh}\n")

    store, real = {}, {}
    for sym in SYMBOLS:
        cost = CostModel.for_prop(sym, "average", contracts=1).round_turn_points
        df = S.load_data(DATA[sym].format(tf="1m"), "America/Chicago")
        d0 = df["dt"].dt.tz_localize(None).to_numpy()
        df = df.iloc[: make_split(sym, "1m", d0).train.stop].reset_index(drop=True)
        store[sym] = (df, cost, session_groups(df))
        log(f"  {sym}: {len(df):,} 1m train bars")
        for near in NEAR_ATR:
            r = evaluate(df, cost, near)
            real[(sym, near)] = r
            log(
                f"    near={near:.2f}  {r['n']:>4} trades  win {100 * r['win']:>3.0f}%  "
                f"meanR {r['meanR']:>+5.2f}  target {100 * r['tgt']:>3.0f}%  "
                f"net {r['net']:>+7.0f} pts = ${r['net'] * POINT_VALUE[sym]:>+7,.0f}  "
                f"per session {r['per_session']:>+6.2f}"
            )

    best_key = max(real, key=lambda k: real[k]["per_session"])
    obs = real[best_key]["per_session"]
    log(f"\n  best cell: {best_key[0]} near={best_key[1]:.2f}  {obs:+.3f} pts/session\n")

    log(f"  running {a.perms} grouped bar permutations (full recompute per draw) ...")
    nulls = np.full(a.perms, -np.inf)
    t0 = time.time()
    for p in range(a.perms):
        best = -np.inf
        for sym in SYMBOLS:
            df, cost, g = store[sym]
            pf = get_permutation_fast(
                df, start_index=0, seed=a.seed + p * 101 + hash(sym) % 97, groups=g
            )
            for near in NEAR_ATR:
                best = max(best, evaluate(pf, cost, near)["per_session"])
        nulls[p] = best
        if p == 0 or (p + 1) % 10 == 0:
            el = time.time() - t0
            log(
                f"    {p + 1:>4}/{a.perms}  null best {np.nanmax(nulls[: p + 1]):+.3f}   "
                f"{el:.0f}s elapsed, ~{el / (p + 1) * (a.perms - p - 1):.0f}s left"
            )

    pv = (1 + int((nulls >= obs).sum())) / (1 + a.perms)
    log("\n" + "=" * 92)
    log("  RESULT")
    log("=" * 92)
    log(f"  observed best        {obs:+.3f} pts/session   ({best_key[0]} near={best_key[1]:.2f})")
    log(f"  null best  median    {np.median(nulls):+.3f}")
    log(f"             95th pct  {np.percentile(nulls, 95):+.3f}")
    log(f"             max       {nulls.max():+.3f}")
    log(f"  FAMILY-CORRECTED p = {pv:.4f}")

    log("\n  every cell, with its own uncorrected p:")
    log(f"  {'sym':<5}{'near':>6}{'trades':>8}{'per session':>13}{'p_uncorr':>10}")
    rows = []
    for (sym, near), r in sorted(real.items(), key=lambda kv: -kv[1]["per_session"]):
        pc = (1 + int((nulls >= r["per_session"]).sum())) / (1 + a.perms)
        rows.append(dict(symbol=sym, near=near, **r, p_vs_family_null=pc))
        log(f"  {sym:<5}{near:>6.2f}{r['n']:>8}{r['per_session']:>+13.3f}{pc:>10.4f}")
    pd.DataFrame(rows).to_csv(out / "cells.csv", index=False)
    np.save(out / "null.npy", nulls)
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                hypothesis=hh,
                permutations=a.perms,
                observed_best=obs,
                best_cell=f"{best_key[0]} near={best_key[1]}",
                null_median=float(np.median(nulls)),
                null_p95=float(np.percentile(nulls, 95)),
                p_family=pv,
                verdict="PASS" if pv < 0.05 else "REJECT",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"\n  VERDICT: {'PASS' if pv < 0.05 else 'REJECT'}")
    log(f"  written to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
