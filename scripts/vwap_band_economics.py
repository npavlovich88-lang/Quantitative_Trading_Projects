#!/usr/bin/env python3
"""What the Q1 band races are worth in points, on BOTH sides of the trade.

    python scripts/vwap_band_economics.py

Reads out/vwap_band_race/cells.csv. Runs no permutation: the p-values are already fixed there.

WHY THIS EXISTS

Q1 reports pA, the share of races that reached the VWAP before the next band out, and pA_be, the
rate needed to break even trading TOWARD the mean. Several cells came in well BELOW their null,
which means price extended rather than reverted -- and for those cells the mean-reversion trade is
the wrong side. The tradeable direction is the extension, whose win rate is 1 - pA and whose
reward and risk are simply swapped:

    revert    win d_mean sigma with probability pA,        lose d_ext sigma
    extend    win d_ext  sigma with probability 1 - pA,    lose d_mean sigma

Both are priced here at one contract, average firm, with the round-turn cost charged in points.
Unresolved races -- neither level reached inside one RTH day -- are excluded from pA by
construction, so the expectancy below is PER RESOLVED RACE and the `unres` column says how much
of the sample that silently drops.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant.costs import POINT_VALUE, CostModel

SRC = pathlib.Path("out/vwap_band_race/cells.csv")
MAX_UNRES = 0.35  # see the note in main(); above this a cell cannot support any claim


def main() -> int:
    raw = pd.read_csv(SRC)
    # A race that reaches NEITHER level inside one RTH day is excluded from pA by construction,
    # so a cell with most races unresolved reports a probability measured on a selected remnant.
    # The continuous anchor is the offender: its sigma is several hundred points, so its bands
    # cannot be raced intraday at all, and it produced apparent expectancies of +646 points off
    # samples where 79% of races never resolved. Gate it out rather than interpret it.
    d = raw[raw["unres"] <= MAX_UNRES].copy()
    print(
        f"  dropped {len(raw) - len(d)} of {len(raw)} cells with more than "
        f"{100 * MAX_UNRES:.0f}% of races unresolved inside the horizon"
    )
    by = raw[raw["unres"] > MAX_UNRES]["anchor"].value_counts().to_dict()
    print(f"  they are, by anchor: {by}")
    print()
    d["cost"] = [
        CostModel.for_prop(s, "average", contracts=1).round_turn_points for s in d["symbol"]
    ]
    d["signed"] = d["sign"] * d["band"]

    # reward and risk in POINTS, measured from the touch bar close
    d["win_rev"] = d["d_mean"] * d["sigma"]
    d["win_ext"] = d["d_ext"] * d["sigma"]
    d["ev_rev"] = d["pA"] * d["win_rev"] - (1 - d["pA"]) * d["win_ext"] - d["cost"]
    d["ev_ext"] = (1 - d["pA"]) * d["win_ext"] - d["pA"] * d["win_rev"] - d["cost"]
    d["be_ext"] = (d["win_rev"] + d["cost"]) / (d["win_rev"] + d["win_ext"])
    d["p_ext"] = 1 - d["pA"]
    d["ev_best"] = d[["ev_rev", "ev_ext"]].max(axis=1)
    d["side"] = np.where(d["ev_ext"] > d["ev_rev"], "extend", "revert")
    d["dollars"] = d["ev_best"] * [POINT_VALUE[s] for s in d["symbol"]]

    print("=" * 104)
    print("  THE BAND PROFILE   (t signed: + is reversion above the null, - is extension)")
    print("=" * 104)
    print(
        f"  {'band':>6}{'cells':>7}{'races':>8}{'pA':>8}{'null':>8}{'mean t':>9}"
        f"{'t>+2':>6}{'t<-2':>6}{'sigma pt':>10}{'best EV pt':>12}{'side':>8}"
    )
    for sg in sorted(d["signed"].unique()):
        s = d[d["signed"] == sg]
        print(
            f"  {sg:>+6}{len(s):>7}{int(s['nA'].sum()):>8}{s['pA'].mean():>8.3f}"
            f"{s['null_mean'].mean():>8.3f}{s['t'].mean():>+9.2f}"
            f"{int((s['t'] > 2).sum()):>6}{int((s['t'] < -2).sum()):>6}"
            f"{s['sigma'].mean():>10.1f}{s['ev_best'].mean():>+12.2f}"
            f"{s['side'].mode().iloc[0]:>8}"
        )

    print("\n" + "=" * 104)
    print("  DOES ANY CELL MAKE MONEY ON EITHER SIDE?   one contract, average firm, cost in pts")
    print("=" * 104)
    pos = d[d["ev_best"] > 0].sort_values("ev_best", ascending=False)
    print(f"  {len(pos)} of {len(d)} cells have positive expectancy per resolved race\n")
    if len(pos):
        print(
            f"  {'sym':<4}{'tf':<4}{'anchor':<11}{'band':>5}{'filt':>10}{'nA':>6}"
            f"{'side':>8}{'p':>7}{'be':>7}{'win':>7}{'risk':>7}{'EV pt':>8}"
            f"{'EV $':>8}{'unres':>7}{'t':>7}{'p_cell':>8}"
        )
        for _, r in pos.head(18).iterrows():
            ext = r["side"] == "extend"
            print(
                f"  {r['symbol']:<4}{r['tf']:<4}{r['anchor']:<11}{int(r['signed']):>+5}"
                f"{r['filter']:>10}{int(r['nA']):>6}{r['side']:>8}"
                f"{(r['p_ext'] if ext else r['pA']):>7.3f}"
                f"{(r['be_ext'] if ext else r['pA_be']):>7.3f}"
                f"{(r['win_ext'] if ext else r['win_rev']):>7.1f}"
                f"{(r['win_rev'] if ext else r['win_ext']):>7.1f}"
                f"{r['ev_best']:>+8.2f}{r['dollars']:>+8.0f}{r['unres']:>7.2f}"
                f"{r['t']:>+7.2f}{r['p_cell']:>8.4f}"
            )

    print("\n" + "=" * 104)
    print("  THE CELLS THAT BEAT THEIR NULL, AND WHETHER THAT SURVIVES COST")
    print("=" * 104)
    sig = d[d["p_cell"] < 0.05].copy()
    sig["dir"] = np.where(sig["t"] > 0, "revert", "extend")
    print(f"  {len(sig)} of {len(d)} cells at uncorrected p < 0.05")
    for dirn in ("revert", "extend"):
        s = sig[sig["dir"] == dirn]
        if s.empty:
            continue
        good = s[s["ev_best"] > 0]
        print(
            f"    {dirn:<7} {len(s):>3} cells, {int(s['nA'].sum()):>6} races, "
            f"mean |t| {s['t'].abs().mean():.2f}, "
            f"{len(good)} with positive EV, best {s['ev_best'].max():+.2f} pts"
        )
    print()
    print("  cross-market support for the extension reading, by band:")
    ext = d[(d["t"] < -2)]
    if len(ext):
        g = ext.groupby("signed").agg(
            cells=("t", "size"),
            syms=("symbol", "nunique"),
            tfs=("tf", "nunique"),
            races=("nA", "sum"),
            mean_t=("t", "mean"),
            mean_ev=("ev_best", "mean"),
        )
        print(g.to_string())
    else:
        print("    no cell reached t < -2")

    d.to_csv("out/vwap_band_race/economics.csv", index=False)
    print("\n  written to out/vwap_band_race/economics.csv\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
