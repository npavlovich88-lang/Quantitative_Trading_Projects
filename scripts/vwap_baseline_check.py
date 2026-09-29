#!/usr/bin/env python3
"""What a driftless walk does at a VWAP band, in closed form and by simulation.

This exists because the intuition was wrong first time round. From a touch of -k sigma:

    distance to the VWAP            k sigma
    distance to the -(k+1) band     1 sigma

so the EXTENSION is the nearer target for every k >= 2, and gambler's ruin gives

    P(reach the VWAP first) = 1 / (k + 1)

which is 0.50 at k=1, 0.333 at k=2, 0.25 at k=3, 0.20 at k=4 -- not "most of the time".
Any reversion claim at -2 sigma has to beat one third, not one half.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tests.test_vwap_bands import walk

from quant.strategies.vwap_bands import anchor_ids, first_touches, race, vwap_bands

HOR = 78


def main() -> int:
    print("=" * 84)
    print("  DRIFTLESS BASELINE FOR THE BAND RACES   (pure Gaussian walk, real volume shape)")
    print("=" * 84)
    print(f"  {'band':>6}{'side':>7}{'events':>9}{'pA':>9}{'1/(k+1)':>10}{'pC':>9}{'unres':>9}")
    for k in (1, 2, 3, 4):
        for sign in (-1, 1):
            pas, pcs, ns, un = [], [], [], []
            for seed in range(6):
                df = walk(40000, seed=100 + seed)
                gid = anchor_ids(df["dt"], "ny")
                mu, sg = vwap_bands(df, gid)
                ev = first_touches(
                    mu, sg, df["high"].to_numpy(float), df["low"].to_numpy(float), gid, k, sign
                )
                r = race(df, mu, sg, ev, k, sign, horizon=HOR)
                if r["nA"] > 30:
                    pas.append(r["pA"])
                    ns.append(r["nA"])
                    un.append(r["unres"])
                if r["nC"] > 30:
                    pcs.append(r["pC"])
            if not pas:
                print(f"  {k:>6}{sign:>7}        too few events")
                continue
            print(
                f"  {k:>6}{sign:>7}{int(np.sum(ns)):>9}{np.mean(pas):>9.3f}"
                f"{1 / (k + 1):>10.3f}{(np.mean(pcs) if pcs else np.nan):>9.3f}"
                f"{np.mean(un):>9.3f}"
            )
    print()
    print("  pA must track 1/(k+1). That column IS the null, and it is the number a real")
    print("  reversion effect has to beat -- one third at the 2 sigma band, not one half.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
