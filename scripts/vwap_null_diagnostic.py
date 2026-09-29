#!/usr/bin/env python3
"""Why the band race on noise does not equal 1/(k+1), and whether the null is stable.

Gambler's ruin for a driftless walk starting AT the -k sigma level, with the VWAP k sigma above
and the -(k+1) band 1 sigma below, gives P(reach the VWAP first) = 1/(k+1). Simulation gives
0.43 at k=2 rather than 0.33, so one of the assumptions is false. The candidate is the starting
point: the event fires when the bar LOW reaches the level, but the forward scan resumes at the
next bar, so the walk actually restarts from the touch bar CLOSE, which sits above the level by
however far the bar wicked. That shortens the trip to the mean and lengthens the trip to the
extension.

If that is the whole story, then measuring the race from the close and re-expressing the two
distances in sigma units should reproduce the ruin formula with the corrected distances.
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
    print("=" * 92)
    print("  WHERE THE RACE ACTUALLY STARTS FROM")
    print("=" * 92)
    print(
        f"  {'band':>5}{'events':>8}{'close vs level':>16}{'d_mean':>9}{'d_ext':>8}"
        f"{'ruin':>8}{'pA':>8}"
    )
    for k in (1, 2, 3, 4):
        allc, adm, ade, apa, n = [], [], [], [], 0
        for seed in range(6):
            df = walk(40000, seed=100 + seed)
            gid = anchor_ids(df["dt"], "ny")
            mu, sg = vwap_bands(df, gid)
            hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
            ev = first_touches(mu, sg, hi, lo, gid, k, -1)
            ev = ev[(ev + HOR) < len(df)]
            if len(ev) < 30:
                continue
            m, s = mu[ev], sg[ev]
            level = m - k * s
            # how far above the touched level the bar actually closed, in sigma
            over = (cl[ev] - level) / s
            d_mean = (m - cl[ev]) / s  # trip to the VWAP, from the close
            d_ext = (cl[ev] - (m - (k + 1) * s)) / s  # trip to the extension, from the close
            allc.append(over.mean())
            adm.append(d_mean.mean())
            ade.append(d_ext.mean())
            r = race(df, mu, sg, ev, k, -1, horizon=HOR)
            apa.append(r["pA"])
            n += r["nA"]
        dm, de = np.mean(adm), np.mean(ade)
        print(
            f"  {-k:>5}{n:>8}{np.mean(allc):>+16.2f}{dm:>9.2f}{de:>8.2f}"
            f"{de / (dm + de):>8.3f}{np.mean(apa):>8.3f}"
        )
    print()
    print("  'close vs level' is how far past the level the touch bar closed back, in sigma.")
    print("  'ruin' is d_ext / (d_mean + d_ext), the gambler's ruin probability using the")
    print("  ACTUAL distances from the close. If that column tracks pA, the null is explained")
    print("  and it is geometry plus the wick, with nothing unaccounted for.\n")

    print("=" * 92)
    print("  IS THE NULL STABLE ACROSS SEEDS?  (a drifting null would make every p meaningless)")
    print("=" * 92)
    print(f"  {'band':>5}{'mean pA':>10}{'sd':>8}{'min':>8}{'max':>8}{'seeds':>7}")
    for k in (1, 2, 3, 4):
        v = []
        for seed in range(12):
            df = walk(40000, seed=300 + seed)
            gid = anchor_ids(df["dt"], "ny")
            mu, sg = vwap_bands(df, gid)
            ev = first_touches(
                mu, sg, df["high"].to_numpy(float), df["low"].to_numpy(float), gid, k, -1
            )
            r = race(df, mu, sg, ev, k, -1, horizon=HOR)
            if r["nA"] > 30:
                v.append(r["pA"])
        print(
            f"  {-k:>5}{np.mean(v):>10.3f}{np.std(v):>8.3f}"
            f"{np.min(v):>8.3f}{np.max(v):>8.3f}{len(v):>7}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
