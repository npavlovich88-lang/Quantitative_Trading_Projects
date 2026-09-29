#!/usr/bin/env python3
"""Is the measured mean reversion (a) microstructure and (b) worth more than a round turn?

    python scripts/fade_economics.py

Hypothesis 11 established that MES and MNQ mean-revert intraday at every horizon tested,
p = 0.0033. That settles the DIRECTION of the structure. It says nothing about whether the
structure is tradeable, and two things stand between the two claims.

THE ARTIFACT: BID-ASK BOUNCE

Negative autocorrelation is what a bid-ask spread produces even in a series with no information
in it at all -- price alternates between bid and ask, and successive returns anti-correlate.
Roll (1984) gives the size: rho_1 = -s^2 / (4 * var(r)). If the measured rho_1 is the same order
as that, the "edge" IS the spread and cannot be traded, because crossing it is what you pay.

THE ARITHMETIC: EDGE VERSUS COST

VR(2) = 1 + rho_1, so the variance ratios already measured give the one-lag autocorrelation
directly. Fading a one-bar move earns |rho_1| * E|r| per trade before costs, with E|r| =
sigma * sqrt(2/pi) for a roughly symmetric return. That number goes straight beside the measured
round turn.

WHAT THIS IS NOT

It is a napkin calculation on in-sample autocorrelation. There is no exit rule, no risk
management, no permutation test on a traded version and no out-of-sample look. It says where to
point a strategy, not whether one works.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant import strategy as S
from quant.costs import MEASURED_SPREAD_POINTS, CostModel
from quant.data_splits import DATA, make_split

STATS = pathlib.Path("out/variance_ratio/statistics.npz")


def main() -> int:
    if not STATS.exists():
        print("run scripts/variance_ratio_survey.py first")
        return 1
    z = np.load(STATS)
    real, sigma = z["real"], z["sigma"]
    syms = [str(x) for x in z["symbols"]]
    tfs = [str(x) for x in z["timeframes"]]

    print("=" * 104)
    print("  FADE ECONOMICS -- is the mean reversion real, and is it bigger than the round turn?")
    print("=" * 104)
    print(
        f"  {'sym tf':<9}{'RTH bars':>10}{'rho_1':>9}{'SE':>8}{'bounce':>10}{'vs bounce':>11}"
        f"{'gross':>8}{'cost':>7}{'NET':>8}{'NET@CI':>9}"
    )
    for si, sym in enumerate(syms):
        cost = CostModel.for_prop(sym, "average", contracts=1).round_turn_points
        spread = MEASURED_SPREAD_POINTS[sym]
        for ti, tf in enumerate(tfs):
            vr2 = real[si, ti, 0, 0]
            if not np.isfinite(vr2):
                continue
            df = S.load_data(DATA[sym].format(tf=tf), "America/Chicago")
            d = df["dt"].dt.tz_localize(None).to_numpy()
            df = df.iloc[: make_split(sym, tf, d).train.stop]
            mins = df["dt"].dt.hour.to_numpy() * 60 + df["dt"].dt.minute.to_numpy()
            n = int(((mins >= 510) & (mins < 900)).sum())

            rho, sg = vr2 - 1.0, sigma[si, ti]
            se = 1.0 / np.sqrt(n)
            bounce = -(spread**2) / (4 * sg**2)
            gross = abs(rho) * sg * np.sqrt(2 / np.pi)
            hi = rho + 1.96 * se  # conservative end
            gross_ci = abs(hi) * sg * np.sqrt(2 / np.pi) if hi < 0 else 0.0
            print(
                f"  {sym} {tf:<5}{n:>10,}{rho:>9.4f}{se:>8.4f}{bounce:>10.5f}"
                f"{abs(rho / bounce):>10.0f}x{gross:>8.2f}{cost:>7.2f}"
                f"{gross - cost:>+8.2f}{gross_ci - cost:>+9.2f}"
            )
    print()
    print("  rho_1 = VR(2) - 1.  bounce = Roll's bid-ask estimate; 'vs bounce' is how many times")
    print("  larger the measured autocorrelation is than the spread can explain.")
    print("  gross = |rho_1| * E|r|.  NET@CI uses the conservative end of a 95% interval.")
    print()
    print("  Read: bounce accounts for the whole effect at 1m and essentially none of it from 3m")
    print("  up. The edge clears the round turn only at 30m and 60m, and survives the")
    print("  conservative end of its own confidence interval only at 60m.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
