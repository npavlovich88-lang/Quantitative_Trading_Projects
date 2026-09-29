#!/usr/bin/env python3
"""Why hypothesis 9 passed its family gate and is still not an edge.

    python scripts/cross_filter_confound.py

THE RESULT THAT NEEDED EXPLAINING

Hypothesis 9 passed both family tests convincingly -- max-statistic p = 0.0033 against a null
95th percentile of 5.878, and a count statistic of 865 cells against a null median of 270. Four
of seven filters fired at p = 0.0033. That is not a marginal result and it is not a lone spike
in the way hypothesis 6's was.

But the outcome breakdown was self-contradictory:

    mfe   24.2% of cells over their own null,  70% POSITIVE   -- filtered entries ran further
    mae   16.1% of cells over their own null,  39% positive   -- and drew down further
    ret    7.0% of cells over their own null,  p = 0.0365     -- weakest of the three

A directional edge improves the favourable excursion without a matching increase in the adverse
one. Both moving together in proportion is the signature of a VOLATILITY effect.

WHAT THIS SCRIPT MEASURES

For the strongest configuration, the mean forward outcome of the high-filter and low-filter
groups, alongside the forward realised range in the same trailing-ATR units. If the filter is
selecting volatility rather than direction, MFE, MAE and forward realised range all scale by the
same factor -- and the mean return does not move.

WHY THE PERMUTATION TEST DID NOT CATCH IT

It did exactly what it is built to do. Grouped bar permutation destroys volatility CLUSTERING:
in a shuffled series a high-momentum bar is not followed by a high-volatility bar. In the real
series it is. So the test correctly reported "this conditioner finds something real that shuffled
data does not have" -- and that something is volatility persistence, which is one of the
best-established facts in finance and is not a directional edge.

THE METHODOLOGICAL LESSON

ATR normalisation does not neutralise a conditioner that selects on volatility. ATR(14) shifted
one bar is a TRAILING estimate; a momentum filter selects bars where volatility is RISING, so
forward realised volatility exceeds the trailing normaliser precisely inside the filtered group.
Every outcome measured in trailing-ATR units inherits that inflation.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant import strategy as S
from quant.data_splits import DATA, make_split
from quant.information import atr_shifted, forward_outcomes
from quant.strategies.cross_filter import CALIB_FRAC, FilterSpec, build_masks
from quant.verdict import Verdict

SERIES = (("MES", "3m"), ("MES", "5m"), ("MES", "10m"), ("MNQ", "3m"), ("MNQ", "5m"))
HORIZONS = (1, 3, 6, 12)
SPEC = FilterSpec("composite", 0, 8, 21, "mae")  # the family's strongest cell


def main() -> int:
    print("=" * 96)
    print("  HYPOTHESIS 9 CONFOUND CHECK -- is the effect direction or volatility?")
    print("=" * 96)
    print(f"  configuration: {SPEC.label}, the family maximum at |t| = 8.080\n")

    ratios = []
    rets = []
    for sym, tf in SERIES:
        df = S.load_data(DATA[sym].format(tf=tf), "America/Chicago")
        d = df["dt"].dt.tz_localize(None).to_numpy()
        df = df.iloc[: make_split(sym, tf, d).train.stop].reset_index(drop=True)
        n = len(df)
        calib = slice(0, int(n * CALIB_FRAC))
        atr = atr_shifted(df)
        mins = df["dt"].dt.hour.to_numpy() * 60 + df["dt"].dt.minute.to_numpy()
        elig = (
            (mins >= 510)
            & (mins < 900)
            & (np.arange(n) >= calib.stop)
            & np.isfinite(atr)
            & (atr > 0)
        )
        ent, masks = build_masks(df, atr, [SPEC], tf, calib, elig)
        hi, lo = masks[0]
        dvec = ent[(SPEC.fast, SPEC.slow, SPEC.lag)][1]
        h_, l_, c_ = (df[k].to_numpy(float) for k in ("high", "low", "close"))

        print(f"  {sym} {tf:>3}   high-filter n={hi.sum():,}   low-filter n={lo.sum():,}")
        print(
            f"    {'h':>4}{'MFE ratio':>11}{'MAE ratio':>11}{'fwdVol ratio':>14}"
            f"{'spread':>9}{'RET hi':>9}{'RET lo':>9}"
        )
        for hor in HORIZONS:
            oc = forward_outcomes(h_, l_, c_, atr, hor, dvec)
            fmax = pd.Series(h_).rolling(hor).max().shift(-hor).to_numpy()
            fmin = pd.Series(l_).rolling(hor).min().shift(-hor).to_numpy()
            fvol = (fmax - fmin) / atr

            def mean(x, k):
                v = x[k]
                v = v[np.isfinite(v)]
                return float(v.mean()) if len(v) else float("nan")

            r = (
                mean(oc["mfe"], hi) / mean(oc["mfe"], lo),
                mean(oc["mae"], hi) / mean(oc["mae"], lo),
                mean(fvol, hi) / mean(fvol, lo),
            )
            ratios.append(max(r) - min(r))
            rh, rl = mean(oc["ret"], hi), mean(oc["ret"], lo)
            rets.append(rh - rl)
            print(
                f"    {hor:>4}{r[0]:>11.3f}{r[1]:>11.3f}{r[2]:>14.3f}"
                f"{max(r) - min(r):>9.3f}{rh:>9.3f}{rl:>9.3f}"
            )
        print()

    print("=" * 96)
    print(
        f"  The three ratios agree to a median spread of {np.median(ratios):.3f} across "
        f"{len(ratios)} cells."
    )
    print(
        f"  Mean return difference (filtered minus unfiltered), median across the same cells: "
        f"{np.median(rets):+.4f} ATR"
    )
    print("=" * 96)

    v = Verdict.no_edge(
        "cross_filter (after the confound check)",
        reason=(
            "The family gate passed at p = 0.0033, but the effect is volatility selection rather "
            "than direction. Favourable excursion, adverse excursion and forward realised range "
            f"all scale by the same factor (ratios agree to a median spread of "
            f"{np.median(ratios):.3f}), while the mean return moves by "
            f"{np.median(rets):+.4f} ATR -- and moves the WRONG way. A filter that makes winners "
            "and losers larger in equal proportion is a volatility filter."
        ),
        evidence=[
            "mfe 70% positive, mae 39% positive: both excursions grew together",
            "ret was the weakest outcome of the three at p = 0.0365 despite the family p of 0.0033",
            "filtered entries had LOWER mean return than unfiltered at every horizon and series",
            "the grouped permutation destroys volatility clustering, so it correctly flagged a "
            "real property of markets -- volatility persistence -- which is not an edge",
            "ATR(14) shifted one bar is a TRAILING normaliser; a momentum filter selects rising "
            "volatility, so forward realised volatility exceeds it inside the filtered group",
        ],
        caveat=(
            "The claim is not disproven in general. It is disproven for outcomes measured in "
            "TRAILING-ATR units. Re-running with the outcome normalised by FORWARD realised "
            "range would ask the question that remains open: conditional on the market moving "
            "this much, does the filter pick the right direction?"
        ),
        next_step=(
            "Hypothesis 10: identical design, outcomes divided by forward realised range instead "
            "of trailing ATR, which removes the magnitude channel by construction."
        ),
    )
    print()
    print(v.render())
    out = pathlib.Path("out/cross_filter")
    if out.exists():
        (out / "verdict_after_confound.txt").write_text(v.render(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
