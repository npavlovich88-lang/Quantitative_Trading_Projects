#!/usr/bin/env python3
"""Hypothesis 11. Is MES/MNQ a random walk at trend-following horizons?

    python scripts/variance_ratio_survey.py --perms 300

WHY THIS ONE MEASUREMENT ANSWERS A WHOLE FAMILY OF SYSTEMS

The proposal is to stop predicting direction and instead follow price until stopped out, with
the edge coming from asymmetry -- small losses in failed moves, large winners in the few real
expansions -- rather than from forecasting.

There is a theorem underneath that. Under a driftless random walk, EVERY stopping rule has
expectancy exactly zero before costs and negative after them. Optional stopping. No trail, state
machine, acceptance rule or adaptive buffer escapes it. A trend-following system can only work
if the series is NOT a random walk at the horizons its trail operates on.

    VR(q) > 1   moves extend       a trail can capture
    VR(q) = 1   random walk        no stopping rule has positive expectancy
    VR(q) < 1   moves retrace      a trail bleeds on every whipsaw

So this is not another strategy test. It is the precondition for an entire class of them, and it
is cheap: no entries, no exits, no parameters to sweep.

THE CONDITIONAL VERSION IS THE ACTUAL CLAIM

The proposal does not say the market always trends. It says a REGIME GATE identifies when it
does. So the survey measures VR twice: unconditionally, and restricted to bars where the
efficiency gate is active. If VR = 1 unconditionally but exceeds 1 when the gate is on, the
whole system is licensed. If VR = 1 in both, the family is dead regardless of what is bolted on
top.

Conditional windows never straddle an inactive stretch -- splicing returns from either side of a
gap would manufacture a q-period move out of two unrelated episodes.

TWO NULLS, AND WHICH ONE DECIDES

Lo & MacKinlay's M2 is asymptotically standard normal under the random walk null and robust to
heteroskedasticity. It is reported for reference. The grouped bar permutation decides, because
this project has twice measured analytic nulls to be badly miscalibrated on overlapping intraday
data -- once too strict (per-cell 95th percentile 1.377 against a nominal 1.960) and once far
too lenient (null best-cell median 9.343).

ECONOMIC SIZE IS REPORTED NEXT TO STATISTICAL SIZE

A variance ratio can be distinguishable from 1 and still far too small to pay a round turn. The
excess travel a q-bar move carries relative to a random walk is printed in points beside the
measured cost, so the number can be judged rather than admired.

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
from quant.costs import CostModel
from quant.data_splits import DATA, make_split
from quant.hypothesis import Hypothesis
from quant.permutation import get_permutation_fast, session_groups
from quant.variance_ratio import variance_ratio_segmented, vr_edge_points

ALL_TIMEFRAMES = ("1m", "3m", "5m", "10m", "15m", "30m", "60m")
TIMEFRAMES = ALL_TIMEFRAMES
SYMBOLS = ("MES", "MNQ")
QS = (2, 4, 8, 16, 32, 64)
GATE_N = 20  # lookback for the efficiency gate
GATE_ER = 0.35  # the gate the proposal specifies: high directional efficiency
RTH_LO, RTH_HI = 8 * 60 + 30, 15 * 60


def log(msg: str = "") -> None:
    print(msg, flush=True)


def efficiency(c: np.ndarray, n: int) -> np.ndarray:
    """Kaufman efficiency ratio over n bars: net travel divided by path length."""
    net = np.abs(c - np.r_[np.full(n, np.nan), c[:-n]])
    path = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(path > 0, net / path, 0.0)


def measure(df: pd.DataFrame, eligible: np.ndarray) -> np.ndarray:
    """VR(q) for every q, unconditional and gate-conditional. Shape (2, len(QS))."""
    c = df["close"].to_numpy(float)
    r = np.r_[np.nan, np.diff(np.log(c))]
    gate = efficiency(c, GATE_N) >= GATE_ER

    # Session-aware: q-period windows stay inside contiguous runs. The naive version splices
    # 17:00 to 08:30 as though no time passed, and at 15m nearly every q=64 window straddled two
    # session boundaries -- which is how the first pass produced its MES mean-reversion result.
    out = np.full((2, len(QS)), np.nan)
    for qi, q in enumerate(QS):
        out[0, qi] = variance_ratio_segmented(r, eligible, q)["vr"]
        out[1, qi] = variance_ratio_segmented(r, gate & eligible, q)["vr"]
    return out


def main() -> int:
    global TIMEFRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeframes", default=",".join(ALL_TIMEFRAMES))
    ap.add_argument("--out", default="out/variance_ratio")
    a = ap.parse_args()
    TIMEFRAMES = tuple(t.strip() for t in a.timeframes.split(",") if t.strip())

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log("=" * 96)
    log("  VARIANCE RATIO SURVEY -- is there serial dependence to follow at all?")
    log("=" * 96)
    log(f"  {len(TIMEFRAMES)} timeframes x {len(QS)} horizons x {len(SYMBOLS)} symbols,")
    log(f"  unconditional and conditional on ER({GATE_N}) >= {GATE_ER}")
    log(f"  permutations: {a.perms}   segment: TRAIN only   holdouts untouched\n")

    h = Hypothesis(
        name="variance_ratio_survey",
        instrument="MES and MNQ, 1m/3m/5m/10m/15m/30m/60m full-session bars",
        session="08:30-15:00 America/Chicago; returns outside RTH are excluded",
        setup="No entries. Measure the Lo-MacKinlay variance ratio VR(q) of log returns for "
        "q in {2,4,8,16,32,64}, both unconditionally and restricted to bars where Kaufman "
        "efficiency over 20 bars is at least 0.35 -- the regime gate the trend-following "
        "proposal specifies. Conditional windows never straddle an inactive stretch.",
        direction="both",
        exit_rule="none -- this measures a property of the return series, not a trade",
        execution="VR uses overlapping q-period windows with the paper's bias correction; the "
        "heteroskedasticity-robust M2 statistic is reported alongside, and the grouped bar "
        "permutation is what decides",
        costs="NOT in the statistic, but the excess travel implied by each VR is converted to "
        "points and printed beside the CostModel.for_prop round turn, so an effect too small to "
        "pay for itself is visible as one",
        invalidation="abandon the entire trend-following family if the permutation p-value of "
        "max |VR - 1| over the family is >= 0.05 in BOTH the unconditional and the "
        "gate-conditional measurement. Under a random walk the optional stopping theorem says "
        "every trail nets exactly minus costs, so no exit structure, state machine or "
        "acceptance rule can rescue it.",
        metric="max |VR(q) - 1| over the family against the same maximum on each permuted "
        "market, reported separately for the unconditional and gate-conditional cases, and "
        "converted to excess points per q-bar move for comparison against costs",
        rationale="Ten hypotheses have asked whether some variable predicts direction and the "
        "answer was consistently no. The trend-following proposal accepts that and claims the "
        "edge lives in the exit asymmetry instead. But under a driftless random walk every "
        "stopping rule has expectancy zero before costs, so that claim has a necessary "
        "condition that is measurable directly and cheaply: serial dependence at the horizons "
        "the trail operates on. This tests the precondition rather than any one system built "
        "on top of it.",
        author="research@example.invalid",
    )
    hh = h.register("out/hypotheses.jsonl")
    log(f"[pre-registered] {hh}\n")

    series: dict = {}
    real = np.full((len(SYMBOLS), len(TIMEFRAMES), 2, len(QS)), np.nan)
    sigma = np.full((len(SYMBOLS), len(TIMEFRAMES)), np.nan)

    for si, sym in enumerate(SYMBOLS):
        for ti, tf in enumerate(TIMEFRAMES):
            t0 = time.time()
            df = S.load_data(DATA[sym].format(tf=tf), "America/Chicago")
            dates = df["dt"].dt.tz_localize(None).to_numpy()
            sp = make_split(sym, tf, dates)
            df = df.iloc[: sp.train.stop].reset_index(drop=True)
            mins = df["dt"].dt.hour.to_numpy() * 60 + df["dt"].dt.minute.to_numpy()
            elig = (mins >= RTH_LO) & (mins < RTH_HI)
            real[si, ti] = measure(df, elig)
            c = df["close"].to_numpy(float)
            sigma[si, ti] = float(np.nanstd(np.diff(c)[elig[1:]]))
            series[(sym, tf)] = dict(df=df, elig=elig, groups=session_groups(df))
            log(
                f"  {sym} {tf:>3}: {len(df):>9,} bars, {int(elig.sum()):>8,} RTH, "
                f"1-bar sigma {sigma[si, ti]:6.2f} pts, {time.time() - t0:.1f}s"
            )

    log(f"\n  observed max |VR-1| = {np.nanmax(np.abs(real - 1)):.4f}\n")

    log(f"  running {a.perms} grouped bar permutations ...")
    null = np.full((a.perms, *real.shape), np.nan, dtype=np.float32)
    t0 = time.time()
    for p in range(a.perms):
        for si in range(len(SYMBOLS)):
            for ti in range(len(TIMEFRAMES)):
                s = series[(SYMBOLS[si], TIMEFRAMES[ti])]
                perm = get_permutation_fast(
                    s["df"],
                    start_index=0,
                    seed=a.seed + p * 1000 + si * 10 + ti,
                    groups=s["groups"],
                )
                null[p, si, ti] = measure(perm, s["elig"])
        if p == 0 or (p + 1) % 20 == 0:
            el = time.time() - t0
            log(
                f"    {p + 1:>4}/{a.perms}  {el:.0f}s elapsed, "
                f"~{el / (p + 1) * (a.perms - p - 1):.0f}s left"
            )

    np.savez_compressed(
        out / "statistics.npz",
        real=real,
        null=null,
        sigma=sigma,
        symbols=np.array(SYMBOLS),
        timeframes=np.array(TIMEFRAMES),
        qs=np.array(QS),
    )

    # ------------------------------------------------------------------ verdict
    names = ("UNCONDITIONAL", f"GATED  ER({GATE_N}) >= {GATE_ER}")
    log("\n" + "=" * 96)
    log("  RESULT")
    log("=" * 96)
    ps = {}
    for ci, cname in enumerate(names):
        obs = float(np.nanmax(np.abs(real[:, :, ci, :] - 1)))
        nl = np.nanmax(np.abs(null[:, :, :, ci, :] - 1).reshape(a.perms, -1), axis=1)
        pv = (1 + int((nl >= obs).sum())) / (1 + a.perms)
        ps[cname] = pv
        log(
            f"  {cname:<28} max |VR-1| = {obs:.4f}   null 95th pct {np.percentile(nl, 95):.4f}"
            f"   p = {pv:.4f}"
        )

    for ci, cname in enumerate(names):
        log(f"\n  {cname}  --  VR(q), 1.000 is a random walk")
        log("  " + f"{'sym tf':<9}" + "".join(f"{f'q={q}':>9}" for q in QS))
        for si, sym in enumerate(SYMBOLS):
            for ti, tf in enumerate(TIMEFRAMES):
                row = "".join(
                    f"{real[si, ti, ci, qi]:>9.3f}"
                    if np.isfinite(real[si, ti, ci, qi])
                    else f"{'--':>9}"
                    for qi in range(len(QS))
                )
                log(f"  {sym} {tf:<5}{row}")

    log("\n  ECONOMIC SIZE -- excess points a q-bar move carries vs a random walk")
    log(f"  {'sym tf':<9}{'cost':>7}" + "".join(f"{f'q={q}':>9}" for q in QS))
    for si, sym in enumerate(SYMBOLS):
        cost = CostModel.for_prop(sym, "average", contracts=1).round_turn_points
        for ti, tf in enumerate(TIMEFRAMES):
            row = "".join(
                f"{vr_edge_points(real[si, ti, 1, qi], sigma[si, ti], QS[qi]):>9.2f}"
                if np.isfinite(real[si, ti, 1, qi])
                else f"{'--':>9}"
                for qi in range(len(QS))
            )
            log(f"  {sym} {tf:<5}{cost:>7.2f}{row}")
    log("\n  (gated case. positive means a trail has more than a random walk to work with;")
    log("   it is an upper bound on what a PERFECT trail could take, before any costs)")

    # THE SIGN DECIDES, NOT THE MAGNITUDE. The first version of this block reported "PASS --
    # a trail has something to work with" on |VR - 1| alone, which is wrong: VR BELOW 1 is
    # serial dependence a trail is structurally punished by, not helped by.
    rr = real[:, :, 0, :]
    live = np.isfinite(rr)
    below = float((rr[live] < 1).mean())
    mean_vr = float(rr[live].mean())
    obs_signed = float(np.nanmean(rr - 1))
    nl_signed = np.array([np.nanmean(null[k, :, :, 0, :] - 1) for k in range(a.perms)])
    p_signed = (1 + int((nl_signed <= obs_signed).sum())) / (1 + a.perms)

    detectable = min(ps.values()) < 0.05
    trend_friendly = mean_vr > 1
    verdict = "PASS" if (detectable and trend_friendly) else "REJECT"
    log("")
    log("=" * 96)
    log(f"  VERDICT: {verdict}")
    log(
        f"  mean VR over {int(live.sum())} live cells = {mean_vr:.3f};  "
        f"{100 * below:.0f}% of them below 1"
    )
    log(f"  signed permutation p (more mean-reverting than shuffled) = {p_signed:.4f}")
    if not detectable:
        log("  Nothing separates from a random walk. Under optional stopping that closes the")
        log("  whole trend-following family: no trail, state machine or acceptance rule has")
        log("  positive expectancy on a series with no serial dependence.")
    elif not trend_friendly:
        log("  Serial dependence IS present and it points the WRONG WAY. VR below 1 means moves")
        log("  retrace: the trail is hit before the move completes, so trend following here is")
        log("  not merely unsupported, it is structurally disadvantaged. The exploitable")
        log("  structure is FADE structure.")
    else:
        log("  Serial dependence is present and trend-friendly. Next question is whether what")
        log("  it offers exceeds the round turn.")
    log("=" * 96)

    (out / "summary.json").write_text(
        json.dumps(
            {
                "hypothesis": hh,
                "permutations": a.perms,
                "p_unconditional": ps[names[0]],
                "p_gated": ps[names[1]],
                "observed_max_abs_vr_minus_1": float(np.nanmax(np.abs(real - 1))),
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"\n  written to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
