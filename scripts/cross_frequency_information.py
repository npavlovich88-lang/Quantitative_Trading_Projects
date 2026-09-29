#!/usr/bin/env python3
"""Hypothesis 8. Does a cross work better when the pair has NOT been flipping repeatedly?

    python scripts/cross_frequency_information.py --perms 300

THE CLAIM, IN THE USER'S WORDS

    "we end up seeing candlesticks just cross the ema over and over not giving us real value"
    "we aren't crossing the emas over and over and we are still showing willingness to go lower"

Same cross, two regimes. The proposal is that recent CROSSING FREQUENCY separates them.

WHY THIS IS NOT HYPOTHESIS 3 AGAIN

Hypothesis 3 tested EMA separation and was rejected. Separation is a level at one instant;
this is a count over history, and the two are nearly uncorrelated (-0.05 to -0.07). The
distinction was the user's: "emas can be spread far apart even if the market is in a chopy
regieme". Measured against the other rejected variables, crossing frequency correlates -0.12 to
-0.18 with |displacement| and -0.01 to -0.04 with rotation count, while separation and
|displacement| correlate 0.58 to 0.80 with each other.

WHAT IS COMPARED

Only crosses are sampled -- one row per cross, which is how the rule would actually be traded.
Each cross is labelled by how many crosses occurred in the previous K bars (shifted, so it never
counts itself), and the low-frequency tercile is compared against the high-frequency tercile.
A positive t means the quiet-regime crosses did better.

THREE OUTCOMES, NOT ONE

Every test in this project so far compared MEAN forward return. A trade with a stop and a target
does not care about the mean; it cares whether the favourable excursion arrives before the
adverse one. So forward return, MFE and MAE are all in the family, all ATR-normalised,
direction-adjusted, and oriented so higher is better -- MAE negated. The question "does the move
give us a 5-15 minute run" is an MFE question.

SEGMENTS

TRAIN ONLY. Terciles from the first 20% of train, applied to the remaining 80%, which is the
only part evaluated. Neither holdout is touched and neither validate segment is read.
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
from quant.information import DEFAULT_HORIZONS, atr_shifted, evaluate_outcomes, family_pvalues
from quant.information import forward_outcomes as fwd_out
from quant.permutation import get_permutation_fast, session_groups
from quant.strategies.cross_frequency import CALIB_FRAC, EMA_PAIRS, build_masks, freq_family
from quant.verdict import from_information

HORIZONS = DEFAULT_HORIZONS
ALL_TIMEFRAMES = ("3m", "5m", "10m", "15m", "20m", "30m")
TIMEFRAMES = ALL_TIMEFRAMES
SYMBOLS = ("MES", "MNQ")
MIN_EVENTS = 100
RTH_LO, RTH_HI = 8 * 60 + 30, 15 * 60
SPECS = freq_family()
N_CELLS = len(SPECS) * len(HORIZONS) * len(TIMEFRAMES) * len(SYMBOLS)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def eval_series(df, atr, calib, eligible):
    """One evaluate call per (EMA pair, horizon): direction and outcomes both depend on them."""
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    out = np.full((len(SPECS), len(HORIZONS)), np.nan)
    ones = np.ones(len(c), bool)
    for f, s in EMA_PAIRS:
        idx = [i for i, sp in enumerate(SPECS) if (sp.fast, sp.slow) == (f, s)]
        sub = [SPECS[i] for i in idx]
        dirs, masks = build_masks(c, sub, calib, eligible)
        for hi, hor in enumerate(HORIZONS):
            oc = fwd_out(h, lo, c, atr, hor, dirs[(f, s)])
            for kind in ("ret", "mfe", "mae"):
                keep = [k for k, sp in enumerate(sub) if sp.outcome == kind]
                if not keep:
                    continue
                t, _d, _a, _b = evaluate_outcomes(
                    oc[kind], [masks[k] for k in keep], ones, MIN_EVENTS
                )
                for pos, k in enumerate(keep):
                    out[idx[k], hi] = t[pos]
    return out


def main() -> int:
    global TIMEFRAMES, N_CELLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeframes", default=",".join(ALL_TIMEFRAMES))
    ap.add_argument("--out", default="out/cross_frequency")
    a = ap.parse_args()
    TIMEFRAMES = tuple(t.strip() for t in a.timeframes.split(",") if t.strip())
    N_CELLS = len(SPECS) * len(HORIZONS) * len(TIMEFRAMES) * len(SYMBOLS)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log("=" * 96)
    log("  CROSSING FREQUENCY -- does a cross work better when the pair has been quiet?")
    log("=" * 96)
    log(
        f"  family: {len(SPECS)} configs x {len(HORIZONS)} horizons x {len(TIMEFRAMES)} "
        f"timeframes x {len(SYMBOLS)} symbols = {N_CELLS:,} cells"
    )
    log(f"  permutations: {a.perms}   segment: TRAIN only   holdouts untouched\n")

    h = Hypothesis(
        name="cross_frequency_regime",
        instrument="MES and MNQ, 3m/5m/10m/15m/20m/30m full-session bars",
        session="08:30-15:00 America/Chicago for the EVENT; the forward window may run past it",
        setup="Sample only EMA crosses (8/21, 10/29, 20/50, 50/200). Label each by the number "
        "of crosses in the previous K bars (K = 50, 100, 200), shifted so the cross never "
        "counts itself. Compare the low-frequency tercile against the high-frequency tercile.",
        direction="both",
        exit_rule="none -- three fixed-horizon outcomes are measured: forward return, maximum "
        "favourable excursion and maximum adverse excursion, at 1, 3, 6, 12, 26 and 78 bars.",
        execution="the cross and the frequency count are both known at the close of bar t; "
        "excursions look at bars t+1..t+h, so the entry bar's own range cannot fill anything; "
        "terciles come from the first 20% of train and apply to the remaining 80%",
        costs="NOT MODELLED: there are no trades. Effects are converted to points and compared "
        "against CostModel.for_prop so a real but untradeably small effect is visible as one.",
        invalidation="abandon if the family-corrected permutation p-value of the maximum |t| "
        "is >= 0.05. A nominal pass carried by a lone spike goes to one pre-committed validate "
        "look, never to a parameter tweak.",
        metric="max |t| over the family, plus the count of cells beating their own null 95th "
        "percentile, reported separately for return, MFE and MAE, train segment only",
        rationale="Separation was rejected as hypothesis 3, but separation is a level at one "
        "instant and cannot see history: a pair can be wide now and have flipped six times in "
        "the last hour. Crossing frequency is measured to be nearly orthogonal to separation "
        "(-0.05), to displacement (-0.15) and to rotation count (-0.02), while separation and "
        "displacement correlate 0.58 to 0.80 with each other -- so the two earlier rejections "
        "were largely one rejection and this is a genuinely different variable. The mechanism "
        "is that repeated flipping is direct evidence of a two-way auction, which is a property "
        "of the recent past rather than of the present bar.",
        author="research@example.invalid",
    )
    hh = h.register("out/hypotheses.jsonl")
    log(f"[pre-registered] {hh}\n")

    series: dict[tuple[str, str], dict] = {}
    real = np.full((len(SYMBOLS), len(TIMEFRAMES), len(SPECS), len(HORIZONS)), np.nan)

    for si, sym in enumerate(SYMBOLS):
        for ti, tf in enumerate(TIMEFRAMES):
            t0 = time.time()
            df = S.load_data(DATA[sym].format(tf=tf), "America/Chicago")
            dates = df["dt"].dt.tz_localize(None).to_numpy()
            sp = make_split(sym, tf, dates)
            df = df.iloc[: sp.train.stop].reset_index(drop=True)
            n = len(df)
            calib = slice(0, int(n * CALIB_FRAC))
            mins = df["dt"].dt.hour.to_numpy() * 60 + df["dt"].dt.minute.to_numpy()
            atr = atr_shifted(df)
            eligible = (
                (mins >= RTH_LO)
                & (mins < RTH_HI)
                & (np.arange(n) >= calib.stop)
                & np.isfinite(atr)
                & (atr > 0)
            )
            real[si, ti] = eval_series(df, atr, calib, eligible)
            series[(sym, tf)] = dict(
                df=df,
                calib=calib,
                eligible=eligible,
                groups=session_groups(df),
                atr_med=float(np.nanmedian(atr[calib.stop :])),
            )
            live = int(np.isfinite(real[si, ti]).sum())
            log(
                f"  {sym} {tf:>3}: {n:>7,} train bars, calib {calib.stop:,}, "
                f"{live:>3}/{real[si, ti].size} live cells, {time.time() - t0:.1f}s"
            )

    finite = np.isfinite(real)
    log(f"\n  {finite.sum():,} of {N_CELLS:,} cells live")
    idx = np.unravel_index(np.nanargmax(np.abs(real)), real.shape)
    log(
        f"  observed max |t| = {np.nanmax(np.abs(real)):.3f}  at  {SYMBOLS[idx[0]]} "
        f"{TIMEFRAMES[idx[1]]} {SPECS[idx[2]].label} h={HORIZONS[idx[3]]}\n"
    )

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
                null[p, si, ti] = eval_series(perm, atr_shifted(perm), s["calib"], s["eligible"])
        if p == 0 or (p + 1) % 20 == 0:
            el = time.time() - t0
            log(
                f"    {p + 1:>4}/{a.perms}  null max |t| {np.nanmax(np.abs(null[: p + 1])):.3f}   "
                f"{el:.0f}s elapsed, ~{el / (p + 1) * (a.perms - p - 1):.0f}s left"
            )

    np.savez_compressed(
        out / "statistics.npz",
        real=real,
        d=np.zeros_like(real),
        null=null,
        symbols=np.array(SYMBOLS),
        timeframes=np.array(TIMEFRAMES),
        horizons=np.array(HORIZONS),
        labels=np.array([s.label for s in SPECS]),
        families=np.array([s.family for s in SPECS]),
    )

    fp = family_pvalues(real, null)
    log("\n" + "=" * 96)
    log("  RESULT")
    log("=" * 96)
    log(f"  observed max |t| over {fp['live_cells']:,} live cells : {fp['observed_max_t']:.3f}")
    log(
        f"  null max |t|  median {fp['null_max_median']:.3f}   "
        f"95th pct {fp['null_max_p95']:.3f}   max {fp['null_max_max']:.3f}"
    )
    log(f"  PRIMARY   max-statistic p = {fp['p_max']:.4f}")
    log(
        f"  SECONDARY {fp['cells_over_own_null']} cells beat their own null 95th pct "
        f"vs null median {fp['null_count_median']:.0f}, p = {fp['p_count']:.4f}"
    )

    thr = fp["cell_threshold"]
    fam_of = np.array([s.family for s in SPECS])
    log("\n  BY OUTCOME -- return, run (MFE), and drawdown (MAE, negated so higher is better)")
    for name in ("ret", "mfe", "mae"):
        sel = fam_of == name
        mask = np.zeros_like(finite)
        mask[:, :, sel, :] = True
        m = finite & mask
        if not m.sum():
            continue
        cr = int((np.abs(real) > thr)[m].sum())
        cn = np.array([int((np.abs(null[p]) > thr)[m].sum()) for p in range(a.perms)])
        pv = (1 + int((cn >= cr).sum())) / (1 + a.perms)
        log(
            f"  {name:<6} cells {m.sum():>5}  over {cr:>4} ({100 * cr / m.sum():>5.1f}%)  "
            f"null med {np.median(cn):>5.0f}  p={pv:.4f}  "
            f"max|t| {np.nanmax(np.abs(np.where(m, real, np.nan))):.2f}  "
            f"{100 * (real[m] > 0).mean():>3.0f}% positive"
        )

    rows = []
    for si, sym in enumerate(SYMBOLS):
        cost = CostModel.for_prop(sym, "average", contracts=1).round_turn_points
        for ti, tf in enumerate(TIMEFRAMES):
            am = series[(sym, tf)]["atr_med"]
            for j, spec in enumerate(SPECS):
                for hi, hor in enumerate(HORIZONS):
                    if finite[si, ti, j, hi]:
                        rows.append(
                            dict(
                                symbol=sym,
                                tf=tf,
                                spec=spec.label,
                                outcome=spec.outcome,
                                horizon=hor,
                                t=float(real[si, ti, j, hi]),
                                p_uncorrected=float(fp["p_cell"][si, ti, j, hi]),
                                atr_pts=am,
                                cost_pts=cost,
                            )
                        )
    tab = pd.DataFrame(rows).sort_values("t", key=np.abs, ascending=False).reset_index(drop=True)
    tab.to_csv(out / "cells.csv", index=False)
    log("\n  TOP 15 CELLS BY |t|   (positive t = the QUIET regime did better)")
    log(f"  {'sym':<4}{'tf':<5}{'spec':<30}{'h':>4}{'t':>8}{'p_unc':>9}")
    for _, r in tab.head(15).iterrows():
        log(f"  {r.symbol:<4}{r.tf:<5}{r.spec:<30}{r.horizon:>4}{r.t:>8.2f}{r.p_uncorrected:>9.4f}")

    v = from_information(
        out.name,
        real,
        null,
        [s.label for s in SPECS],
        list(SYMBOLS),
        list(TIMEFRAMES),
        list(HORIZONS),
        fam_of,
    )
    log("")
    log(v.render())
    (out / "verdict.txt").write_text(v.render(), encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps(v.as_dict(), indent=2), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps({k: x for k, x in fp.items() if not isinstance(x, np.ndarray)}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
