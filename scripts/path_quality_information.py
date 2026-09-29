#!/usr/bin/env python3
"""Hypothesis 7. Within a directional state, does the QUALITY of the move predict what follows?

    python scripts/path_quality_information.py --perms 300

WHY THIS IS NOT HYPOTHESIS 1-6 AGAIN

Those asked whether an event predicts DIRECTION. Answer: no, across 6,060 moving-average
configurations. This holds direction fixed and varies only path quality: among the bars where
EMA(8) sits above EMA(21), do the ones that got there efficiently behave differently from the
ones that rotated their way there?

The EMA is used as PERMISSION, never as a trigger. That is the one use of a moving average this
project has not tested, and it is what the framework spec actually proposes.

THE TEST

    statistic   Welch t on (mean forward return | top-tercile conditioner)
                          - (mean forward return | bottom-tercile conditioner),
                forward return in ATR units and multiplied by the state's direction, so a
                positive t always means "better path quality, better outcome".
    family      7 conditioners x 3 lookbacks x 2 EMA states x 2 sides x 6 horizons
                x 7 timeframes x 2 symbols
    primary     max |t| over the whole family against the same maximum recomputed on each
                grouped bar permutation.
    secondary   count of cells beating their own null 95th percentile.

The spec's composite (0.30/0.25/0.20/0.15/0.10) is one cell of the family. Those weights are
PRE-SPECIFIED, so testing them costs nothing. The fitted version is not tested and will not be:
a continuous search over a simplex is not something a permutation test can charge for.

CALIBRATION

Terciles, z-scores and time-of-day baselines come from the first 20% of train and are applied to
the remaining 80%, which is the only part evaluated. Computing them over the whole sample would
label bar 500 using bar 400,000.

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
from quant.information import DEFAULT_HORIZONS, atr_shifted, evaluate, family_pvalues
from quant.permutation import get_permutation_fast, session_groups
from quant.strategies.path_quality import CALIB_FRAC, EMA_STATES, build_masks, cond_family
from quant.verdict import from_information

HORIZONS = DEFAULT_HORIZONS
ALL_TIMEFRAMES = ("5m", "10m", "15m", "20m", "30m", "45m", "60m")
TIMEFRAMES = ALL_TIMEFRAMES
SYMBOLS = ("MES", "MNQ")
MIN_EVENTS = 100
RTH_LO, RTH_HI = 8 * 60 + 30, 15 * 60
SPECS = cond_family()
N_CELLS = len(SPECS) * len(HORIZONS) * len(TIMEFRAMES) * len(SYMBOLS)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def eval_series(df, atr, tf, calib, eligible):
    """Evaluate every spec on one series, one evaluate() call per EMA state.

    Split by state because `direction` is a single per-bar array: inside the EMA(8)/EMA(21)
    state a bar is long-permitted, inside EMA(20)/EMA(50) it may not be, so the two need
    different sign conventions and cannot share one call.
    """
    c = df["close"].to_numpy(dtype=float)
    out = np.full((len(SPECS), len(HORIZONS)), np.nan)
    for f, s in EMA_STATES:
        idx = [i for i, sp in enumerate(SPECS) if (sp.fast, sp.slow) == (f, s)]
        dirs, masks = build_masks(df, atr, [SPECS[i] for i in idx], tf, calib, eligible)
        cells = evaluate(
            c, atr, masks, np.ones(len(c), bool), HORIZONS, MIN_EVENTS, direction=dirs[(f, s)]
        )
        out[idx] = cells.t
    return out


def main() -> int:
    global TIMEFRAMES, N_CELLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeframes", default=",".join(ALL_TIMEFRAMES))
    ap.add_argument("--out", default="out/path_quality")
    a = ap.parse_args()
    TIMEFRAMES = tuple(t.strip() for t in a.timeframes.split(",") if t.strip())
    N_CELLS = len(SPECS) * len(HORIZONS) * len(TIMEFRAMES) * len(SYMBOLS)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log("=" * 96)
    log("  PATH QUALITY AS A CONDITIONER -- does HOW price moved predict what follows?")
    log("=" * 96)
    log(
        f"  family: {len(SPECS)} conditioning configs x {len(HORIZONS)} horizons x "
        f"{len(TIMEFRAMES)} timeframes x {len(SYMBOLS)} symbols = {N_CELLS:,} cells"
    )
    log(f"  permutations: {a.perms}   segment: TRAIN only   holdouts untouched\n")

    h = Hypothesis(
        name="path_quality_conditioner",
        instrument="MES and MNQ, 5m/10m/15m/20m/30m/45m/60m full-session bars",
        session="08:30-15:00 America/Chicago for the EVENT; the forward window may run past it",
        setup="Hold direction FIXED using an EMA state as permission (EMA(8)>EMA(21) or "
        "EMA(20)>EMA(50) for the bull side, below for the bear side), then split the bars "
        "inside that state into top and bottom terciles of a path-quality conditioner: "
        "ATR-normalised displacement, signed efficiency ratio, trend persistence, negated "
        "rotation count, deseasonalised speed z-score, relative volume, or the spec's "
        "pre-specified 0.30/0.25/0.20/0.15/0.10 composite of them.",
        direction="both",
        exit_rule="none -- this measures a forward return over 1, 3, 6, 12, 26 or 78 bars, "
        "not a trade. Risk management is gated on this result.",
        execution="the conditioner is known at the close of bar t and uses bars <= t only; "
        "terciles, z-scores and time-of-day baselines are fitted on the first 20% of train and "
        "applied to the remaining 80%, which is the only part evaluated",
        costs="NOT MODELLED: there are no trades. Any effect is converted to points and "
        "compared against CostModel.for_prop, so a real but untradeably small effect shows up "
        "as one.",
        invalidation="abandon if the family-corrected permutation p-value of the maximum |t| "
        "over all cells is >= 0.05. A cell significant only on its own uncorrected p-value is "
        "not a finding. A nominal pass carried by a lone spike goes to a single pre-committed "
        "validate look, not to a parameter tweak.",
        metric="max |t| over the family, plus the count of cells beating their own null 95th "
        "percentile, train segment only, reported separately per conditioner",
        rationale="Six hypotheses asked whether an event predicts direction and the answer was "
        "no every time. This holds direction fixed and varies only the path taken to get there, "
        "which is a different quantity: displacement and path length are independent, and every "
        "moving-average test so far has measured only the first. If rotation is what makes a "
        "crossover fail, then within one direction the efficient bars should behave differently "
        "from the rotating ones, and that difference cannot come from drift because both groups "
        "sit inside the same directional state.",
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
            real[si, ti] = eval_series(df, atr, tf, calib, eligible)
            series[(sym, tf)] = dict(
                df=df,
                atr=atr,
                tf=tf,
                calib=calib,
                eligible=eligible,
                groups=session_groups(df),
                atr_med=float(np.nanmedian(atr[calib.stop :])),
            )
            live = int(np.isfinite(real[si, ti]).sum())
            log(
                f"  {sym} {tf:>3}: {n:>7,} train bars, calib {calib.stop:,}, "
                f"eval {int(eligible.sum()):>7,} RTH bars, {live:>3}/{real[si, ti].size} live, "
                f"{time.time() - t0:.1f}s"
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
                null[p, si, ti] = eval_series(
                    perm, atr_shifted(perm), s["tf"], s["calib"], s["eligible"]
                )
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
    log("\n  PER CONDITIONER -- which of the six (plus the composite) carries anything?")
    for name in ("disp", "ser", "persist", "rot", "speed", "relvol", "composite"):
        sel = fam_of == name
        mask = np.zeros_like(finite)
        mask[:, :, sel, :] = True
        m = finite & mask
        if not m.sum():
            log(f"  {name:<12} no live cells")
            continue
        cr = int((np.abs(real) > thr)[m].sum())
        cn = np.array([int((np.abs(null[p]) > thr)[m].sum()) for p in range(a.perms)])
        pv = (1 + int((cn >= cr).sum())) / (1 + a.perms)
        best = np.nanmax(np.abs(np.where(m, real, np.nan)))
        log(
            f"  {name:<12} cells {m.sum():>5}  over {cr:>4} ({100 * cr / m.sum():>5.1f}%)  "
            f"null med {np.median(cn):>5.0f}  p={pv:.4f}  max|t| {best:.2f}  "
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
                                family=spec.family,
                                horizon=hor,
                                t=float(real[si, ti, j, hi]),
                                p_uncorrected=float(fp["p_cell"][si, ti, j, hi]),
                                atr_pts=am,
                                cost_pts=cost,
                            )
                        )
    tab = pd.DataFrame(rows).sort_values("t", key=np.abs, ascending=False).reset_index(drop=True)
    tab.to_csv(out / "cells.csv", index=False)
    log("\n  TOP 15 CELLS BY |t|")
    log(f"  {'sym':<4}{'tf':<5}{'spec':<34}{'h':>4}{'t':>8}{'p_unc':>9}")
    for _, r in tab.head(15).iterrows():
        log(f"  {r.symbol:<4}{r.tf:<5}{r.spec:<34}{r.horizon:>4}{r.t:>8.2f}{r.p_uncorrected:>9.4f}")

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
