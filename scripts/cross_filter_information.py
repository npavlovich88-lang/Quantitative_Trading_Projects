#!/usr/bin/env python3
"""Hypothesis 9. At an EMA cross, does a momentum filter give better entries on average?

    python scripts/cross_filter_information.py --perms 300

THE CLAIM, IN THE USER'S WORDS

    "if we dont trade chop and trade when we have momentum and for say a ema cross we should
     have on average better entries"

WHY THIS IS A GAP AND NOT A REPEAT

Hypothesis 7 rejected these same conditioners measured on EVERY BAR inside an EMA state. That
does not settle this, and the reason is structural: at a cross bar the averages have just met,
so displacement over the preceding N bars is near zero BY CONSTRUCTION. The conditioner
distribution at a cross is a different distribution, and a null on one is not a null on the
other. Hypothesis 3 hit exactly this with separation -- median 0.035 ATR at the cross bar,
0.231 three bars later -- which is why the filter has to be CONFIRMATION rather than an
instantaneous gate.

So the entry is evaluated at cross + L bars for L in {0, 3, 6}, with L = 0 kept as the control
that should be degenerate, and the cross direction is required to still hold at the entry bar.

WHAT IS COMPARED

Only confirmed entries are sampled, split into the top and bottom terciles of the filter. A
positive t means the filtered entries were better. Terciles come from confirmed entries in the
calibration slice, not from all bars: the conditioner's distribution at an entry is not its
distribution over the series.

THREE OUTCOMES, BECAUSE "BETTER" IS AMBIGUOUS

    ret   is the mean better
    mfe   does it give a RUN -- the actual question for a 5-15 minute hold
    mae   is the drawdown smaller

A filter that improves entries should move at least one. A filter that only cuts trade count
without moving any has cost sample and bought nothing, and the run prints the surviving trade
count so that trade-off is visible rather than implied.

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
from quant.strategies.cross_filter import CALIB_FRAC, build_masks, filter_family
from quant.verdict import from_information

HORIZONS = DEFAULT_HORIZONS
ALL_TIMEFRAMES = ("3m", "5m", "10m", "15m", "30m")
TIMEFRAMES = ALL_TIMEFRAMES
SYMBOLS = ("MES", "MNQ")
MIN_EVENTS = 100
RTH_LO, RTH_HI = 8 * 60 + 30, 15 * 60
SPECS = filter_family()
N_CELLS = len(SPECS) * len(HORIZONS) * len(TIMEFRAMES) * len(SYMBOLS)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def eval_series(df, atr, tf, calib, eligible, counts=None):
    """t per (spec, horizon). Grouped by (pair, lag) because direction depends on both."""
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    out = np.full((len(SPECS), len(HORIZONS)), np.nan)
    ones = np.ones(len(c), bool)
    keys = {(sp.fast, sp.slow, sp.lag) for sp in SPECS}
    for f, s, lag in sorted(keys):
        idx = [i for i, sp in enumerate(SPECS) if (sp.fast, sp.slow, sp.lag) == (f, s, lag)]
        sub = [SPECS[i] for i in idx]
        ent, masks = build_masks(df, atr, sub, tf, calib, eligible)
        d = ent[(f, s, lag)][1]
        if counts is not None:
            e = ent[(f, s, lag)][0] & eligible
            counts[(f, s, lag)] = int(e.sum())
        for hi, hor in enumerate(HORIZONS):
            oc = fwd_out(h, lo, c, atr, hor, d)
            for kind in ("ret", "mfe", "mae"):
                keep = [k for k, sp in enumerate(sub) if sp.outcome == kind]
                if not keep:
                    continue
                t, _dd, _a, _b = evaluate_outcomes(
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
    ap.add_argument("--out", default="out/cross_filter")
    a = ap.parse_args()
    TIMEFRAMES = tuple(t.strip() for t in a.timeframes.split(",") if t.strip())
    N_CELLS = len(SPECS) * len(HORIZONS) * len(TIMEFRAMES) * len(SYMBOLS)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log("=" * 96)
    log("  MOMENTUM FILTERS AT THE CROSS -- do they give better entries on average?")
    log("=" * 96)
    log(
        f"  family: {len(SPECS)} configs x {len(HORIZONS)} horizons x {len(TIMEFRAMES)} "
        f"timeframes x {len(SYMBOLS)} symbols = {N_CELLS:,} cells"
    )
    log(f"  permutations: {a.perms}   segment: TRAIN only   holdouts untouched\n")

    h = Hypothesis(
        name="cross_momentum_filter",
        instrument="MES and MNQ, 3m/5m/10m/15m/30m full-session bars",
        session="08:30-15:00 America/Chicago for the ENTRY; the forward window may run past it",
        setup="Sample EMA crosses (8/21 and 20/50). Confirm the entry L bars later for L in "
        "{0, 3, 6}, requiring the cross direction to still hold at that bar. At the entry bar "
        "measure one momentum filter over a fixed 20-bar lookback -- ATR-normalised "
        "displacement, signed efficiency ratio, trend persistence, negated rotation count, "
        "deseasonalised speed z-score, relative volume, or the pre-specified composite -- and "
        "compare the top tercile of confirmed entries against the bottom tercile.",
        direction="both",
        exit_rule="none -- three fixed-horizon outcomes are measured at 1, 3, 6, 12, 26 and 78 "
        "bars: forward return, maximum favourable excursion, maximum adverse excursion.",
        execution="the cross, the confirmation and the filter value are all known at the close "
        "of the entry bar and use bars <= it; excursions look at bars t+1..t+h; terciles come "
        "from confirmed entries inside the first 20% of train and apply to the remaining 80%",
        costs="NOT MODELLED: there are no trades. Effects convert to points and print beside "
        "CostModel.for_prop, and surviving trade counts print too, so a filter that only "
        "removes sample is visible as one.",
        invalidation="abandon if the family-corrected permutation p-value of the maximum |t| "
        "is >= 0.05. A nominal pass carried by a lone spike goes to one pre-committed validate "
        "look, never to a parameter tweak.",
        metric="max |t| over the family, plus the count of cells beating their own null 95th "
        "percentile, reported separately per filter, per confirmation lag and per outcome",
        rationale="Hypothesis 7 rejected these conditioners measured over every bar inside an "
        "EMA state, but that does not settle this. At a cross bar the averages have just met, "
        "so displacement over the preceding window is near zero by construction and the "
        "conditioner distribution at a cross is a different distribution -- hypothesis 3 "
        "measured exactly this for separation, median 0.035 ATR at the cross against 0.231 "
        "three bars later. The mechanism claimed is that a cross occurring while the market is "
        "already moving is a different event from one occurring in balance, and the "
        "confirmation lag is what lets the difference exist at all.",
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
            counts: dict = {}
            real[si, ti] = eval_series(df, atr, tf, calib, eligible, counts)
            series[(sym, tf)] = dict(
                df=df,
                tf=tf,
                calib=calib,
                eligible=eligible,
                groups=session_groups(df),
                atr_med=float(np.nanmedian(atr[calib.stop :])),
                counts=counts,
            )
            live = int(np.isfinite(real[si, ti]).sum())
            c821 = counts.get((8, 21, 3), 0)
            log(
                f"  {sym} {tf:>3}: {n:>7,} bars, {live:>3}/{real[si, ti].size} live, "
                f"ema8-21 L3 entries {c821:>6,}, {time.time() - t0:.1f}s"
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

    def slice_test(mask, name):
        m = finite & mask
        if not m.sum():
            return f"  {name:<22} no live cells"
        cr = int((np.abs(real) > thr)[m].sum())
        cn = np.array([int((np.abs(null[p]) > thr)[m].sum()) for p in range(a.perms)])
        pv = (1 + int((cn >= cr).sum())) / (1 + a.perms)
        return (
            f"  {name:<22} cells {m.sum():>5}  over {cr:>4} ({100 * cr / m.sum():>5.1f}%)  "
            f"null med {np.median(cn):>5.0f}  p={pv:.4f}  "
            f"max|t| {np.nanmax(np.abs(np.where(m, real, np.nan))):.2f}  "
            f"{100 * (real[m] > 0).mean():>3.0f}% positive"
        )

    log("\n  BY FILTER")
    fam_of = np.array([s.family for s in SPECS])
    for name in ("disp", "ser", "persist", "rot", "speed", "relvol", "composite"):
        sel = fam_of == name
        mk = np.zeros_like(finite)
        mk[:, :, sel, :] = True
        log(slice_test(mk, name))

    log("\n  BY CONFIRMATION LAG  (L=0 is the control that should be degenerate)")
    lag_of = np.array([s.lag for s in SPECS])
    for lag in (0, 3, 6):
        sel = lag_of == lag
        mk = np.zeros_like(finite)
        mk[:, :, sel, :] = True
        log(slice_test(mk, f"L={lag}"))

    log("\n  BY OUTCOME")
    out_of = np.array([s.outcome for s in SPECS])
    for kind in ("ret", "mfe", "mae"):
        sel = out_of == kind
        mk = np.zeros_like(finite)
        mk[:, :, sel, :] = True
        log(slice_test(mk, kind))

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
                                filt=spec.family,
                                lag=spec.lag,
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
    log("\n  TOP 15 CELLS BY |t|   (positive t = the FILTERED entries were better)")
    log(f"  {'sym':<4}{'tf':<5}{'spec':<38}{'h':>4}{'t':>8}{'p_unc':>9}")
    for _, r in tab.head(15).iterrows():
        log(f"  {r.symbol:<4}{r.tf:<5}{r.spec:<38}{r.horizon:>4}{r.t:>8.2f}{r.p_uncorrected:>9.4f}")

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
