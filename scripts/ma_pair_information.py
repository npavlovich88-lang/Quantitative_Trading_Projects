#!/usr/bin/env python3
"""HMA/EMA, TEMA/EMA and VWAP/EMA crossovers: do they carry directional information?

    python scripts/ma_pair_information.py --perms 1000

THE HYPOTHESIS AND WHAT MAKES IT DIFFERENT FROM THE FOUR ALREADY REJECTED

HMA and TEMA are low-lag smoothers of the same closes an EMA smooths, so HMA(21)/EMA(50) is
structurally the same object as the EMA(10)/EMA(29), EMA(8)/EMA(21), EMA(20)/EMA(50) and
KAMA(20)/EMA(50) crossovers already rejected. Two things stop this being a parameter tweak:

  1. All four rejections were measured on 5-minute bars, and the Ichimoku information test
     (2026-09-24, hypothesis e0b371b2) found 5m and 10m sit at CHANCE for both symbols while 30m
     and 60m do not. The family has never been tested where the evidence points.
  2. VWAP is a different object: volume-weighted, session-anchored, resets daily. It is the first
     thing in this project to use the volume column.

So the question is not "is the statistic non-zero" -- the Ichimoku run already showed a weak
broad continuation tendency at 30m+ that the plain EMA control picked up too. The question is
whether HMA, TEMA or VWAP SEPARATE from that EMA control. The EMA/EMA pairs are inside the
family for exactly that comparison, and are reported as their own row.

THE TEST

Identical machinery to the Ichimoku study (quant.information), so the two are directly
comparable:

  statistic    Welch t on (mean forward return after a bullish cross) - (after a bearish cross),
               forward return in ATR(14) units, ATR shifted one bar
  family       61 pairs x 6 horizons x 5 timeframes x 2 symbols = 3,660 cells, enumerated in
               code before the first run
  primary      max |t| over the whole family against the max |t| over the same family on each of
               N grouped bar permutations -- the search is charged for, not quoted after
  secondary    count of cells beating their OWN null 95th percentile, same permutation null.
               Powerful against a diffuse effect that the max test would miss.

Costs are not in the statistic because there are no trades. Every effect is converted to points
and printed beside the round turn, so a statistically real but untradeably small effect shows up
as one. Risk management is the NEXT stage and is deliberately gated on this one: an exit rule
cannot manufacture information that is not in the signal, and four strategies have now died at
precisely that step.

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
from quant.information import DEFAULT_HORIZONS, atr_shifted, evaluate, family_pvalues, verify
from quant.permutation import get_permutation_fast, session_groups
from quant.strategies.ma_pairs import build_pair, cross_masks, pair_family
from quant.verdict import from_information

HORIZONS = DEFAULT_HORIZONS
TIMEFRAMES = ("5m", "10m", "15m", "30m", "60m")
SYMBOLS = ("MES", "MNQ")
WARMUP = 1000  # the slowest line is EMA(200); 1,000 bars leaves <0.3% of its initial value
MIN_EVENTS = 100
RTH_LO, RTH_HI = 8 * 60 + 30, 15 * 60
SPECS = pair_family()
N_CELLS = len(SPECS) * len(HORIZONS) * len(TIMEFRAMES) * len(SYMBOLS)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def masks_for(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    cache: dict = {}
    return [cross_masks(*build_pair(s, df, cache)) for s in SPECS]


HDR = (
    f"{'sym':<4}{'tf':<5}{'pair':<18}{'h':>4}{'t':>8}{'d(ATR)':>9}"
    f"{'d(pts)':>9}{'cost':>7}{'net':>8}{'p_unc':>8}"
)


def show(frame: pd.DataFrame, n: int) -> None:
    for _, r in frame.head(n).iterrows():
        log(
            f"  {r.symbol:<4}{r.tf:<5}{r.pair:<18}{r.horizon:>4}{r.t:>8.2f}"
            f"{r.d_atr:>9.4f}{r.d_pts:>9.3f}{r.cost_pts:>7.2f}"
            f"{r.edge_after_cost:>8.3f}{r.p_uncorrected:>8.4f}"
        )


def subfamily_count_test(real, null, thr, finite, mask, name) -> str:
    """The count test restricted to a slice of the family. Same null, same thresholds."""
    m = finite & mask
    if m.sum() == 0:
        return f"  {name:<26} no live cells"
    cr = int((np.abs(real) > thr)[m].sum())
    cn = np.array([int((np.abs(null[p]) > thr)[m].sum()) for p in range(null.shape[0])])
    p = (1 + int((cn >= cr).sum())) / (1 + null.shape[0])
    return (
        f"  {name:<26} cells {m.sum():>5}  over {cr:>4} ({100 * cr / m.sum():>5.1f}%)  "
        f"null med {np.median(cn):>6.0f}  p={p:.4f}  {100 * (real[m] > 0).mean():>4.0f}% positive"
    )


def main() -> int:
    global TIMEFRAMES, N_CELLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/ma_pair_information")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 96)
    log("  HMA / TEMA / VWAP  versus  EMA:  DO THE CROSSES CARRY DIRECTIONAL INFORMATION?")
    log("=" * 96)
    log(
        f"  family: {len(SPECS)} pairs x {len(HORIZONS)} horizons x {len(TIMEFRAMES)} "
        f"timeframes x {len(SYMBOLS)} symbols = {N_CELLS:,} cells"
    )
    log(f"  permutations: {a.perms}   segment: TRAIN only   holdouts untouched\n")

    h = Hypothesis(
        name="ma_pair_information",
        instrument="MES and MNQ, 5m/10m/15m/30m/60m full-session bars",
        session="08:30-15:00 America/Chicago for the EVENT; the forward window may run past it",
        setup="A fast line crosses an EMA. Fast line is HMA(9/14/21/34/55), TEMA(9/14/21/34/55), "
        "session-anchored VWAP, or -- as the control -- a faster EMA. Slow line is "
        "EMA(21/50/100/200). The cross is the event, identified at the close of the bar that "
        "completes it.",
        direction="both",
        exit_rule="none -- this measures a forward return over a fixed horizon of 1, 3, 6, 12, "
        "26 or 78 bars, not a trade. Risk management is the next stage and is gated on this one.",
        execution="the event is known at the close of bar t; the forward return runs from "
        "close[t] to close[t+h] and is divided by ATR(14) shifted one bar",
        costs="NOT MODELLED, deliberately: there are no trades. The measured effect is "
        "converted to points and printed beside the CostModel.for_prop round turn, so a "
        "statistically real but untradeably small effect is visible as such.",
        invalidation="abandon if the family-corrected permutation p-value of the maximum |t| "
        "over all 3,660 cells is >= 0.05 AND no non-EMA sub-family separates from the EMA "
        "control on the count test. A cell significant only on its own uncorrected p-value is "
        "not a finding, it is the best of 3,660 draws.",
        metric="max |t| over the family, plus the count of cells beating their own null 95th "
        "percentile, on the train segment only; and the same two statistics computed separately "
        "for the HMA, TEMA, VWAP and EMA-control sub-families",
        rationale="HMA and TEMA are expected to behave like the EMA control, because a third "
        "smoothing kernel over the same closes is a re-parameterisation, not a new mechanism. "
        "Two things make the run worth doing anyway: every prior rejection in this family was "
        "measured on 5m bars, and the Ichimoku test found 5m/10m at chance while 30m/60m were "
        "not; and VWAP is volume-weighted and session-anchored, so it is a genuinely different "
        "object and the first use of the volume column here. The pre-declared expectation is "
        "that any separation comes from VWAP or from the timeframe, not from the kernel.",
        author="research@example.invalid",
    )
    hh = h.register("out/hypotheses.jsonl")
    log(f"[pre-registered] {hh}\n")

    # ------------------------------------------------------------------ real statistics
    series: dict[tuple[str, str], dict] = {}
    real = np.full((len(SYMBOLS), len(TIMEFRAMES), len(SPECS), len(HORIZONS)), np.nan)
    dreal = np.full_like(real, np.nan)

    for si, sym in enumerate(SYMBOLS):
        for ti, tf in enumerate(TIMEFRAMES):
            t0 = time.time()
            df = S.load_data(DATA[sym].format(tf=tf), "America/Chicago")
            dates = df["dt"].dt.tz_localize(None).to_numpy()
            sp = make_split(sym, tf, dates)
            df = df.iloc[: sp.train.stop].reset_index(drop=True)
            n = len(df)
            mins = df["dt"].dt.hour.to_numpy() * 60 + df["dt"].dt.minute.to_numpy()
            atr = atr_shifted(df)
            base = (
                (mins >= RTH_LO)
                & (mins < RTH_HI)
                & (np.arange(n) >= WARMUP)
                & np.isfinite(atr)
                & (atr > 0)
            )
            c = df["close"].to_numpy(float)
            m = masks_for(df)
            cells = evaluate(c, atr, m, base, HORIZONS, MIN_EVENTS)
            real[si, ti] = cells.t
            dreal[si, ti] = cells.d
            series[(sym, tf)] = dict(
                df=df,
                base=base,
                groups=session_groups(df),
                atr_med=float(np.nanmedian(atr[WARMUP:])),
            )
            err = verify(c, atr, m, base, cells, HORIZONS, MIN_EVENTS, n_check=8)
            live = int(np.isfinite(cells.t).sum())
            log(
                f"  {sym} {tf:>3}: {n:>7,} train bars, {live:>3}/{cells.t.size} live cells, "
                f"ATR med {series[(sym, tf)]['atr_med']:6.2f} pts, "
                f"float32 check {err:.2e}, {time.time() - t0:.1f}s"
            )
            if err > 1e-3:
                log("  !! float32 statistic disagrees with the exact recomputation -- ABORT")
                return 1

    finite = np.isfinite(real)
    log(f"\n  {finite.sum():,} of {N_CELLS:,} cells have >= {MIN_EVENTS} events on both sides")
    idx = np.unravel_index(np.nanargmax(np.abs(real)), real.shape)
    log(
        f"  observed max |t| = {np.nanmax(np.abs(real)):.3f}  at  {SYMBOLS[idx[0]]} "
        f"{TIMEFRAMES[idx[1]]} {SPECS[idx[2]].label} h={HORIZONS[idx[3]]}\n"
    )

    # ------------------------------------------------------------------ permutation null
    log(f"  running {a.perms} grouped bar permutations over the whole family ...")
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
                null[p, si, ti] = evaluate(
                    perm["close"].to_numpy(float),
                    atr_shifted(perm),
                    masks_for(perm),
                    s["base"],
                    HORIZONS,
                    MIN_EVENTS,
                ).t
        if p == 0 or (p + 1) % 25 == 0:
            el = time.time() - t0
            log(
                f"    {p + 1:>4}/{a.perms}  null max |t| so far "
                f"{np.nanmax(np.abs(null[: p + 1])):.3f}   "
                f"{el:.0f}s elapsed, ~{el / (p + 1) * (a.perms - p - 1):.0f}s left"
            )

    np.savez_compressed(
        out / "statistics.npz",
        real=real,
        d=dreal,
        null=null,
        symbols=np.array(SYMBOLS),
        timeframes=np.array(TIMEFRAMES),
        horizons=np.array(HORIZONS),
        labels=np.array([s.label for s in SPECS]),
        families=np.array([s.family for s in SPECS]),
    )

    # ------------------------------------------------------------------ verdict
    fp = family_pvalues(real, null)
    log("\n" + "=" * 96)
    log("  RESULT")
    log("=" * 96)
    log(f"  observed max |t| over {fp['live_cells']:,} live cells : {fp['observed_max_t']:.3f}")
    log(
        f"  null max |t|  median {fp['null_max_median']:.3f}   "
        f"95th pct {fp['null_max_p95']:.3f}   max {fp['null_max_max']:.3f}"
    )
    log(f"  PRIMARY   family-corrected p (max statistic) = {fp['p_max']:.4f}")
    log(
        f"  SECONDARY count statistic: {fp['cells_over_own_null']} cells beat their own null "
        f"95th pct vs null median {fp['null_count_median']:.0f}, p = {fp['p_count']:.4f}"
    )
    log(
        f"  (per-cell null 95th pct of |t| has median {fp['per_cell_threshold_median']:.3f}, "
        f"not 1.960 -- a t-table would be wrong here)"
    )

    thr, p_cell = fp["cell_threshold"], fp["p_cell"]
    fam_of = np.array([s.family for s in SPECS])

    log("\n  THE COMPARISON THAT MATTERS -- does any kernel separate from the EMA control?")
    for name in ("hma", "tema", "vwap", "ema  <- CONTROL"):
        key = name.split()[0]
        sel = fam_of == key
        mask = np.zeros_like(finite)
        mask[:, :, sel, :] = True
        log(subfamily_count_test(real, null, thr, finite, mask, name))

    log("\n  BY TIMEFRAME -- the Ichimoku run predicted 5m/10m at chance, 30m/60m not")
    for si, sym in enumerate(SYMBOLS):
        for ti, tf in enumerate(TIMEFRAMES):
            mask = np.zeros_like(finite)
            mask[si, ti] = True
            log(subfamily_count_test(real, null, thr, finite, mask, f"{sym} {tf}"))

    # ------------------------------------------------------------------ the cells themselves
    rows = []
    for si, sym in enumerate(SYMBOLS):
        cost = CostModel.for_prop(sym, "average", contracts=1).round_turn_points
        for ti, tf in enumerate(TIMEFRAMES):
            atr_med = series[(sym, tf)]["atr_med"]
            for j, spec in enumerate(SPECS):
                for hi, hor in enumerate(HORIZONS):
                    if not finite[si, ti, j, hi]:
                        continue
                    rows.append(
                        dict(
                            symbol=sym,
                            tf=tf,
                            pair=spec.label,
                            family=spec.family,
                            horizon=hor,
                            t=float(real[si, ti, j, hi]),
                            d_atr=float(dreal[si, ti, j, hi]),
                            d_pts=float(dreal[si, ti, j, hi]) * atr_med,
                            cost_pts=cost,
                            p_uncorrected=float(p_cell[si, ti, j, hi]),
                        )
                    )
    tab = pd.DataFrame(rows)
    tab["edge_after_cost"] = tab["d_pts"].abs() - tab["cost_pts"]
    tab = tab.sort_values("t", key=np.abs, ascending=False).reset_index(drop=True)
    tab.to_csv(out / "cells.csv", index=False)

    log("\n  TOP 15 CELLS BY |t|")
    log("  " + HDR)
    show(tab, 15)

    for key, title in (("vwap", "VWAP"), ("hma", "HMA"), ("tema", "TEMA")):
        log(f"\n  BEST {title} CELLS")
        log("  " + HDR)
        show(tab[tab["family"] == key].sort_values("t", key=np.abs, ascending=False), 6)

    above = tab[tab["t"].abs() > fp["null_max_p95"]]
    log(f"\n  cells above the family 5% threshold (|t| > {fp['null_max_p95']:.3f}): {len(above)}")
    if len(above):
        log("  " + HDR)
        show(above, 15)
        both = above[above["edge_after_cost"] > 0]
        log(f"  of those, beating one round turn: {len(both)}")

    summary = dict(
        hypothesis=hh,
        family_cells=N_CELLS,
        live_cells=fp["live_cells"],
        permutations=a.perms,
        observed_max_t=fp["observed_max_t"],
        observed_max_cell=(
            f"{SYMBOLS[idx[0]]} {TIMEFRAMES[idx[1]]} {SPECS[idx[2]].label} h={HORIZONS[idx[3]]}"
        ),
        null_max_p95=fp["null_max_p95"],
        p_max=fp["p_max"],
        cells_over_own_null=fp["cells_over_own_null"],
        null_count_median=fp["null_count_median"],
        p_count=fp["p_count"],
        cells_above_family_threshold=len(above),
        verdict="PASS" if fp["p_max"] < 0.05 else "REJECT",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ the standing verdict
    # Rendered from the statistics rather than written by hand, so it cannot drift from them.
    v = from_information(
        out.name,
        real,
        null,
        [s.label for s in SPECS],
        list(SYMBOLS),
        list(TIMEFRAMES),
        list(HORIZONS),
        np.array([s.family for s in SPECS]),
    )
    log("")
    log(v.render())
    (out / "verdict.txt").write_text(v.render(), encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps(v.as_dict(), indent=2), encoding="utf-8")
    log(f"\n  PRIMARY VERDICT: {summary['verdict']}")
    log(f"  written to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
