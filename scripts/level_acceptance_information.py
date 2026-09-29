#!/usr/bin/env python3
"""Hypothesis 10. When price breaks a level, does ACCEPTANCE tell you anything?

    python scripts/level_acceptance_information.py --perms 200

THE CLAIM

    "does price breaking above or below a meaningful local reference ... have any conviction
     when it does break, does it reject price or does it accept
     price accepts above it -- meaning it remains above it long enough, trades enough volume
     there, or retests it and holds"

THE THREE THINGS THAT MAKE THIS DIFFERENT FROM HYPOTHESES 1-9

1. A MECHANISM. Every variable tested so far was a smoothing of past closes. Nothing rests on a
   moving average and no one defends it. A prior-day value-area edge, a session VWAP or a swing
   high marks a price where contracts changed hands and where stops sit.

2. A POST-EVENT MEASUREMENT. Everything so far was a point-in-time state or a trailing window.
   Acceptance watches what happens AFTER the break. The entry therefore sits at the END of the
   acceptance window and every outcome is measured from there, so nothing inside the window can
   reach the trade.

3. A MATCHED CONTROL, which the framework spec never asked for. One of the thirteen levels is a
   random line placed at a plausible distance with no structural meaning. If prior-day VAH
   cannot beat a randomly placed line, then acceptance is measuring drift and the premise dies
   before anything else matters.

OUTCOMES ARE NORMALISED BY FORWARD RANGE, NOT TRAILING ATR

Hypothesis 9 passed its family gate at p = 0.0033 and turned out to be selecting volatility:
favourable excursion, adverse excursion and forward realised range all scaled by the same
factor, because ATR(14) shifted one bar is a TRAILING normaliser and a momentum filter selects
RISING volatility. Dividing by the FORWARD realised range removes the magnitude channel by
construction -- ret_r asks "of the range the market actually delivered, what fraction did we
capture directionally", which cannot be inflated by the market simply moving more.

SEGMENTS

TRAIN ONLY. Terciles from the first 20% of train, applied to the remaining 80%, which is the
only part evaluated. Neither holdout is touched and neither validate segment is read.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant import strategy as S
from quant.data_splits import DATA, make_split
from quant.hypothesis import Hypothesis
from quant.information import evaluate_outcomes, family_pvalues
from quant.permutation import get_permutation_fast, session_groups
from quant.strategies.levels import (
    ACCEPT_MODES,
    ACCEPT_WINDOWS,
    LEVEL_TYPES,
    all_levels,
    breaks_and_acceptance,
)
from quant.verdict import from_information

HORIZONS = (3, 6, 12, 26)
ALL_TIMEFRAMES = ("5m", "15m", "30m")
TIMEFRAMES = ALL_TIMEFRAMES
SYMBOLS = ("MES", "MNQ")
OUTCOMES = ("ret_r", "mfe_r", "mae_r")
MIN_EVENTS = 100
RTH_LO, RTH_HI = 8 * 60 + 30, 15 * 60
CALIB_FRAC = 0.20


@dataclasses.dataclass(frozen=True)
class Spec:
    level: str
    window: int
    mode: str
    outcome: str

    @property
    def label(self) -> str:
        return f"{self.level}/W{self.window}/{self.mode}/{self.outcome}"

    @property
    def family(self) -> str:
        return self.level


SPECS = [
    Spec(lv, w, m, o)
    for lv in LEVEL_TYPES
    for w in ACCEPT_WINDOWS
    for m in ACCEPT_MODES
    for o in OUTCOMES
]
N_CELLS = len(SPECS) * len(HORIZONS) * len(TIMEFRAMES) * len(SYMBOLS)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def range_outcomes(h, lo, c, hor, d):
    """Forward outcomes divided by the FORWARD realised range -- magnitude-free by construction.

    ret_r is the fraction of the range the market actually delivered that was captured in the
    trade's direction. A filter that only selects bigger moves cannot move it.
    """
    fmax = pd.Series(h).rolling(hor).max().shift(-hor).to_numpy()
    fmin = pd.Series(lo).rolling(hor).min().shift(-hor).to_numpy()
    n = len(c)
    fwd = np.full(n, np.nan)
    fwd[: n - hor] = c[hor:] - c[: n - hor]
    rng = fmax - fmin
    up, dn = fmax - c, c - fmin
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = rng > 0
        return {
            "ret_r": np.where(ok, fwd * d / rng, np.nan),
            "mfe_r": np.where(ok, np.where(d > 0, up, dn) / rng, np.nan),
            "mae_r": np.where(ok, -np.where(d > 0, dn, up) / rng, np.nan),
        }


def eval_series(df, calib, eligible, seed=0, counts=None):
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    lv = all_levels(df, seed)
    out = np.full((len(SPECS), len(HORIZONS)), np.nan)
    ones = np.ones(len(c), bool)

    for level in LEVEL_TYPES:
        for window in ACCEPT_WINDOWS:
            ba = breaks_and_acceptance(df, lv[level], window)
            ent, d = ba["entry"], ba["direction"]
            if counts is not None and window == ACCEPT_WINDOWS[0]:
                counts[level] = int((ent & eligible).sum())
            masks_by_mode = {}
            for mode in ACCEPT_MODES:
                acc = ba[f"acc_{mode}"]
                pool = acc[calib][ent[calib] & np.isfinite(acc[calib])]
                if len(pool) < 150:
                    empty = np.zeros(len(c), bool)
                    masks_by_mode[mode] = (empty, empty)
                    continue
                q1, q2 = np.quantile(pool, [1 / 3, 2 / 3])
                base = ent & eligible & np.isfinite(acc) & np.isfinite(d) & (d != 0)
                # ACCEPTED first, so a positive t means acceptance beat rejection.
                masks_by_mode[mode] = (base & (acc >= q2), base & (acc <= q1))
            for hi, hor in enumerate(HORIZONS):
                oc = range_outcomes(h, lo, c, hor, d)
                for mode in ACCEPT_MODES:
                    for o in OUTCOMES:
                        j = SPECS.index(Spec(level, window, mode, o))
                        t, _dd, _a, _b = evaluate_outcomes(
                            oc[o], [masks_by_mode[mode]], ones, MIN_EVENTS
                        )
                        out[j, hi] = t[0]
    return out


def main() -> int:
    global TIMEFRAMES, N_CELLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeframes", default=",".join(ALL_TIMEFRAMES))
    ap.add_argument("--out", default="out/level_acceptance")
    a = ap.parse_args()
    TIMEFRAMES = tuple(t.strip() for t in a.timeframes.split(",") if t.strip())
    N_CELLS = len(SPECS) * len(HORIZONS) * len(TIMEFRAMES) * len(SYMBOLS)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log("=" * 96)
    log("  LEVEL BREAKS AND ACCEPTANCE -- does holding beyond a level mean anything?")
    log("=" * 96)
    log(
        f"  family: {len(SPECS)} configs x {len(HORIZONS)} horizons x {len(TIMEFRAMES)} "
        f"timeframes x {len(SYMBOLS)} symbols = {N_CELLS:,} cells"
    )
    log(f"  permutations: {a.perms}   segment: TRAIN only   holdouts untouched\n")

    h = Hypothesis(
        name="level_break_acceptance",
        instrument="MES and MNQ, 5m/15m/30m full-session bars",
        session="08:30-15:00 America/Chicago for the ENTRY; the forward window may run past it",
        setup="Price closes through one of thirteen levels: prior-session POC, VAH, VAL, the "
        "nearest high- and low-volume node from the prior session's profile, session VWAP and "
        "its +/-1 sigma bands, the opening-range high and low, the most recent confirmed swing "
        "high and low, and a RANDOM control line placed at a plausible distance. Over the next "
        "W bars (W = 6, 12) measure acceptance as the fraction of bars, and separately the "
        "fraction of volume, that held beyond the level. Compare the top tercile of acceptance "
        "against the bottom tercile.",
        direction="both",
        exit_rule="none -- three fixed-horizon outcomes at 3, 6, 12 and 26 bars, each divided "
        "by the FORWARD realised range so the magnitude channel is removed by construction.",
        execution="the break is at bar t, acceptance is measured over t+1..t+W, and the ENTRY "
        "is at t+W with every outcome measured from there, so nothing in the acceptance window "
        "reaches the trade. All thirteen levels are recomputed from the bars, including under "
        "permutation; the tick-derived profiles are deliberately NOT used because a level fixed "
        "at a real price would break the null on a re-strung series.",
        costs="NOT MODELLED: there are no trades. Outcomes are range-relative fractions rather "
        "than points, so costs enter at the traded stage, which is gated on this result.",
        invalidation="abandon if the family-corrected permutation p-value of the maximum |t| "
        "is >= 0.05, OR if no structural level separates from the random control on the count "
        "statistic. The second condition matters as much as the first: beating noise while "
        "failing to beat an arbitrary line means the levels are not doing the work.",
        metric="max |t| over the family, plus the count of cells beating their own null 95th "
        "percentile, reported per level type so every structural level can be compared directly "
        "against the random control, train segment only",
        rationale="Every variable in hypotheses 1-9 was a smoothing of past closes, which has no "
        "participant behind it. These levels mark prices where contracts actually changed hands "
        "and where resting orders sit, which is a reason for a level to matter rather than a "
        "pattern that happened to work. Acceptance is the one property a level can express that "
        "a state variable cannot: whether the orders defending it were consumed or held.",
        author="research@example.invalid",
    )
    hh = h.register("out/hypotheses.jsonl")
    log(f"[pre-registered] {hh}\n")

    series: dict = {}
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
            eligible = (mins >= RTH_LO) & (mins < RTH_HI) & (np.arange(n) >= calib.stop)
            counts: dict = {}
            real[si, ti] = eval_series(df, calib, eligible, a.seed, counts)
            series[(sym, tf)] = dict(
                df=df, calib=calib, eligible=eligible, groups=session_groups(df)
            )
            live = int(np.isfinite(real[si, ti]).sum())
            log(
                f"  {sym} {tf:>3}: {n:>7,} bars, {live:>3}/{real[si, ti].size} live, "
                f"breaks: vwap {counts.get('vwap', 0):,} poc {counts.get('pd_poc', 0):,} "
                f"rand {counts.get('rand', 0):,}, {time.time() - t0:.1f}s"
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
                null[p, si, ti] = eval_series(perm, s["calib"], s["eligible"], a.seed + p * 7919)
        if p == 0 or (p + 1) % 10 == 0:
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

    def slice_test(sel, name):
        mk = np.zeros_like(finite)
        mk[:, :, sel, :] = True
        m = finite & mk
        if not m.sum():
            return f"  {name:<12} no live cells", 0.0
        cr = int((np.abs(real) > thr)[m].sum())
        cn = np.array([int((np.abs(null[p]) > thr)[m].sum()) for p in range(a.perms)])
        pv = (1 + int((cn >= cr).sum())) / (1 + a.perms)
        pct = 100 * cr / m.sum()
        return (
            f"  {name:<12} cells {m.sum():>5}  over {cr:>4} ({pct:>5.1f}%)  "
            f"null med {np.median(cn):>5.0f}  p={pv:.4f}  "
            f"max|t| {np.nanmax(np.abs(np.where(m, real, np.nan))):.2f}  "
            f"{100 * (real[m] > 0).mean():>3.0f}% positive"
        ), pct

    log("\n  BY LEVEL TYPE  -- every structural level against the RANDOM control")
    pcts = {}
    for name in LEVEL_TYPES:
        line, pct = slice_test(fam_of == name, name)
        pcts[name] = pct
        log(line + ("   <-- CONTROL" if name == "rand" else ""))
    beat = [k for k, v in pcts.items() if k != "rand" and v > pcts.get("rand", 0)]
    log(f"\n  structural levels beating the random control: {len(beat)} of 12  {beat}")

    log("\n  BY OUTCOME  (range-normalised: magnitude cannot inflate these)")
    out_of = np.array([s.outcome for s in SPECS])
    for kind in OUTCOMES:
        log(slice_test(out_of == kind, kind)[0])

    rows = []
    for si, sym in enumerate(SYMBOLS):
        for ti, tf in enumerate(TIMEFRAMES):
            for j, spec in enumerate(SPECS):
                for hi, hor in enumerate(HORIZONS):
                    if finite[si, ti, j, hi]:
                        rows.append(
                            dict(
                                symbol=sym,
                                tf=tf,
                                spec=spec.label,
                                level=spec.level,
                                window=spec.window,
                                mode=spec.mode,
                                outcome=spec.outcome,
                                horizon=hor,
                                t=float(real[si, ti, j, hi]),
                                p_uncorrected=float(fp["p_cell"][si, ti, j, hi]),
                            )
                        )
    tab = pd.DataFrame(rows).sort_values("t", key=np.abs, ascending=False).reset_index(drop=True)
    tab.to_csv(out / "cells.csv", index=False)
    log("\n  TOP 15 CELLS BY |t|   (positive t = ACCEPTED breaks did better)")
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
