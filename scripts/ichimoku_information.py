#!/usr/bin/env python3
"""Does a price cross of an Ichimoku line carry directional information?

    python scripts/ichimoku_information.py --perms 300

THE QUESTION, AND WHY IT IS NOT A BACKTEST

The ask was whether crossing one of these lines tells you anything about whether the market is
about to be bullish or bearish. That is a question about CONDITIONAL RETURN, not about a
strategy, and it is both cheaper to answer and harder to fool yourself with:

    statistic   t = (mean forward return after an UP cross
                     - mean forward return after a DOWN cross) / its standard error,
                forward return measured in ATR(14) units so 2020 and 2026 are comparable.

If t is indistinguishable from what shuffled data produces, no exit rule, no filter and no
position sizing can retrieve it. So this runs before any of that, and it costs almost nothing
to run because there are no trades to simulate.

THE SEARCH IS PAID FOR, NOT HIDDEN

The family is 40 line configurations x 6 horizons x 5 timeframes x 2 symbols = 2,400 cells,
enumerated in code before the first run. Taking the best of 2,400 noisy numbers and quoting its
own p-value is the most common way a backtest lies. So the test statistic is the MAXIMUM |t|
over the entire family, and the null distribution is the maximum |t| over the same family
recomputed on each permuted market. A cell only wins if it beats what the best of 2,400 pure
noise cells manages.

Both readings are reported: the family-corrected p-value, which is the one that counts, and
each cell's own uncorrected p-value, which is the one that would have been quoted.

THE NULL

Grouped bar permutation (quant.permutation). Intrabar shapes and overnight gaps are shuffled
within (first-bar-of-session, RTH) groups, so the permuted series keeps the real intraday
volatility profile and the real marginal bar distribution, and loses only the serial structure.
That is a harder null than a plain shuffle, which a session-aware indicator can beat on the
volatility profile alone.

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
from quant.strategies.ichimoku import build_line, cross_masks, line_family, mid_bank
from quant.verdict import from_information

HORIZONS = (1, 3, 6, 12, 26, 78)
# Every built and integrity-checked timeframe from 3m to 60m. 1m is excluded on a stated
# assumption, not a measurement: a 0.544 pt MES round turn is roughly a fifth of a 1m ATR,
# so an effect would have to be enormous to survive cost there. Overridable with
# --timeframes, so the set is never silently narrowed again.
ALL_TIMEFRAMES = ("3m", "5m", "10m", "15m", "20m", "30m", "45m", "60m")
TIMEFRAMES = ALL_TIMEFRAMES
SYMBOLS = ("MES", "MNQ")
WARMUP = 400  # exceeds the longest line lookback (120 + 52 displacement)
MIN_EVENTS = 100
RTH_LO, RTH_HI = 8 * 60 + 30, 15 * 60
ATR_N = 14
SPECS = line_family()
N_CELLS = len(SPECS) * len(HORIZONS) * len(TIMEFRAMES) * len(SYMBOLS)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _atr(df: pd.DataFrame) -> np.ndarray:
    """ATR(14), shifted one bar. The R-unit must not know the current bar's own range."""
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    return (
        pd.Series(S.true_range(h, lo, c))
        .ewm(alpha=1 / ATR_N, adjust=False)
        .mean()
        .shift(1)
        .to_numpy()
    )


# --------------------------------------------------------------------------- the statistic
def evaluate(df: pd.DataFrame, inwin: np.ndarray, eligible: np.ndarray) -> dict[str, np.ndarray]:
    """t-statistics for every (line, horizon) cell on one series.

    `inwin` and `eligible` do not depend on prices, so they are computed once and reused across
    every permutation. Everything that does depend on prices is recomputed here.
    """
    c = df["close"].to_numpy(dtype=float)
    n = len(c)
    atr = _atr(df)
    atr_ok = np.isfinite(atr) & (atr > 0)
    bank = mid_bank(df)

    # Event masks: two rows per line configuration (up cross, down cross).
    base = inwin & eligible & atr_ok
    M = np.zeros((2 * len(SPECS), n), dtype=np.float32)
    for j, spec in enumerate(SPECS):
        up, dn = cross_masks(c, build_line(spec, c, bank))
        M[2 * j] = up & base
        M[2 * j + 1] = dn & base

    shape = (len(SPECS), len(HORIZONS))
    t_out = np.full(shape, np.nan)
    d_out = np.full(shape, np.nan)
    n_up = np.zeros(shape, dtype=np.int64)
    n_dn = np.zeros(shape, dtype=np.int64)

    for hi, hor in enumerate(HORIZONS):
        fr = np.full(n, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            fr[: n - hor] = (c[hor:] - c[: n - hor]) / atr[: n - hor]
        ok = np.isfinite(fr)
        f0 = np.where(ok, fr, 0.0).astype(np.float32)
        X = np.empty((n, 3), dtype=np.float32)
        X[:, 0] = f0
        X[:, 1] = f0 * f0
        X[:, 2] = ok
        R = (M @ X).astype(np.float64)  # (2S, 3): sum, sum of squares, count

        cnt = R[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = R[:, 0] / cnt
            var = np.maximum(R[:, 1] / cnt - mean**2, 0.0)
        mu_u, mu_d = mean[0::2], mean[1::2]
        v_u, v_d = var[0::2], var[1::2]
        cu, cd = cnt[0::2], cnt[1::2]
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(v_u / np.maximum(cu - 1, 1) + v_d / np.maximum(cd - 1, 1))
            d = mu_u - mu_d
            t = np.where(se > 0, d / se, np.nan)
        enough = (cu >= MIN_EVENTS) & (cd >= MIN_EVENTS)
        t_out[:, hi] = np.where(enough, t, np.nan)
        d_out[:, hi] = np.where(enough, d, np.nan)
        n_up[:, hi] = cu.astype(np.int64)
        n_dn[:, hi] = cd.astype(np.int64)

    return dict(t=t_out, d=d_out, n_up=n_up, n_dn=n_dn)


def verify_float32_path(df, inwin, eligible, res) -> float:
    """The matmul accumulates in float32. Confirm against an exact float64 recomputation of a
    handful of cells -- a fast statistic that is subtly wrong is worse than a slow one."""
    c = df["close"].to_numpy(dtype=float)
    atr = _atr(df)
    base = inwin & eligible & np.isfinite(atr) & (atr > 0)
    bank = mid_bank(df)
    n = len(c)
    worst = 0.0
    rng = np.random.default_rng(0)
    for j in rng.choice(len(SPECS), size=6, replace=False):
        up, dn = cross_masks(c, build_line(SPECS[j], c, bank))
        up, dn = up & base, dn & base
        for hi, hor in enumerate(HORIZONS):
            fr = np.full(n, np.nan)
            fr[: n - hor] = (c[hor:] - c[: n - hor]) / atr[: n - hor]
            ok = np.isfinite(fr)
            a, b = fr[up & ok], fr[dn & ok]
            if len(a) < MIN_EVENTS or len(b) < MIN_EVENTS:
                continue
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            ref = res["t"][j, hi]
            if np.isfinite(ref) and se > 0:
                worst = max(worst, abs((a.mean() - b.mean()) / se - ref))
    return worst


HDR = (
    f"{'sym':<4}{'tf':<5}{'line':<18}{'h':>4}{'t':>8}{'d(ATR)':>9}"
    f"{'d(pts)':>9}{'cost':>7}{'net':>8}{'p_unc':>8}"
)


def show(frame: pd.DataFrame, n: int) -> None:
    for _, r in frame.head(n).iterrows():
        log(
            f"  {r.symbol:<4}{r.tf:<5}{r.line:<18}{r.horizon:>4}{r.t:>8.2f}"
            f"{r.d_atr:>9.4f}{r.d_pts:>9.3f}{r.cost_pts:>7.2f}"
            f"{r.edge_after_cost:>8.3f}{r.p_uncorrected:>8.4f}"
        )


# --------------------------------------------------------------------------- main
def main() -> int:
    global TIMEFRAMES, N_CELLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--timeframes",
        default=",".join(ALL_TIMEFRAMES),
        help="comma-separated; the default is every built timeframe from 3m to 60m",
    )
    ap.add_argument("--out", default="out/ichimoku_information")
    a = ap.parse_args()

    TIMEFRAMES = tuple(t.strip() for t in a.timeframes.split(",") if t.strip())
    N_CELLS = len(SPECS) * len(HORIZONS) * len(TIMEFRAMES) * len(SYMBOLS)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 92)
    log("  ICHIMOKU LINES: DO PRICE CROSSES CARRY DIRECTIONAL INFORMATION?")
    log("=" * 92)
    log(
        f"  family: {len(SPECS)} lines x {len(HORIZONS)} horizons x {len(TIMEFRAMES)} "
        f"timeframes x {len(SYMBOLS)} symbols = {N_CELLS:,} cells"
    )
    log(f"  permutations: {a.perms}   segment: TRAIN only   holdouts untouched\n")

    h = Hypothesis(
        name="ichimoku_line_information",
        instrument="MES and MNQ, 5m/10m/15m/30m/60m full-session bars",
        session="08:30-15:00 America/Chicago for the EVENT; the forward window may run past it",
        setup="The close crosses an Ichimoku line (mid-Donchian of length n, optionally "
        "displaced d bars; the (Tenkan+Kijun)/2 span line; or close vs close[n]). The cross "
        "is the event, identified at the close of the bar that completes it.",
        direction="both",
        exit_rule="none -- this measures a forward return over a fixed horizon of 1, 3, 6, 12, "
        "26 or 78 bars, not a trade",
        execution="the event is known at the close of bar t; the forward return runs from "
        "close[t] to close[t+h] and is divided by ATR(14) shifted one bar",
        costs="NOT MODELLED, deliberately: there are no trades. The measured effect is "
        "converted to points and compared against the CostModel.for_prop round turn, so a "
        "statistically real but untradeably small effect is visible as such.",
        invalidation="abandon if the family-corrected permutation p-value of the maximum |t| "
        "over all 2,400 cells is >= 0.05. A cell that is significant only on its own "
        "uncorrected p-value is not a finding, it is the best of 2,400 draws.",
        metric="max |t| over the family, where t is the Welch statistic comparing mean forward "
        "return after an up-cross against after a down-cross, in ATR units, on the train "
        "segment only",
        rationale="The lines are midpoints of extremes rather than averages of closes, so they "
        "are not the same object as the EMAs already rejected: a mid-Donchian only moves when "
        "a new extreme enters or leaves the window, which makes it a level that holds still "
        "through chop. The displaced version is slower still. If any line in this family marks "
        "a structural level, the slow displaced one is the candidate.",
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
            inwin = (mins >= RTH_LO) & (mins < RTH_HI)
            eligible = np.arange(n) >= WARMUP
            res = evaluate(df, inwin, eligible)
            real[si, ti] = res["t"]
            dreal[si, ti] = res["d"]
            atr_med = float(np.nanmedian(_atr(df)[WARMUP:]))
            series[(sym, tf)] = dict(
                df=df,
                inwin=inwin,
                eligible=eligible,
                groups=session_groups(df),
                atr_med=atr_med,
            )
            err = verify_float32_path(df, inwin, eligible, res)
            cells = int(np.isfinite(res["t"]).sum())
            log(
                f"  {sym} {tf:>3}: {n:>7,} train bars, {cells:>3}/{res['t'].size} cells with "
                f">={MIN_EVENTS} events, ATR med {atr_med:6.2f} pts, "
                f"float32 check {err:.2e}, {time.time() - t0:.1f}s"
            )
            if err > 1e-3:
                log("  !! float32 statistic disagrees with the exact recomputation -- ABORT")
                return 1

    finite = np.isfinite(real)
    log(f"\n  {finite.sum():,} of {N_CELLS:,} cells have enough events to evaluate")
    obs_max = float(np.nanmax(np.abs(real)))
    idx = np.unravel_index(np.nanargmax(np.abs(real)), real.shape)
    log(
        f"  observed max |t| = {obs_max:.3f}  at  {SYMBOLS[idx[0]]} {TIMEFRAMES[idx[1]]} "
        f"{SPECS[idx[2]].label} h={HORIZONS[idx[3]]}\n"
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
                null[p, si, ti] = evaluate(perm, s["inwin"], s["eligible"])["t"]
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
    )

    # ------------------------------------------------------------------ verdict
    with np.errstate(invalid="ignore"):
        null_max = np.nanmax(np.abs(null.reshape(a.perms, -1)), axis=1)
    p_family = (1 + int((null_max >= obs_max).sum())) / (1 + a.perms)

    log("\n" + "=" * 92)
    log("  RESULT")
    log("=" * 92)
    log(f"  observed max |t| over {finite.sum():,} live cells : {obs_max:.3f}")
    log(
        f"  null max |t|  median {np.median(null_max):.3f}   "
        f"95th pct {np.percentile(null_max, 95):.3f}   max {null_max.max():.3f}"
    )
    log(f"  FAMILY-CORRECTED p = {p_family:.4f}   (reject the family at p >= 0.05)")

    with np.errstate(invalid="ignore"):
        p_cell = (1 + (np.abs(null) >= np.abs(real)[None]).sum(axis=0)) / (1 + a.perms)
    p_cell = np.where(finite, p_cell, np.nan)
    n_nominal = int(np.nansum(p_cell < 0.05))
    log(
        f"\n  cells with uncorrected p < 0.05: {n_nominal} of {finite.sum():,} "
        f"({100 * n_nominal / max(finite.sum(), 1):.1f}%) -- "
        f"chance alone gives ~5.0% ({0.05 * finite.sum():.0f} cells)"
    )

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
                            line=spec.label,
                            classic=spec.is_classic,
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

    log("\n  TOP 12 CELLS BY |t|  (p_unc is the number that would have been quoted)")
    log("  " + HDR)
    show(tab, 12)

    log("\n  THE FIVE CLASSIC LINES (9 / 26 / 52 / 26), zero free parameters")
    log("  " + HDR)
    cl = tab[tab["classic"]].sort_values("t", key=np.abs, ascending=False)
    show(cl, 10)
    log(f"\n  classic cells with |t| > 2: {int((cl['t'].abs() > 2).sum())} of {len(cl)}")

    log("\n  THE LINE ASKED ABOUT -- script 'SenkouA' = midDonchian(52) displaced 26")
    log("  " + HDR)
    sa = tab[tab["line"] == "middisp(52,26)"].sort_values("t", key=np.abs, ascending=False)
    show(sa, 8)

    tradeable = tab[tab["edge_after_cost"] > 0]
    log(f"\n  cells whose measured effect EXCEEDS one round turn: {len(tradeable)} of {len(tab):,}")
    if len(tradeable):
        log(f"    best net: {tradeable['edge_after_cost'].max():.3f} pts")

    summary = dict(
        hypothesis=hh,
        family_cells=N_CELLS,
        live_cells=int(finite.sum()),
        permutations=a.perms,
        observed_max_t=obs_max,
        observed_max_cell=(
            f"{SYMBOLS[idx[0]]} {TIMEFRAMES[idx[1]]} {SPECS[idx[2]].label} h={HORIZONS[idx[3]]}"
        ),
        null_max_t_median=float(np.median(null_max)),
        null_max_t_p95=float(np.percentile(null_max, 95)),
        p_family=p_family,
        nominal_significant_cells=n_nominal,
        expected_by_chance=float(0.05 * finite.sum()),
        cells_beating_cost=len(tradeable),
        verdict="PASS" if p_family < 0.05 else "REJECT",
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
        None,
    )
    log("")
    log(v.render())
    (out / "verdict.txt").write_text(v.render(), encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps(v.as_dict(), indent=2), encoding="utf-8")
    log(f"\n  VERDICT: {summary['verdict']}")
    log(f"  written to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
