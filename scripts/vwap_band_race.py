#!/usr/bin/env python3
"""Q1. At a VWAP standard-deviation band, does price revert to the mean or extend?

    python scripts/vwap_band_race.py --perms 200

THE QUESTION, RESTATED SO IT IS ANSWERABLE

    "if we reject off the -2 standard deviation does it hit the mean of vwap more often
     or go to +2 standard deviation"

From -2 sigma both the mean and +2 sigma are ABOVE price, so that is not a race and "the mean
more often" is true by construction. The three races that are answerable:

    A  REVERSION   the VWAP before the -(k+1) band
    B  TRAVERSE    having reached the VWAP, the +k band before the -(k+1) band
    C  FULL        the +k band before the -(k+1) band, straight from the touch

WHAT THE NULL IS, AND WHY IT IS NOT ONE HALF OR ONE THIRD

Measured on a driftless Gaussian walk with the real volume shape (scripts/vwap_null_diagnostic):
pA is 0.43 at the 1 and 2 sigma bands and 0.38 at 3 and 4 sigma. The naive 1/(k+1) is wrong
because the event fires on the bar LOW but the race resumes on the next bar, so the walk
restarts from the touch bar CLOSE, which on NOISE ALONE already sits 0.28 sigma back inside the
2 sigma band and 0.90 sigma back inside the 4 sigma band. Gambler's ruin on those actual
distances reproduces the simulated pA to within 0.02 at every band.

The practical consequence is worth stating plainly: the "rejection candle" at an extreme band is
free. It appears in pure noise, and it gets bigger the further out the band is.

So the reference here is the PERMUTATION null, which is self-calibrating: each permuted market
gets its own VWAP, its own sigma, its own bands, its own touches and its own races.

THE FAMILY

  3 anchors (ny, session, continuous) x 4 bands x 2 signs x 2 session filters x 2 timeframes
  x 2 symbols = 192 cells. Reported three ways, because one number cannot carry this:

    per-cell p       uncorrected, for reading the shape of the result
    family-max p     max |t| over all 192 against the same max on each permuted market
    count p          how many cells beat their own per-cell 95th percentile, against the same

BREAKEVEN, WHICH IS THE ONLY REASON A PROBABILITY MATTERS

Beating the null is not the same as being tradeable. Race A wins d_mean sigma and loses d_ext
sigma, both measured from the touch bar close, so after costs the required rate is

    pA* = (d_ext * sigma + C) / ((d_mean + d_ext) * sigma)

reported per cell alongside pA. A cell can beat its null and still sit below pA*.

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
from quant.strategies.vwap_bands import (
    BANDS,
    anchor_ids,
    first_touches,
    in_rth,
    race,
    vwap_bands,
)

SYMBOLS = ("MNQ", "MES")
TFS = ("5m", "15m")
ANCHORS = ("ny", "session", "continuous")
SIGNS = (-1, 1)
FILTERS = ("rth", "overnight")
HORIZON = {"5m": 78, "15m": 26}  # one RTH day on each frame


def log(m: str = "") -> None:
    print(m, flush=True)


def cells_for(df: pd.DataFrame, cost: float, pre: dict) -> dict:
    """Every cell for one (symbol, timeframe) series. `pre` carries the timestamp-derived
    arrays, which are identical between the real market and any permutation of it."""
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    out: dict[tuple, dict] = {}
    for anc in ANCHORS:
        mu, sg = vwap_bands(df, pre["gid"][anc])
        for k in BANDS:
            for sign in SIGNS:
                ev_all = first_touches(mu, sg, hi, lo, pre["day"], k, sign)
                for filt in FILTERS:
                    m = pre["rth"][ev_all] if filt == "rth" else ~pre["rth"][ev_all]
                    ev = ev_all[m]
                    r = race(df, mu, sg, ev, k, sign, pre["hor"])
                    if r["nA"] > 0:
                        e = ev[(ev + pre["hor"]) < len(df)]
                        s = sg[e]
                        dm = np.abs(mu[e] - cl[e]) / s
                        de = np.abs(cl[e] - (mu[e] + sign * (k + 1) * s)) / s
                        r["d_mean"] = float(dm.mean())
                        r["d_ext"] = float(de.mean())
                        r["sigma"] = float(s.mean())
                        denom = (dm + de) * s
                        r["pA_be"] = float(((de * s + cost) / denom).mean())
                    out[(anc, k, sign, filt)] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=25)
    ap.add_argument("--out", default="out/vwap_band_race")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("  Q1  VWAP STANDARD-DEVIATION BANDS: REVERT TO THE MEAN, OR EXTEND?")
    log("=" * 100)
    n_cells = len(ANCHORS) * len(BANDS) * len(SIGNS) * len(FILTERS) * len(TFS) * len(SYMBOLS)
    log(f"  family: {n_cells} cells   permutations: {a.perms}   TRAIN only, holdouts untouched")
    log("  null baseline on a driftless walk: pA 0.43 at 1-2 sigma, 0.38 at 3-4 sigma")
    log("  (the permutation null below is the reference; that figure is only a sanity anchor)\n")

    h = Hypothesis(
        name="vwap_band_race",
        instrument="MNQ and MES, 5m and 15m",
        session="touches split into RTH 08:30-15:00 America/Chicago and overnight",
        setup="first touch per session day of the anchored-VWAP sign*k sigma band, k in 1..4, "
        "for anchors resetting at 08:30 (ny), at 17:00 (session), and never (continuous). "
        "Bands are volume-weighted about the anchored VWAP of typical price and are computed "
        "from bars STRICTLY BEFORE the bar tested for the touch.",
        direction="both",
        exit_rule="race to frozen levels: the VWAP against the sign*(k+1) band (A); the mirror "
        "-sign*k band against the sign*(k+1) band after reaching the VWAP (B) and directly (C). "
        "Horizon one RTH day. A bar covering both targets is charged to the EXTENSION.",
        execution="levels frozen at the touch bar; the race scans from the bar AFTER the touch, "
        "so the effective start is the touch bar close, which is reported as d_mean and d_ext.",
        costs=f"MNQ {CostModel.for_prop('MNQ', 'average', contracts=1).round_turn_points:.3f} "
        f"pts, MES {CostModel.for_prop('MES', 'average', contracts=1).round_turn_points:.3f} "
        "pts, entering the breakeven rate pA_be rather than a P&L",
        invalidation="abandon the reversion reading if the family-max permutation p is >= 0.05. "
        "A pass carried by a single cell goes to one pre-committed validate look, not a sweep.",
        metric="pA, the share of resolved races reaching the VWAP before the next band out, "
        "standardised against a per-cell permutation null; family-max and count statistics",
        rationale="These markets were measured to mean-revert intraday (98% of 64 variance-ratio "
        "cells below 1.0, p=0.0033) while twelve trend-continuation hypotheses were rejected. A "
        "VWAP band is a dispersion-scaled distance from a volume-weighted mean, so it is the "
        "natural place to look for that reversion, and it is the first hypothesis here built in "
        "the direction the variance-ratio evidence actually points.",
        author="research@example.invalid",
    )
    hh = h.register("out/hypotheses.jsonl")
    log(f"[pre-registered] {hh}\n")

    store, obs = {}, {}
    for sym in SYMBOLS:
        cost = CostModel.for_prop(sym, "average", contracts=1).round_turn_points
        for tf in TFS:
            df = S.load_data(DATA[sym].format(tf=tf), "America/Chicago")
            d0 = df["dt"].dt.tz_localize(None).to_numpy()
            df = df.iloc[: make_split(sym, tf, d0).train.stop].reset_index(drop=True)
            pre = dict(
                gid={anc: anchor_ids(df["dt"], anc) for anc in ANCHORS},
                day=anchor_ids(df["dt"], "session"),
                rth=in_rth(df["dt"]),
                hor=HORIZON[tf],
            )
            store[(sym, tf)] = (df, cost, pre, session_groups(df))
            c = cells_for(df, cost, pre)
            for key, r in c.items():
                obs[(sym, tf, *key)] = r
            log(f"  {sym} {tf}: {len(df):,} train bars, {len(c)} cells")

    all_keys = sorted(obs)
    keys = [k for k in all_keys if obs[k]["nA"] >= a.min_n]
    kidx = {k: i for i, k in enumerate(keys)}
    nA = np.array([obs[k]["nA"] for k in keys])
    o = np.array([obs[k]["pA"] for k in keys], dtype=float)
    log(f"\n  {len(keys)} of {len(all_keys)} cells have at least {a.min_n} resolved races")
    log(f"  {nA.sum():,} resolved races in the tested family")
    log("  cells below the minimum are dropped BEFORE the family correction: a null with almost")
    log("  no spread manufactures a large |t| from a sample of two and would set the family max\n")

    log(f"  running {a.perms} grouped bar permutations (full recompute per draw) ...")
    null = np.full((a.perms, len(keys)), np.nan)
    t0 = time.time()
    for p in range(a.perms):
        for (sym, tf), (df, cost, pre, g) in store.items():
            pf = get_permutation_fast(
                df, start_index=0, seed=a.seed + p * 1009 + hash(sym + tf) % 997, groups=g
            )
            pf["dt"] = df["dt"]
            c = cells_for(pf, cost, pre)
            for key, r in c.items():
                full = (sym, tf, *key)
                if full in kidx:
                    null[p, kidx[full]] = r["pA"]
        if p == 0 or (p + 1) % 10 == 0:
            el = time.time() - t0
            log(
                f"    {p + 1:>4}/{a.perms}  {el:.0f}s elapsed, ~{el / (p + 1) * (a.perms - p - 1):.0f}s left"
            )

    mu_n = np.nanmean(null, axis=0)
    sd_n = np.nanstd(null, axis=0)
    sd_n[~np.isfinite(sd_n) | (sd_n <= 0)] = np.inf
    t_obs = (o - mu_n) / sd_n
    t_null = (null - mu_n) / sd_n

    fam_obs = np.nanmax(np.abs(t_obs))
    fam_null = np.nanmax(np.abs(t_null), axis=1)
    p_fam = (1 + int((fam_null >= fam_obs).sum())) / (1 + a.perms)

    hi95 = np.nanpercentile(null, 95, axis=0)
    lo05 = np.nanpercentile(null, 5, axis=0)
    cnt_obs = int(((o > hi95) | (o < lo05)).sum())
    cnt_null = ((null > hi95) | (null < lo05)).sum(axis=1)
    p_cnt = (1 + int((cnt_null >= cnt_obs).sum())) / (1 + a.perms)

    rows = []
    for i, k in enumerate(keys):
        r = obs[k]
        pc = (1 + int((np.abs(t_null[:, i]) >= abs(t_obs[i])).sum())) / (1 + a.perms)
        rows.append(
            dict(
                symbol=k[0],
                tf=k[1],
                anchor=k[2],
                band=k[3],
                sign=k[4],
                filter=k[5],
                n=r["n"],
                nA=r["nA"],
                pA=r["pA"],
                null_mean=mu_n[i],
                null_sd=sd_n[i],
                t=t_obs[i],
                p_cell=pc,
                pA_be=r.get("pA_be", np.nan),
                d_mean=r.get("d_mean", np.nan),
                d_ext=r.get("d_ext", np.nan),
                sigma=r.get("sigma", np.nan),
                pB=r["pB"],
                nB=r["nB"],
                pC=r["pC"],
                nC=r["nC"],
                unres=r["unres"],
                ambig=r["ambig"],
            )
        )
    d = pd.DataFrame(rows)
    d.to_csv(out / "cells.csv", index=False)
    np.save(out / "null.npy", null)

    log("\n" + "=" * 100)
    log("  RESULT")
    log("=" * 100)
    log(
        f"  family-max  |t| = {fam_obs:.2f}   null 95th {np.percentile(fam_null, 95):.2f}"
        f"   p = {p_fam:.4f}"
    )
    log(
        f"  count       {cnt_obs} of {len(keys)} cells outside their own 5-95 band"
        f"   null median {np.median(cnt_null):.0f}   p = {p_cnt:.4f}"
    )
    log(f"  expected by chance at 10% two-sided: {0.10 * len(keys):.0f}")

    log("\n  the ten strongest cells by |t|:")
    log(
        f"  {'sym':<4}{'tf':<4}{'anchor':<11}{'band':>5}{'filt':>10}{'nA':>6}"
        f"{'pA':>7}{'null':>7}{'t':>7}{'p':>8}{'pA_be':>7}"
    )
    for _, r in d.reindex(d["t"].abs().sort_values(ascending=False).index).head(10).iterrows():
        log(
            f"  {r['symbol']:<4}{r['tf']:<4}{r['anchor']:<11}"
            f"{int(r['sign']) * int(r['band']):>+5}{r['filter']:>10}{int(r['nA']):>6}"
            f"{r['pA']:>7.3f}{r['null_mean']:>7.3f}{r['t']:>+7.2f}{r['p_cell']:>8.4f}"
            f"{r['pA_be']:>7.3f}"
        )

    (out / "summary.json").write_text(
        json.dumps(
            dict(
                hypothesis=hh,
                permutations=a.perms,
                cells=len(keys),
                resolved_races=int(nA.sum()),
                family_max_t=float(fam_obs),
                p_family=p_fam,
                count_obs=cnt_obs,
                p_count=p_cnt,
                verdict="PASS" if min(p_fam, p_cnt) < 0.05 else "REJECT",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"\n  VERDICT: {'PASS' if min(p_fam, p_cnt) < 0.05 else 'REJECT'}")
    log(f"  written to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
