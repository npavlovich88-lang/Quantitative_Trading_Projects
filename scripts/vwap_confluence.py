#!/usr/bin/env python3
"""Q2. Does a higher-timeframe VWAP band, at the same sign, improve the session band rejection?

    python scripts/vwap_confluence.py --perms 1000

THE QUESTION

    "if we layer vwap standard deviations 1,2,3,4 to -1,-2,-3,-4 to the session, weekly,
     monthly charts and match the standard deviation with current session standard deviation
     does it provide a better rejection? ... are the chances of it rejecting better than just
     using a session vwap"

So the statistic is a DIFFERENCE, not a level:

    pA(session band touched AND every other anchor also outside its 1 sigma band, same sign)
  - pA(session band touched AND that confluence absent)

Both halves come from the same band, the same anchor, the same session filter and the same
horizon, so the comparison is within-cell and the geometric baseline cancels. A confluence that
adds nothing scores zero here however high its raw pA is -- which matters, because the raw pA at
a 3 sigma band is dominated by geometry and would look impressive on its own.

DECLARED CHOICES, not swept

  * The base anchor is `ny`, resetting at 08:30 America/Chicago. That is what a chart means by
    "current session VWAP" on an index future. The 17:00 anchor is tested separately in Q1.
  * "at a point of interest" means the close is outside the 1 sigma band of that anchor, i.e.
    abs(band index) >= 1. The looser reading in the request -- "or the vwap it self" -- would
    make almost every bar a point of interest on some anchor, so it is not used here and is not
    swept as an alternative.
  * Same-sign is required, as asked: a session lower band pairs only with other lower bands.

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
from quant.data_splits import DATA, make_split
from quant.hypothesis import Hypothesis
from quant.permutation import get_permutation_fast, session_groups
from quant.strategies.vwap_bands import (
    BANDS,
    anchor_ids,
    band_index,
    first_touches,
    in_rth,
    race,
    vwap_bands,
)

SYMBOLS = ("MNQ", "MES")
TFS = ("5m", "15m")
BASE = "ny"
HTF = ("weekly", "monthly", "quarterly")
COMBOS = {
    "S+W": ("weekly",),
    "S+M": ("monthly",),
    "S+W+M": ("weekly", "monthly"),
    "S+Q": ("quarterly",),
    "S+W+M+Q": ("weekly", "monthly", "quarterly"),
}
SIGNS = (-1, 1)
FILTERS = ("rth", "overnight")
HORIZON = {"5m": 78, "15m": 26}


def log(m: str = "") -> None:
    print(m, flush=True)


def cells_for(df: pd.DataFrame, pre: dict) -> dict:
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    cl = df["close"].to_numpy(float)

    mu, sg = vwap_bands(df, pre["gid"][BASE])
    zone = {}
    for a in HTF:
        m2, s2 = vwap_bands(df, pre["gid"][a])
        zone[a] = band_index(m2, s2, cl)

    out: dict[tuple, dict] = {}
    for k in BANDS:
        for sign in SIGNS:
            ev_all = first_touches(mu, sg, hi, lo, pre["day"], k, sign)
            for filt in FILTERS:
                sel = pre["rth"][ev_all] if filt == "rth" else ~pre["rth"][ev_all]
                ev = ev_all[sel]
                for name, anchors in COMBOS.items():
                    agree = np.ones(len(ev), dtype=bool)
                    for a in anchors:
                        z = zone[a][ev]
                        agree &= (np.abs(z) >= 1) & (np.sign(z) == sign)
                    rc = race(df, mu, sg, ev[agree], k, sign, pre["hor"])
                    rn = race(df, mu, sg, ev[~agree], k, sign, pre["hor"])
                    out[(k, sign, filt, name)] = dict(
                        nA_c=rc["nA"],
                        nA_n=rn["nA"],
                        pA_c=rc["pA"],
                        pA_n=rn["pA"],
                        diff=(rc["pA"] - rn["pA"]) if (rc["nA"] and rn["nA"]) else np.nan,
                        share=float(agree.mean()) if len(ev) else np.nan,
                        pC_c=rc["pC"],
                        pC_n=rn["pC"],
                    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=25)
    ap.add_argument("--out", default="out/vwap_confluence")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("  Q2  MULTI-TIMEFRAME VWAP BAND CONFLUENCE: BETTER REJECTION THAN THE SESSION ALONE?")
    log("=" * 100)
    n_cells = len(BANDS) * len(SIGNS) * len(FILTERS) * len(COMBOS) * len(TFS) * len(SYMBOLS)
    log(f"  family: {n_cells} cells   permutations: {a.perms}   TRAIN only")
    log(f"  statistic: pA(confluence) - pA(no confluence), within cell, min n {a.min_n} a side\n")

    h = Hypothesis(
        name="vwap_band_confluence",
        instrument="MNQ and MES, 5m and 15m",
        session="touches split into RTH 08:30-15:00 America/Chicago and overnight",
        setup="first touch per session day of the 08:30-anchored VWAP sign*k sigma band, k in "
        "1..4, split by whether every anchor in the combo (weekly, monthly, quarterly, resetting "
        "at the Sunday session open / first session of the month / of the quarter) simultaneously "
        "has its close outside its own 1 sigma band with the SAME sign.",
        direction="both",
        exit_rule="race to frozen levels: the session VWAP against the session sign*(k+1) band, "
        "horizon one RTH day; a bar covering both is charged to the extension",
        execution="all bands causal (computed from bars strictly before the bar tested); the race "
        "scans from the bar after the touch; higher-timeframe zones read at the touch bar",
        costs="not entered: the statistic is a difference of two probabilities measured on the "
        "same levels, so cost cancels. Q1 carries the breakeven rate.",
        invalidation="abandon if the family-max permutation p of the difference is >= 0.05. "
        "Confluence that merely inherits the geometric baseline scores zero by construction.",
        metric="pA(confluence) - pA(no confluence) per cell, standardised against a per-cell "
        "permutation null; family-max and count statistics over the whole family",
        rationale="A session band is one dispersion-scaled distance from one volume-weighted "
        "mean. If bands carry information, two anchors agreeing should carry more than one, and "
        "the within-cell difference isolates that claim from the geometry that dominates the raw "
        "rate. If the difference is zero, layering timeframes is decoration.",
        author="research@example.invalid",
    )
    hh = h.register("out/hypotheses.jsonl")
    log(f"[pre-registered] {hh}\n")

    store, obs = {}, {}
    for sym in SYMBOLS:
        for tf in TFS:
            df = S.load_data(DATA[sym].format(tf=tf), "America/Chicago")
            d0 = df["dt"].dt.tz_localize(None).to_numpy()
            df = df.iloc[: make_split(sym, tf, d0).train.stop].reset_index(drop=True)
            pre = dict(
                gid={a: anchor_ids(df["dt"], a) for a in (BASE, *HTF)},
                day=anchor_ids(df["dt"], "session"),
                rth=in_rth(df["dt"]),
                hor=HORIZON[tf],
            )
            store[(sym, tf)] = (df, pre, session_groups(df))
            for key, r in cells_for(df, pre).items():
                obs[(sym, tf, *key)] = r
            log(f"  {sym} {tf}: {len(df):,} train bars")

    keys = [
        k
        for k in sorted(obs)
        if obs[k]["nA_c"] >= a.min_n and obs[k]["nA_n"] >= a.min_n and np.isfinite(obs[k]["diff"])
    ]
    log(f"\n  {len(keys)} of {len(obs)} cells have at least {a.min_n} resolved races a side")
    log(
        f"  confluence share of touches: median {100 * np.nanmedian([obs[k]['share'] for k in obs]):.1f}%\n"
    )
    if not keys:
        log("  nothing testable -- confluence never reaches the minimum sample. STOP.")
        return 1

    kidx = {k: i for i, k in enumerate(keys)}
    o = np.array([obs[k]["diff"] for k in keys], dtype=float)

    log(f"  running {a.perms} grouped bar permutations ...")
    null = np.full((a.perms, len(keys)), np.nan)
    t0 = time.time()
    for p in range(a.perms):
        for (sym, tf), (df, pre, g) in store.items():
            pf = get_permutation_fast(
                df, start_index=0, seed=a.seed + p * 1009 + hash(sym + tf) % 997, groups=g
            )
            pf["dt"] = df["dt"]
            for key, r in cells_for(pf, pre).items():
                full = (sym, tf, *key)
                if full in kidx:
                    null[p, kidx[full]] = r["diff"]
        if p == 0 or (p + 1) % 25 == 0:
            el = time.time() - t0
            log(
                f"    {p + 1:>5}/{a.perms}  {el:.0f}s elapsed, ~{el / (p + 1) * (a.perms - p - 1):.0f}s left"
            )

    mu_n = np.nanmean(null, axis=0)
    sd_n = np.nanstd(null, axis=0)
    sd_n[~np.isfinite(sd_n) | (sd_n <= 0)] = np.inf
    t_obs = (o - mu_n) / sd_n
    t_null = (null - mu_n) / sd_n

    fam_obs = float(np.nanmax(np.abs(t_obs)))
    fam_null = np.nanmax(np.abs(t_null), axis=1)
    p_fam = (1 + int((fam_null >= fam_obs).sum())) / (1 + a.perms)

    hi95 = np.nanpercentile(null, 95, axis=0)
    lo05 = np.nanpercentile(null, 5, axis=0)
    cnt_obs = int(((o > hi95) | (o < lo05)).sum())
    cnt_null = np.nansum((null > hi95) | (null < lo05), axis=1)
    p_cnt = (1 + int((cnt_null >= cnt_obs).sum())) / (1 + a.perms)

    rows = []
    for i, k in enumerate(keys):
        r = obs[k]
        pc = (1 + int((np.abs(t_null[:, i]) >= abs(t_obs[i])).sum())) / (1 + a.perms)
        rows.append(
            dict(
                symbol=k[0],
                tf=k[1],
                band=k[2],
                sign=k[3],
                filter=k[4],
                combo=k[5],
                nA_c=r["nA_c"],
                nA_n=r["nA_n"],
                pA_c=r["pA_c"],
                pA_n=r["pA_n"],
                diff=r["diff"],
                null_mean=mu_n[i],
                null_sd=sd_n[i],
                t=t_obs[i],
                p_cell=pc,
                share=r["share"],
            )
        )
    d = pd.DataFrame(rows)
    d.to_csv(out / "cells.csv", index=False)
    np.save(out / "null.npy", null)

    log("\n" + "=" * 100)
    log("  RESULT")
    log("=" * 100)
    log(
        f"  family-max |t| = {fam_obs:.2f}   null 95th {np.percentile(fam_null, 95):.2f}"
        f"   p = {p_fam:.4f}"
    )
    log(
        f"  count      {cnt_obs} of {len(keys)} cells outside their own 5-95 band"
        f"   null median {np.median(cnt_null):.0f}   p = {p_cnt:.4f}"
    )

    log("\n  by combo, averaged over every cell it appears in:")
    log(f"  {'combo':<10}{'cells':>7}{'mean diff':>11}{'mean |t|':>10}{'best t':>9}{'share':>8}")
    for name in COMBOS:
        s = d[d["combo"] == name]
        if s.empty:
            log(f"  {name:<10}{'-- no cell reached the minimum sample --':>45}")
            continue
        log(
            f"  {name:<10}{len(s):>7}{s['diff'].mean():>+11.4f}{s['t'].abs().mean():>10.2f}"
            f"{s.loc[s['t'].abs().idxmax(), 't']:>+9.2f}{100 * s['share'].mean():>7.1f}%"
        )

    log("\n  the eight strongest cells by |t|:")
    log(
        f"  {'sym':<4}{'tf':<4}{'combo':<9}{'band':>5}{'filt':>10}{'n_c':>6}{'n_n':>6}"
        f"{'pA_c':>7}{'pA_n':>7}{'diff':>8}{'t':>7}{'p':>8}"
    )
    for _, r in d.reindex(d["t"].abs().sort_values(ascending=False).index).head(8).iterrows():
        log(
            f"  {r['symbol']:<4}{r['tf']:<4}{r['combo']:<9}"
            f"{int(r['sign']) * int(r['band']):>+5}{r['filter']:>10}{int(r['nA_c']):>6}"
            f"{int(r['nA_n']):>6}{r['pA_c']:>7.3f}{r['pA_n']:>7.3f}{r['diff']:>+8.3f}"
            f"{r['t']:>+7.2f}{r['p_cell']:>8.4f}"
        )

    (out / "summary.json").write_text(
        json.dumps(
            dict(
                hypothesis=hh,
                permutations=a.perms,
                cells=len(keys),
                family_max_t=fam_obs,
                p_family=p_fam,
                count_obs=cnt_obs,
                p_count=p_cnt,
                mean_diff=float(np.nanmean(o)),
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
