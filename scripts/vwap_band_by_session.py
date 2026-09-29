#!/usr/bin/env python3
"""Q5. Once the clock is controlled for, is there any rejection at the VWAP bands at all?

    python scripts/vwap_band_by_session.py --perms 500

WHY THIS RUN EXISTS

Q1 found information at the bands, but the baseline diagnostics then found that an anchored-VWAP
sigma band is largely a statement about ELAPSED SESSION TIME. On MES 5m, three quarters of every
band touch happens before 09:50 and fewer than 12% after noon, because sigma triples across the
session (2.77 points in the first half hour, 8.88 points after 13:00) so the bands are
meaninglessly tight at the open and unreachable by the afternoon.

That makes the Q1 result ambiguous: the 4 sigma extension may be the opening drive wearing a VWAP
costume, and the opening drive is hypothesis 13, which was rejected. This run removes the
ambiguity by bucketing every touch by WHEN it happened and testing each bucket against its own
permutation null.

THE BUCKETS, America/Chicago, all measured against the SAME 08:30-anchored VWAP

    or         08:30-09:00   the 30-minute opening range is still forming
    post_or    09:00-09:30   opening range set, Initial Balance not yet complete
    post_ib    09:30-11:00   after the Initial Balance (first RTH hour)
    midday     11:00-13:00
    pm         13:00-15:00
    evening    17:00-19:00   after the cash close, before Asia
    asia       19:00-02:00
    london     02:00-08:30

The overnight buckets answer the second question directly: the anchor stays at the New York open
and runs until the next one, so an Asian or London touch is measured against the VWAP anchored at
the PREVIOUS New York open. That is the "keep it anchored at New York but use it for Asia and
London" case, not a re-anchored session VWAP.

RIDING VERSUS REJECTING, WITHOUT THE MOVING AVERAGE

Q4 established that an MA cross does not separate the two: the clean statistic was a coin flip at
49.9% over 1010 cells. This run tries the conditioner the baseline work actually points at, and it
is free:

    WICK       the bar reached the band but closed back INSIDE it
    ACCEPT     the bar CLOSED beyond the band

The baseline showed the wick is free -- on pure noise a 4 sigma touch closes 0.90 sigma back
inside -- so a close beyond the band is a materially different event from a wick, and the
difference between them is the acceptance/rejection test in the request, measured rather than
assumed. A second conditioner, whether sigma is still immature (touch inside the first 12 bars of
the anchor), is reported next to it because the clock finding predicts it should matter.

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
from quant.strategies.vwap_bands import BANDS, anchor_ids, race, vwap_bands

SYMBOLS = ("MNQ", "MES")
TFS = ("5m", "15m")
ANCHOR = "ny"
SIGNS = (-1, 1)
HORIZON = {"5m": 78, "15m": 26}
IMMATURE_BARS = 12  # a touch this early in the anchor has a sigma built from very few bars

# name -> (start minute, end minute); a start above the end wraps past midnight
BUCKETS = {
    "or": (8 * 60 + 30, 9 * 60),
    "post_or": (9 * 60, 9 * 60 + 30),
    "post_ib": (9 * 60 + 30, 11 * 60),
    "midday": (11 * 60, 13 * 60),
    "pm": (13 * 60, 15 * 60),
    "evening": (17 * 60, 19 * 60),
    "asia": (19 * 60, 2 * 60),
    "london": (2 * 60, 8 * 60 + 30),
}


def log(m: str = "") -> None:
    print(m, flush=True)


def bucket_mask(mins: np.ndarray, name: str) -> np.ndarray:
    a, b = BUCKETS[name]
    return (mins >= a) | (mins < b) if a > b else (mins >= a) & (mins < b)


def first_per_anchor(idx: np.ndarray, gid: np.ndarray) -> np.ndarray:
    """One event per anchor period, keeping the earliest. Bucketing already restricts the window,
    so this is the first touch WITHIN that bucket rather than the first of the day -- otherwise a
    late bucket would only ever see days on which the band was not reached earlier."""
    if len(idx) == 0:
        return idx
    g = gid[idx]
    return idx[np.r_[True, g[1:] != g[:-1]]]


def cells_for(df: pd.DataFrame, pre: dict) -> dict:
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    mu, sg = vwap_bands(df, pre["gid"])
    out: dict[tuple, dict] = {}
    for k in BANDS:
        for sign in SIGNS:
            level = mu + sign * k * sg
            reach = (hi >= level) if sign > 0 else (lo <= level)
            reach &= np.isfinite(level)
            beyond = (cl > level) if sign > 0 else (cl < level)
            for bname in BUCKETS:
                bm = pre["bucket"][bname]
                ev = first_per_anchor(np.flatnonzero(reach & bm), pre["gid"])
                r = race(df, mu, sg, ev, k, sign, pre["hor"])
                r["sigma"] = float(np.nanmean(sg[ev])) if len(ev) else np.nan
                r["immature"] = (
                    float(np.nanmean(pre["age"][ev] < IMMATURE_BARS)) if len(ev) else np.nan
                )
                out[(bname, k, sign, "all")] = r
                # acceptance conditioner: closed beyond the band, versus wicked and closed back
                acc = first_per_anchor(np.flatnonzero(reach & beyond & bm), pre["gid"])
                wic = first_per_anchor(np.flatnonzero(reach & ~beyond & bm), pre["gid"])
                ra = race(df, mu, sg, acc, k, sign, pre["hor"])
                rw = race(df, mu, sg, wic, k, sign, pre["hor"])
                out[(bname, k, sign, "accept")] = ra
                out[(bname, k, sign, "wick")] = rw
    return out


def fam(o: np.ndarray, null: np.ndarray, perms: int) -> dict:
    mn = np.nanmean(null, axis=0)
    sd = np.nanstd(null, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.inf)
    t_o, t_n = (o - mn) / sd, (null - mn) / sd
    f = float(np.nanmax(np.abs(t_o)))
    fn = np.nanmax(np.abs(t_n), axis=1)
    hi = np.nanpercentile(null, 95, axis=0)
    lq = np.nanpercentile(null, 5, axis=0)
    co = int(((o > hi) | (o < lq)).sum())
    cn = np.nansum((null > hi) | (null < lq), axis=1)
    return dict(
        t=t_o,
        t_null=t_n,
        null_mean=mn,
        fam=f,
        fam_p95=float(np.percentile(fn, 95)),
        p_fam=(1 + int((fn >= f).sum())) / (1 + perms),
        count=co,
        count_med=float(np.median(cn)),
        p_count=(1 + int((cn >= co).sum())) / (1 + perms),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=25)
    ap.add_argument("--max-unres", type=float, default=0.35)
    ap.add_argument("--out", default="out/vwap_band_by_session")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 104)
    log("  Q5  VWAP BANDS BY TIME OF DAY: IS THERE REJECTION ONCE THE CLOCK IS CONTROLLED FOR?")
    log("=" * 104)
    log(f"  anchor: {ANCHOR} (08:30 America/Chicago, running to the next 08:30 -- so the")
    log("  overnight buckets are measured against the PREVIOUS New York open, as asked)")
    log(f"  buckets: {', '.join(BUCKETS)}")
    log(f"  permutations: {a.perms}   TRAIN only, holdouts untouched\n")

    h = Hypothesis(
        name="vwap_band_by_session",
        instrument="MNQ and MES, 5m and 15m",
        session="every touch bucketed by clock time America/Chicago into or, post_or, post_ib, "
        "midday, pm, evening, asia, london -- all against the same 08:30-anchored VWAP",
        setup="first touch WITHIN each bucket, per anchor period, of the 08:30-anchored VWAP "
        "sign*k sigma band, k in 1..4. Split three ways: all touches; touches that CLOSED beyond "
        "the band (accept); touches that reached it and closed back inside (wick).",
        direction="both",
        exit_rule="race to frozen levels: the VWAP against the sign*(k+1) band, horizon one RTH "
        "day of bars; a bar covering both is charged to the extension",
        execution="bands causal, computed from bars strictly before the bar tested; the race scans "
        "from the bar after the touch, so the effective start is the touch bar close",
        costs=f"MNQ {CostModel.for_prop('MNQ', 'average', contracts=1).round_turn_points:.3f} "
        f"pts, MES {CostModel.for_prop('MES', 'average', contracts=1).round_turn_points:.3f} pts, "
        "priced afterwards against the per-cell breakeven rather than entering the statistic",
        invalidation="abandon the time-conditioned reversion reading if the family-max "
        "permutation p over the bucket family is >= 0.05. A pass confined to the 'or' bucket is "
        "treated as the opening drive, not as a band effect, because hypothesis 13 covers it.",
        metric="pA per bucket x band x sign against a per-cell permutation null; and the "
        "accept-minus-wick difference as the acceptance/rejection discriminator",
        rationale="The bands were found to encode elapsed session time: three quarters of touches "
        "occur before 09:50 and sigma triples across the session, so Q1's result cannot be "
        "separated from the opening drive without bucketing by clock. An MA cross was already "
        "shown not to separate riding from rejecting (49.9% over 1010 cells), so the conditioner "
        "tried here is acceptance -- a close beyond the band against a wick that closed back "
        "inside -- which the baseline work showed are materially different events.",
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
            gid = anchor_ids(df["dt"], ANCHOR)
            mins = df["dt"].dt.hour.to_numpy() * 60 + df["dt"].dt.minute.to_numpy()
            first = np.r_[True, gid[1:] != gid[:-1]]
            age = np.arange(len(gid)) - np.flatnonzero(first)[np.cumsum(first) - 1]
            pre = dict(
                gid=gid,
                age=age,
                hor=HORIZON[tf],
                bucket={b: bucket_mask(mins, b) for b in BUCKETS},
            )
            store[(sym, tf)] = (df, pre, session_groups(df))
            for key, r in cells_for(df, pre).items():
                obs[(sym, tf, *key)] = r
            log(f"  {sym} {tf}: {len(df):,} train bars")

    def testable(arm: str) -> list:
        return [
            k
            for k in sorted(obs)
            if k[5] == arm
            and obs[k]["nA"] >= a.min_n
            and np.isfinite(obs[k]["pA"])
            and (obs[k]["unres"] <= a.max_unres if np.isfinite(obs[k]["unres"]) else False)
        ]

    k_all, k_acc, k_wic = testable("all"), testable("accept"), testable("wick")
    keys = k_all + k_acc + k_wic
    kidx = {k: i for i, k in enumerate(keys)}
    log(f"\n  {len(k_all)} bucket cells, {len(k_acc)} accept and {len(k_wic)} wick cells testable")
    log(f"  (at least {a.min_n} resolved races and at most {100 * a.max_unres:.0f}% unresolved)\n")

    o = np.array([obs[k]["pA"] for k in keys], dtype=float)
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
                    null[p, kidx[full]] = r["pA"]
        if p == 0 or (p + 1) % 25 == 0:
            el = time.time() - t0
            log(
                f"    {p + 1:>5}/{a.perms}  {el:.0f}s elapsed, "
                f"~{el / (p + 1) * (a.perms - p - 1):.0f}s left"
            )

    na, nacc, nwic = len(k_all), len(k_acc), len(k_wic)
    sa = fam(o[:na], null[:, :na], a.perms)
    s_acc = fam(o[na : na + nacc], null[:, na : na + nacc], a.perms) if nacc else None
    s_wic = fam(o[na + nacc :], null[:, na + nacc :], a.perms) if nwic else None

    # ---- the acceptance discriminator, done so the geometry cancels.
    # A bar that CLOSES beyond the band starts further from the mean than one that wicked and
    # closed back inside, so the raw difference of the two reversion rates is mostly a restatement
    # of where each arm started: measured that way it was negative in 188 of 188 cells. Comparing
    # each arm to ITS OWN permutation null removes that, because each null carries the same
    # starting geometry as its arm. The statistic is therefore a difference of deviations.
    acc_map = {k[:5]: i for i, k in enumerate(k_acc)}
    wic_map = {k[:5]: i for i, k in enumerate(k_wic)}
    pairs = sorted(set(acc_map) & set(wic_map))
    dev_rows, dev_obs, dev_null = [], [], []
    for key in pairs:
        ia, iw = acc_map[key], wic_map[key]
        ca, cw = na + ia, na + nacc + iw
        da_ = o[ca] - np.nanmean(null[:, ca])
        dw_ = o[cw] - np.nanmean(null[:, cw])
        dev_obs.append(da_ - dw_)
        dev_null.append(
            (null[:, ca] - np.nanmean(null[:, ca])) - (null[:, cw] - np.nanmean(null[:, cw]))
        )
        dev_rows.append(key)
    dev_obs = np.asarray(dev_obs)
    dev_null = np.asarray(dev_null).T if len(dev_rows) else np.zeros((a.perms, 0))
    s_dev = fam(dev_obs, dev_null, a.perms) if len(dev_rows) else None

    rows = []
    for i, k in enumerate(keys):
        r = obs[k]
        if i < na:
            s, j = sa, i
        elif i < na + nacc:
            s, j = s_acc, i - na
        else:
            s, j = s_wic, i - na - nacc
        pc = (1 + int((np.abs(s["t_null"][:, j]) >= abs(s["t"][j])).sum())) / (1 + a.perms)
        rows.append(
            dict(
                symbol=k[0],
                tf=k[1],
                bucket=k[2],
                band=k[3],
                sign=k[4],
                arm=k[5],
                signed=k[4] * k[3],
                n=r["n"],
                nA=r["nA"],
                pA=r["pA"],
                null_mean=s["null_mean"][j],
                t=s["t"][j],
                p_cell=pc,
                unres=r["unres"],
                sigma=r.get("sigma", np.nan),
                immature=r.get("immature", np.nan),
            )
        )
    d = pd.DataFrame(rows)
    d.to_csv(out / "cells.csv", index=False)
    np.save(out / "null.npy", null)

    log("\n" + "=" * 104)
    log("  RESULT")
    log("=" * 104)
    log("  pA by time bucket (all touches):")
    log(
        f"    family-max |t| = {sa['fam']:.2f}   null 95th {sa['fam_p95']:.2f}"
        f"   p = {sa['p_fam']:.4f}"
    )
    log(
        f"    count {sa['count']} of {na} outside their own 5-95 band"
        f"   null median {sa['count_med']:.0f}   p = {sa['p_count']:.4f}"
    )
    if s_dev:
        log("  ACCEPTANCE DISCRIMINATOR, close-beyond minus wick, each against its own null:")
        log(
            f"    family-max |t| = {s_dev['fam']:.2f}   null 95th {s_dev['fam_p95']:.2f}"
            f"   p = {s_dev['p_fam']:.4f}"
        )
        log(
            f"    count {s_dev['count']} of {len(dev_rows)} outside their own 5-95 band"
            f"   null median {s_dev['count_med']:.0f}   p = {s_dev['p_count']:.4f}"
        )
        log(
            f"    {int((dev_obs > 0).sum())} of {len(dev_obs)} positive = "
            f"{100 * (dev_obs > 0).mean():.1f}%, mean {dev_obs.mean():+.4f}"
        )
        dv = pd.DataFrame(dev_rows, columns=["symbol", "tf", "bucket", "band", "sign"])
        dv["dev_diff"] = dev_obs
        dv["t"] = s_dev["t"]
        dv.to_csv(out / "acceptance.csv", index=False)
        log()
        log("    by bucket (positive = a close beyond the band beats its null by MORE):")
        log(f"    {'bucket':<10}{'cells':>7}{'mean dev diff':>15}{'mean t':>9}{'frac +':>8}")
        for b in BUCKETS:
            sb = dv[dv["bucket"] == b]
            if sb.empty:
                continue
            log(
                f"    {b:<10}{len(sb):>7}{sb['dev_diff'].mean():>+15.4f}"
                f"{sb['t'].mean():>+9.2f}{100 * (sb['dev_diff'] > 0).mean():>7.0f}%"
            )

    da = d[d["arm"] == "all"]
    log("\n  BY BUCKET, all bands pooled  (t signed: + reversion above null, - extension):")
    log(
        f"  {'bucket':<10}{'cells':>7}{'touches':>9}{'races':>8}{'pA':>8}{'null':>8}"
        f"{'mean t':>9}{'t>+2':>6}{'t<-2':>6}{'sigma':>8}{'immature':>10}"
    )
    for b in BUCKETS:
        s = da[da["bucket"] == b]
        if s.empty:
            log(f"  {b:<10}{'-- no cell passed the sample and resolution gates --':>50}")
            continue
        log(
            f"  {b:<10}{len(s):>7}{int(s['n'].sum()):>9}{int(s['nA'].sum()):>8}"
            f"{s['pA'].mean():>8.3f}{s['null_mean'].mean():>8.3f}{s['t'].mean():>+9.2f}"
            f"{int((s['t'] > 2).sum()):>6}{int((s['t'] < -2).sum()):>6}"
            f"{s['sigma'].mean():>8.2f}{100 * s['immature'].mean():>9.0f}%"
        )

    log("\n  BY BUCKET x BAND, mean t  (rows are buckets, columns the signed band):")
    pv = da.pivot_table(index="bucket", columns="signed", values="t", aggfunc="mean")
    pv = pv.reindex([b for b in BUCKETS if b in pv.index])
    log("  " + pv.to_string(float_format=lambda v: f"{v:+6.2f}").replace("\n", "\n  "))

    dd = d[d["arm"] == "acc_minus_wick"]
    if not dd.empty:
        log("\n  ACCEPT MINUS WICK by bucket  (positive = a close beyond the band reverts MORE):")
        log(f"  {'bucket':<10}{'cells':>7}{'mean diff':>11}{'mean t':>9}{'p<0.05':>8}")
        for b in BUCKETS:
            s = dd[dd["bucket"] == b]
            if s.empty:
                continue
            log(
                f"  {b:<10}{len(s):>7}{s['pA'].mean():>+11.4f}{s['t'].mean():>+9.2f}"
                f"{int((s['p_cell'] < 0.05).sum()):>8}"
            )
        log(
            f"  overall: {int((dd['pA'] > 0).sum())} of {len(dd)} cells positive "
            f"= {100 * (dd['pA'] > 0).mean():.1f}%, mean {dd['pA'].mean():+.4f}"
        )

    log("\n  the twelve strongest bucket cells by |t|:")
    log(
        f"  {'sym':<4}{'tf':<4}{'bucket':<10}{'band':>5}{'nA':>6}{'pA':>7}{'null':>7}"
        f"{'t':>7}{'p':>8}{'unres':>7}{'sigma':>8}"
    )
    for _, r in da.reindex(da["t"].abs().sort_values(ascending=False).index).head(12).iterrows():
        log(
            f"  {r['symbol']:<4}{r['tf']:<4}{r['bucket']:<10}{int(r['signed']):>+5}"
            f"{int(r['nA']):>6}{r['pA']:>7.3f}{r['null_mean']:>7.3f}{r['t']:>+7.2f}"
            f"{r['p_cell']:>8.4f}{r['unres']:>7.2f}{r['sigma']:>8.2f}"
        )

    verdict = "PASS" if min(sa["p_fam"], sa["p_count"]) < 0.05 else "REJECT"
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                hypothesis=hh,
                permutations=a.perms,
                bucket_cells=na,
                accept_cells=nacc,
                wick_cells=nwic,
                bucket_family_max_t=sa["fam"],
                bucket_p_family=sa["p_fam"],
                bucket_count=sa["count"],
                bucket_p_count=sa["p_count"],
                accept_p_family=(s_dev["p_fam"] if s_dev else None),
                accept_p_count=(s_dev["p_count"] if s_dev else None),
                accept_mean_dev_diff=float(dev_obs.mean()) if len(dev_obs) else None,
                accept_frac_pos=float((dev_obs > 0).mean()) if len(dev_obs) else None,
                verdict=verdict,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"\n  VERDICT: {verdict}")
    log(f"  written to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
