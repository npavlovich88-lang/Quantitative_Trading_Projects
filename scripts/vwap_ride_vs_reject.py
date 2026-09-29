#!/usr/bin/env python3
"""Q4. At a VWAP band, does a moving-average cross tell riding apart from rejecting?

    python scripts/vwap_ride_vs_reject.py --perms 600

THE QUESTION

    "we know that we have a tendency to ride the stv bands when we have a trending day, but how
     can we tell we are riding the bands vs rejecting off ... if we reject off a +2 stv but ema
     hasnt crossed we are still and could be riding the bands? And if we do cross the 20/50 and
     price has rejected off the stv that we are now more sure that price has really rejected"

That is a claim about DISCRIMINATION, not about the cross on its own. So the statistic is a
difference between two arms of the same band touch:

    A   a cross happens within W bars after the touch, in the reverting direction
    B   the same touch, no such cross within W bars
    C   the same cross ANYWHERE in the series, with no band condition at all

A - B is the question as asked: does the cross separate the touches that revert from the ones
that keep riding. A - C is the confound check that hypotheses 1-12 make mandatory: twelve
moving-average families were already rejected unconditionally, so if a cross at a band performs
like a cross anywhere, the band is contributing nothing and the result is just the old rejected
signal wearing a VWAP costume.

OUTCOME

Forward close-to-close return from the signal bar, in ATR(14) units, at 2, 3, 6, 12 and 24 bars
-- 10, 15, 30, 60 and 120 minutes on the 5m frame, which is the window the request names. Signed
so POSITIVE always means the reversion worked: a lower-band touch scores +1 for price going up.

Return rather than "did it exceed the lowest price we have seen": maximum favourable excursion and
maximum adverse excursion were already measured to be the SAME variable under range normalisation
here (t-stats matched to 1.9e-05), so an MFE framing would add a column and no information.

MA FAMILIES -- four, declared, not swept

    HMA(9) / EMA(43)     the pair named in the request
    EMA(20) / EMA(50)    the pair named in the request
    TEMA(9) / EMA(21)    a lower-lag smoother against a conventional one
    MACD(12,26) / signal(9)

"Could be any length and any combo" is an invitation to sweep, and a sweep is exactly what the
permutation test charges for. Four declared pairs keep the noise ceiling where it can be read.

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
from quant.strategies.ma_pairs import hma
from quant.strategies.vwap_bands import (
    BANDS,
    anchor_ids,
    first_touches,
    in_rth,
    vwap_bands,
)

SYMBOLS = ("MNQ", "MES")
TFS = ("5m", "15m")
ANCHOR = "ny"
SIGNS = (-1, 1)
FILTERS = ("rth", "overnight")
HORIZONS = (2, 3, 6, 12, 24)  # 10, 15, 30, 60, 120 minutes on the 5m frame
WINDOW = 12  # bars after the touch in which the cross must occur
ATR_LEN = 14
PAIRS = ("HMA9_EMA43", "EMA20_EMA50", "TEMA9_EMA21", "MACD")


def log(m: str = "") -> None:
    print(m, flush=True)


def atr_shifted(df: pd.DataFrame, n: int = ATR_LEN) -> np.ndarray:
    h, lo, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    tr[0] = h[0] - lo[0]
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().shift(1).to_numpy()


def ma_pair(close: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "HMA9_EMA43":
        return hma(close, 9), S.ema(close, 43)
    if name == "EMA20_EMA50":
        return S.ema(close, 20), S.ema(close, 50)
    if name == "TEMA9_EMA21":
        return S.tema(close, 9), S.ema(close, 21)
    if name == "MACD":
        line = S.ema(close, 12) - S.ema(close, 26)
        return line, S.ema(line, 9)
    raise ValueError(name)


def cross_bars(fast: np.ndarray, slow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays: fast crossed above slow at this bar, and below."""
    d = fast - slow
    p = np.r_[np.nan, d[:-1]]
    up = (d > 0) & (p <= 0) & np.isfinite(p)
    dn = (d < 0) & (p >= 0) & np.isfinite(p)
    return up, dn


def fwd(close: np.ndarray, atr: np.ndarray, bars: np.ndarray, h: int, sign: int) -> np.ndarray:
    """Forward return in ATR units, signed so positive means the reversion worked."""
    ok = (bars + h) < len(close)
    b = bars[ok]
    r = (close[b + h] - close[b]) / atr[b] * (-sign)
    return r[np.isfinite(r)]


def cells_for(df: pd.DataFrame, pre: dict) -> dict:
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    atr = atr_shifted(df)
    mu, sg = vwap_bands(df, pre["gid"])

    out: dict[tuple, dict] = {}
    for pair in PAIRS:
        f, s = ma_pair(cl, pair)
        up, dn = cross_bars(f, s)
        for k in BANDS:
            for sign in SIGNS:
                want = up if sign < 0 else dn  # a lower-band touch reverts by crossing UP
                ev_all = first_touches(mu, sg, hi, lo, pre["day"], k, sign)
                for filt in FILTERS:
                    m = pre["rth"][ev_all] if filt == "rth" else ~pre["rth"][ev_all]
                    ev = ev_all[m]
                    ev = ev[(ev + WINDOW + max(HORIZONS)) < len(df)]
                    if len(ev) == 0:
                        for h in HORIZONS:
                            out[(pair, k, sign, filt, h)] = dict(
                                nA=0,
                                nB=0,
                                nC=0,
                                mA=np.nan,
                                mB=np.nan,
                                dA_B=np.nan,
                                dA_C=np.nan,
                            )
                        continue
                    # first qualifying cross in (e, e+WINDOW]
                    w = ev[:, None] + 1 + np.arange(WINDOW)[None, :]
                    hitm = want[w]
                    got = hitm.any(axis=1)
                    sigbar = ev + 1 + hitm.argmax(axis=1)
                    a_bars = sigbar[got]
                    b_bars = (ev + WINDOW)[~got]
                    # arm C: the same cross with no band condition, same session filter
                    cmask = want & (pre["rth"] if filt == "rth" else ~pre["rth"])
                    c_bars = np.flatnonzero(cmask)
                    for h in HORIZONS:
                        ra = fwd(cl, atr, a_bars, h, sign)
                        rb = fwd(cl, atr, b_bars, h, sign)
                        rc = fwd(cl, atr, c_bars, h, sign)
                        mA = float(ra.mean()) if len(ra) else np.nan
                        mB = float(rb.mean()) if len(rb) else np.nan
                        mC = float(rc.mean()) if len(rc) else np.nan
                        out[(pair, k, sign, filt, h)] = dict(
                            nA=len(ra),
                            nB=len(rb),
                            nC=len(rc),
                            mA=mA,
                            mB=mB,
                            dA_B=mA - mB if np.isfinite(mA) and np.isfinite(mB) else np.nan,
                            dA_C=mA - mC if np.isfinite(mA) and np.isfinite(mC) else np.nan,
                        )
    return out


def family_stats(o: np.ndarray, null: np.ndarray, perms: int):
    mu_n = np.nanmean(null, axis=0)
    sd_n = np.nanstd(null, axis=0)
    sd_n = np.where(np.isfinite(sd_n) & (sd_n > 0), sd_n, np.inf)
    t_o = (o - mu_n) / sd_n
    t_n = (null - mu_n) / sd_n
    fam = float(np.nanmax(np.abs(t_o)))
    fam_n = np.nanmax(np.abs(t_n), axis=1)
    p_fam = (1 + int((fam_n >= fam).sum())) / (1 + perms)
    hi = np.nanpercentile(null, 95, axis=0)
    loq = np.nanpercentile(null, 5, axis=0)
    c_o = int(((o > hi) | (o < loq)).sum())
    c_n = np.nansum((null > hi) | (null < loq), axis=1)
    p_cnt = (1 + int((c_n >= c_o).sum())) / (1 + perms)
    return dict(
        t=t_o,
        t_null=t_n,
        null_mean=mu_n,
        null_sd=sd_n,
        fam=fam,
        fam_null_p95=float(np.percentile(fam_n, 95)),
        p_fam=p_fam,
        count=c_o,
        count_null_med=float(np.median(c_n)),
        p_count=p_cnt,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=25)
    ap.add_argument("--out", default="out/vwap_ride_vs_reject")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 104)
    log("  Q4  AT A VWAP BAND, DOES AN MA CROSS SEPARATE RIDING FROM REJECTING?")
    log("=" * 104)
    n = len(PAIRS) * len(BANDS) * len(SIGNS) * len(FILTERS) * len(HORIZONS) * len(TFS)
    log(f"  family: {n * len(SYMBOLS)} cells   window {WINDOW} bars   permutations: {a.perms}")
    log("  statistic: mean forward return in ATR units, signed so + means the reversion worked")
    log("  arms: A cross after a touch | B touch with no cross | C the cross with no band\n")

    h = Hypothesis(
        name="vwap_band_ride_vs_reject",
        instrument="MNQ and MES, 5m and 15m",
        session="touches split into RTH 08:30-15:00 America/Chicago and overnight",
        setup="first touch per session day of the 08:30-anchored VWAP sign*k sigma band, k in "
        "1..4, then split by whether one of four declared MA pairs (HMA9/EMA43, EMA20/EMA50, "
        f"TEMA9/EMA21, MACD 12/26/9) crossed in the reverting direction within {WINDOW} bars.",
        direction="both",
        exit_rule="none -- the outcome is the forward close-to-close return in ATR(14) units at "
        "2, 3, 6, 12 and 24 bars from the signal bar, signed so positive means reversion worked",
        execution="bands causal; MA values at the cross bar use closes up to that bar and the "
        "return is measured from that same close forward; ATR(14) shifted one bar",
        costs="not entered: the statistic is a difference of mean ATR-normalised returns between "
        "two arms of the same event, so a constant per-trade cost cancels. A positive result "
        "would then be priced in points before any claim of tradeability.",
        invalidation="abandon if the family-max permutation p of A-B is >= 0.05. A pass whose "
        "A-C is also near zero is rejected regardless: that is an unconditional MA cross, and "
        "twelve of those have already been rejected here.",
        metric="mean(A) - mean(B) per cell, standardised against a per-cell permutation null; "
        "family-max and count statistics; A-C reported as the confound check",
        rationale="Twelve MA hypotheses were rejected unconditionally. The claim being tested is "
        "not that the cross predicts, but that it DISCRIMINATES -- that at a band, a cross marks "
        "the touches that reverse and its absence marks the ones still trending. That is a "
        "conditional claim the earlier tests never measured, and Q1 found the bands do carry "
        "information that flips sign with distance, which is what would make a discriminator "
        "possible.",
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
                gid=anchor_ids(df["dt"], ANCHOR),
                day=anchor_ids(df["dt"], "session"),
                rth=in_rth(df["dt"]),
            )
            store[(sym, tf)] = (df, pre, session_groups(df))
            for key, r in cells_for(df, pre).items():
                obs[(sym, tf, *key)] = r
            log(f"  {sym} {tf}: {len(df):,} train bars")

    all_keys = sorted(obs)
    keys = [
        k
        for k in all_keys
        if obs[k]["nA"] >= a.min_n and obs[k]["nB"] >= a.min_n and np.isfinite(obs[k]["dA_B"])
    ]
    kidx = {k: i for i, k in enumerate(keys)}
    log(f"\n  {len(keys)} of {len(all_keys)} cells have at least {a.min_n} events in both arms\n")
    if not keys:
        log("  nothing testable. STOP.")
        return 1

    o_ab = np.array([obs[k]["dA_B"] for k in keys])
    o_a = np.array([obs[k]["mA"] for k in keys])
    o_ac = np.array([obs[k]["dA_C"] for k in keys])

    log(f"  running {a.perms} grouped bar permutations ...")
    n_ab = np.full((a.perms, len(keys)), np.nan)
    n_a = np.full((a.perms, len(keys)), np.nan)
    n_ac = np.full((a.perms, len(keys)), np.nan)
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
                    n_ab[p, kidx[full]] = r["dA_B"]
                    n_a[p, kidx[full]] = r["mA"]
                    n_ac[p, kidx[full]] = r["dA_C"]
        if p == 0 or (p + 1) % 25 == 0:
            el = time.time() - t0
            log(
                f"    {p + 1:>5}/{a.perms}  {el:.0f}s elapsed, ~{el / (p + 1) * (a.perms - p - 1):.0f}s left"
            )

    st = family_stats(o_ab, n_ab, a.perms)
    sa = family_stats(o_a, n_a, a.perms)
    sc = family_stats(o_ac, n_ac, a.perms)

    rows = []
    for i, k in enumerate(keys):
        r = obs[k]
        pc = (1 + int((np.abs(st["t_null"][:, i]) >= abs(st["t"][i])).sum())) / (1 + a.perms)
        rows.append(
            dict(
                symbol=k[0],
                tf=k[1],
                pair=k[2],
                band=k[3],
                sign=k[4],
                filter=k[5],
                horizon=k[6],
                nA=r["nA"],
                nB=r["nB"],
                nC=r["nC"],
                mA=r["mA"],
                mB=r["mB"],
                dA_B=r["dA_B"],
                dA_C=r["dA_C"],
                null_mean=st["null_mean"][i],
                t=st["t"][i],
                p_cell=pc,
                t_mA=sa["t"][i],
            )
        )
    d = pd.DataFrame(rows)
    d.to_csv(out / "cells.csv", index=False)
    np.save(out / "null_dAB.npy", n_ab)

    log("\n" + "=" * 104)
    log("  RESULT")
    log("=" * 104)
    log("  NOTE ON A - B: arm B is touches with NO cross, scored on a reversion-signed return.")
    log("  The absence of a cross up after a low is already correlated with price not going up,")
    log("  so A - B is inflated by construction. A - C is the clean statistic: it asks whether")
    log("  the cross AT A BAND beats the same cross anywhere, which is what the band must add.")
    log()
    for label, s in (
        ("A - B  (inflated by construction, see note)", st),
        ("A - C  (THE CLEAN TEST: does the band add anything)", sc),
        ("A alone (mean return)", sa),
    ):
        log(f"  {label}")
        log(
            f"    family-max |t| = {s['fam']:.2f}   null 95th {s['fam_null_p95']:.2f}"
            f"   p = {s['p_fam']:.4f}"
        )
        log(
            f"    count {s['count']} of {len(keys)} outside their own 5-95 band"
            f"   null median {s['count_null_med']:.0f}   p = {s['p_count']:.4f}"
        )
    log(f"  expected by chance at 10% two-sided: {0.10 * len(keys):.0f}")

    log("\n  A - B by MA pair, averaged over every cell (ATR units):")
    log(
        f"  {'pair':<13}{'cells':>7}{'mean A':>9}{'mean B':>9}{'mean A-B':>10}"
        f"{'mean A-C':>10}{'mean |t|':>10}{'p<0.05':>8}"
    )
    for pair in PAIRS:
        s2 = d[d["pair"] == pair]
        if s2.empty:
            continue
        log(
            f"  {pair:<13}{len(s2):>7}{s2['mA'].mean():>+9.4f}{s2['mB'].mean():>+9.4f}"
            f"{s2['dA_B'].mean():>+10.4f}{s2['dA_C'].mean():>+10.4f}"
            f"{s2['t'].abs().mean():>10.2f}{int((s2['p_cell'] < 0.05).sum()):>8}"
        )

    log("\n  A - B by horizon:")
    log(f"  {'bars':>6}{'minutes(5m)':>13}{'cells':>7}{'mean A':>9}{'mean A-B':>10}{'p<0.05':>8}")
    for hz in HORIZONS:
        s2 = d[d["horizon"] == hz]
        log(
            f"  {hz:>6}{hz * 5:>13}{len(s2):>7}{s2['mA'].mean():>+9.4f}"
            f"{s2['dA_B'].mean():>+10.4f}{int((s2['p_cell'] < 0.05).sum()):>8}"
        )

    log("\n  the ten strongest cells by |t| on A-B:")
    log(
        f"  {'sym':<4}{'tf':<4}{'pair':<12}{'band':>5}{'filt':>10}{'hz':>4}{'nA':>5}"
        f"{'nB':>5}{'A':>8}{'B':>8}{'A-B':>8}{'A-C':>8}{'t':>7}{'p':>8}"
    )
    for _, r in d.reindex(d["t"].abs().sort_values(ascending=False).index).head(10).iterrows():
        log(
            f"  {r['symbol']:<4}{r['tf']:<4}{r['pair']:<12}"
            f"{int(r['sign']) * int(r['band']):>+5}{r['filter']:>10}{int(r['horizon']):>4}"
            f"{int(r['nA']):>5}{int(r['nB']):>5}{r['mA']:>+8.3f}{r['mB']:>+8.3f}"
            f"{r['dA_B']:>+8.3f}{r['dA_C']:>+8.3f}{r['t']:>+7.2f}{r['p_cell']:>8.4f}"
        )

    # The verdict rides on A - C, not A - B: a band that adds nothing to an already rejected
    # unconditional cross is not a finding however large A - B looks.
    verdict = "PASS" if min(sc["p_fam"], sc["p_count"]) < 0.05 else "REJECT"
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                hypothesis=hh,
                permutations=a.perms,
                cells=len(keys),
                dAB_family_max_t=st["fam"],
                dAB_p_family=st["p_fam"],
                dAB_count=st["count"],
                dAB_p_count=st["p_count"],
                mA_p_family=sa["p_fam"],
                mA_p_count=sa["p_count"],
                dAC_family_max_t=sc["fam"],
                dAC_p_family=sc["p_fam"],
                dAC_count=sc["count"],
                dAC_p_count=sc["p_count"],
                mean_dA_B=float(np.nanmean(o_ab)),
                mean_dA_C=float(d["dA_C"].mean()),
                mean_A=float(np.nanmean(o_a)),
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
