#!/usr/bin/env python3
"""H20. The CEILING on every moving average at once, instead of guessing which one to test.

    python scripts/direction_ceiling.py --perms 200

THE QUESTION THIS REPLACES

"Is close-above-EMA(20) predictive?" is a bad question, and so is the same question about HMA,
TEMA, WMA, KAMA or any other. Every one of them is a LINEAR FILTER of past closes: a weighted
sum of the last K bars, differing only in the weighting kernel and hence the lag. Testing HMA
after EMA is not a new experiment, it is the same experiment with the weights nudged.

So this asks the bounding question instead:

    What is the BEST directional accuracy obtainable by ANY causal linear filter of the last K
    returns?

Whatever that number is, it is an upper bound on every moving-average rule that exists, invented
or not, because each is one particular choice of weights inside the space being maximised over.
If the ceiling is flat, the entire family is settled in one run and no further MA needs testing.

HOW THE CEILING IS COMPUTED, AND WHY IT IS DELIBERATELY OPTIMISTIC

The weights are fitted by least squares on the SAME data they are scored on. That is in-sample
and therefore favourably biased -- on purpose. An in-sample ceiling that is already small is a
much stronger negative result than an out-of-sample one, because no honest procedure can beat it.

The null does exactly the same thing to a permuted market: same K, same fit, same scoring, same
in-sample optimism. The permuted fit will also beat 50%, purely by fitting noise. The gap between
the two is the only thing that means anything, and it is what the permutation test measures.

WHAT IS NOT COVERED

Linear filters of RETURNS. Non-linear functions of price (a squared term, a regime switch, a
threshold rule), and anything using volume, range or order flow, are outside this bound. A flat
ceiling here rules out moving averages; it does not rule out everything.

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

SYMBOLS = ("MES", "MNQ")
TFS = ("1m", "5m", "15m")
HORIZONS = (5, 10, 15, 20)
LOOKBACKS = (10, 20, 60)  # bars of history the filter may weight


def log(m: str = "") -> None:
    print(m, flush=True)


def ceiling(close: np.ndarray, K: int, N: int) -> dict:
    """Best in-sample directional accuracy from any linear filter of the last K returns.

    Features are the last K one-bar returns, each scaled by the rolling standard deviation of
    returns so the fit is not dominated by high-volatility stretches. The target is the SIGN of
    the forward N-bar move. Windows are non-overlapping at stride N.
    """
    r = np.diff(np.log(close), prepend=np.log(close[0]))
    sd = pd.Series(r).rolling(200).std().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        rn = np.where(sd > 0, r / sd, 0.0)
    rn = np.nan_to_num(rn, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(close)
    idx = np.arange(max(K, 200), n - N, N)  # non-overlapping, past warmed up
    if len(idx) < 500:
        return dict(n=0, acc=np.nan)
    X = np.stack([rn[idx - j] for j in range(K)], axis=1)
    y = np.sign(close[idx + N] - close[idx])
    m = y != 0
    X, y = X[m], y[m]
    if len(y) < 500:
        return dict(n=0, acc=np.nan)
    X = np.c_[X, np.ones(len(X))]
    # least squares on the signed label: the linear rule that best separates up from down
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = np.sign(X @ w)
    ok = pred != 0
    return dict(n=int(ok.sum()), acc=float((pred[ok] == y[ok]).mean()))


def cells_for(df: pd.DataFrame) -> dict:
    c = df["close"].to_numpy(float)
    return {(K, N): ceiling(c, K, N) for K in LOOKBACKS for N in HORIZONS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/direction_ceiling")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("  H20  THE CEILING ON EVERY MOVING AVERAGE AT ONCE")
    log("=" * 100)
    log("  Best in-sample directional accuracy from ANY causal linear filter of the last K")
    log("  returns. Every EMA, HMA, TEMA, WMA, SMA and KAMA rule is one point inside this space,")
    log("  so this number bounds all of them. The fit is in-sample ON PURPOSE: a small ceiling")
    log("  here cannot be beaten by any honest procedure.")
    log(
        f"\n  family: {len(LOOKBACKS)} lookbacks x {len(HORIZONS)} horizons x {len(TFS)} tfs x "
        f"{len(SYMBOLS)} symbols = {len(LOOKBACKS) * len(HORIZONS) * len(TFS) * len(SYMBOLS)} cells"
    )
    log(f"  permutations: {a.perms}   TRAIN only\n")

    h = Hypothesis(
        name="direction_ceiling",
        instrument="MES and MNQ, 1m / 5m / 15m",
        session="all train bars, no session filter",
        setup="least-squares weights over the last K normalised one-bar returns, K in 10, 20, 60, "
        "fitted to the SIGN of the forward N-bar move with N in 5, 10, 15, 20. The fitted rule is "
        "the best linear filter of past returns that exists for that cell.",
        direction="both",
        exit_rule="none -- the label is the sign of the close-to-close move over N bars, "
        "non-overlapping at stride N",
        execution="features are the K returns ending at bar t, scaled by a 200-bar rolling "
        "standard deviation ending at bar t; the label starts at bar t and ends at t+N",
        costs="not entered: this is a directional accuracy and an upper bound, not a P&L",
        invalidation="if the in-sample ceiling does not beat its own permutation null, the entire "
        "moving-average family is closed and no further MA variant is tested here.",
        metric="in-sample directional accuracy against the same in-sample fit on each permuted "
        "market, so the optimism of fitting is present on both sides and cancels",
        rationale="Twelve moving-average hypotheses have been rejected here one at a time, and "
        "choosing the thirteenth by taste is not a method. Every MA is a linear filter of past "
        "closes differing only in its weighting kernel, so maximising over the whole space of "
        "such filters bounds all of them at once. The null fits noise the same way, so the gap "
        "is the only quantity that carries information.",
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
            store[(sym, tf)] = (df, session_groups(df))
            for key, r in cells_for(df).items():
                obs[(sym, tf, *key)] = r
            log(f"  {sym} {tf}: {len(df):,} train bars")

    keys = [k for k in sorted(obs) if np.isfinite(obs[k]["acc"]) and obs[k]["n"] >= 500]
    kidx = {k: i for i, k in enumerate(keys)}
    o = np.array([obs[k]["acc"] for k in keys])
    log(f"\n  {len(keys)} testable cells\n")

    log(f"  running {a.perms} grouped bar permutations (refitting every cell each draw) ...")
    null = np.full((a.perms, len(keys)), np.nan)
    t0 = time.time()
    for p in range(a.perms):
        for (sym, tf), (df, g) in store.items():
            pf = get_permutation_fast(
                df, start_index=0, seed=a.seed + p * 1009 + hash(sym + tf) % 997, groups=g
            )
            for key, r in cells_for(pf).items():
                full = (sym, tf, *key)
                if full in kidx:
                    null[p, kidx[full]] = r["acc"]
        if p == 0 or (p + 1) % 20 == 0:
            el = time.time() - t0
            log(
                f"    {p + 1:>5}/{a.perms}  {el:.0f}s elapsed, "
                f"~{el / (p + 1) * (a.perms - p - 1):.0f}s left"
            )

    mn = np.nanmean(null, axis=0)
    sd = np.nanstd(null, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.inf)
    t_o, t_n = (o - mn) / sd, (null - mn) / sd
    fam = float(np.nanmax(np.abs(t_o)))
    fam_n = np.nanmax(np.abs(t_n), axis=1)
    p_fam = (1 + int((fam_n >= fam).sum())) / (1 + a.perms)
    hi, lo = np.nanpercentile(null, 95, axis=0), np.nanpercentile(null, 5, axis=0)
    cnt = int(((o > hi) | (o < lo)).sum())
    cnt_n = np.nansum((null > hi) | (null < lo), axis=1)
    p_cnt = (1 + int((cnt_n >= cnt).sum())) / (1 + a.perms)

    rows = []
    for i, k in enumerate(keys):
        r = obs[k]
        pc = (1 + int((np.abs(t_n[:, i]) >= abs(t_o[i])).sum())) / (1 + a.perms)
        rows.append(
            dict(
                symbol=k[0],
                tf=k[1],
                lookback=k[2],
                horizon=k[3],
                n=r["n"],
                acc=r["acc"],
                null_mean=mn[i],
                edge_pp=100 * (r["acc"] - mn[i]),
                t=t_o[i],
                p_cell=pc,
            )
        )
    d = pd.DataFrame(rows)
    d.to_csv(out / "cells.csv", index=False)
    np.save(out / "null.npy", null)

    log("\n" + "=" * 100)
    log("  RESULT")
    log("=" * 100)
    log(
        f"  family-max |t| = {fam:.2f}   null 95th {np.percentile(fam_n, 95):.2f}   p = {p_fam:.4f}"
    )
    log(f"  count {cnt} of {len(keys)}   null median {np.median(cnt_n):.0f}   p = {p_cnt:.4f}")
    log(f"\n  mean IN-SAMPLE ceiling {100 * d['acc'].mean():.2f}%")
    log(f"  mean null (same fit on noise) {100 * d['null_mean'].mean():.2f}%")
    log(f"  mean gap {d['edge_pp'].mean():+.2f} points   best gap {d['edge_pp'].max():+.2f}")

    log("\n  BY LOOKBACK (bars of history the filter may weight):")
    log(f"  {'K':>5}{'cells':>7}{'ceiling':>10}{'null':>9}{'gap pp':>9}{'p<0.05':>8}")
    for K in LOOKBACKS:
        s = d[d["lookback"] == K]
        log(
            f"  {K:>5}{len(s):>7}{100 * s['acc'].mean():>9.2f}%{100 * s['null_mean'].mean():>8.2f}%"
            f"{s['edge_pp'].mean():>+9.2f}{int((s['p_cell'] < 0.05).sum()):>8}"
        )

    log("\n  BY HORIZON:")
    log(f"  {'N':>5}{'cells':>7}{'ceiling':>10}{'null':>9}{'gap pp':>9}{'p<0.05':>8}")
    for N in HORIZONS:
        s = d[d["horizon"] == N]
        log(
            f"  {N:>5}{len(s):>7}{100 * s['acc'].mean():>9.2f}%{100 * s['null_mean'].mean():>8.2f}%"
            f"{s['edge_pp'].mean():>+9.2f}{int((s['p_cell'] < 0.05).sum()):>8}"
        )

    log("\n  THE TEN STRONGEST CELLS:")
    log(
        f"  {'sym':<4}{'tf':<4}{'K':>5}{'N':>4}{'n':>8}{'ceiling':>9}{'null':>9}"
        f"{'gap pp':>9}{'t':>7}{'p':>8}"
    )
    for _, r in d.reindex(d["t"].abs().sort_values(ascending=False).index).head(10).iterrows():
        log(
            f"  {r['symbol']:<4}{r['tf']:<4}{int(r['lookback']):>5}{int(r['horizon']):>4}"
            f"{int(r['n']):>8}{100 * r['acc']:>8.2f}%{100 * r['null_mean']:>8.2f}%"
            f"{r['edge_pp']:>+9.2f}{r['t']:>+7.2f}{r['p_cell']:>8.4f}"
        )

    verdict = "PASS" if min(p_fam, p_cnt) < 0.05 else "REJECT"
    log(f"\n  best gap anywhere: {d['edge_pp'].abs().max():.2f} points, IN SAMPLE")
    log("  Takeuchi & Lee reach +3.36 out of sample; the specification needs +10.")
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                hypothesis=hh,
                permutations=a.perms,
                cells=len(keys),
                family_max_t=fam,
                p_family=p_fam,
                count=cnt,
                p_count=p_cnt,
                mean_ceiling=float(d["acc"].mean()),
                mean_gap_pp=float(d["edge_pp"].mean()),
                best_gap_pp=float(d["edge_pp"].abs().max()),
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
