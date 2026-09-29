#!/usr/bin/env python3
"""H19. Is there ANY directional predictability at 5-20 bars, before we test a single idea?

    python scripts/direction_screen.py --perms 400

THE QUESTION

Not "will price go up 40 points", but the weaker and more useful one: does knowing some state at
bar t move P(close[t+N] > close[t]) away from its base rate? A conditioner that shifts the
direction probability by enough is all the target specification needs.

WHY THIS RUNS BEFORE ANY INDICATOR IS TESTED

Sixteen hypotheses here have tested specific signals. This tests the CHANNEL: if no simple state
variable moves the directional probability at all, no refinement of those signals will either,
and that is worth one run rather than twenty.

THE BENCHMARKS, FIXED BEFORE THE RUN

  base rate     whatever P(up) is unconditionally, reported rather than assumed
  the paper     Takeuchi & Lee (2013) reach 53.36% directional accuracy on monthly US stocks
                with a stacked-RBM autoencoder and 848,000 training examples: +3.36 points
  our target    the prop specification needs about +10 points of win rate

So three outcomes, named in advance so none of them can be talked into being a success later:
nothing beats the null and direction is not predictable at these horizons; something beats it by
1-4 points, which is real and still a third of what is needed; something beats it by 10 points or
more, which would be extraordinary and should be treated as a bug until proven otherwise.

LABELS

Sign of the net move over N bars, close to close -- the same binary object the paper uses.
NON-OVERLAPPING windows (stride N): overlapping samples would inflate the effective sample size
and every statistic built on it. There are millions of bars, so the overlap is affordable to
discard.

WHAT THIS DOES NOT MEASURE

Net-sign accuracy is not the win rate the specification needs. That figure is P(target before
stop) on a 0.8:1 barrier geometry, a race between two levels rather than the sign of a close.
This is a SCREEN: flat here means the barrier version is flat too; positive here still has to be
re-measured as a barrier race before it means anything tradeable.

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
from quant.strategies.vwap_bands import anchor_ids, band_index, vwap_bands

SYMBOLS = ("MES", "MNQ")
TFS = ("1m", "5m", "15m")
HORIZONS = (5, 10, 15, 20)
EMA_LEN, SLOPE_LOOKBACK, Z_WIN = 20, 5, 100
CONDITIONERS = (
    "close_vs_ema",
    "ema_slope",
    "vwap_zone",
    "past_return",
    "past_return_z",
    "two_scale",
)


def log(m: str = "") -> None:
    print(m, flush=True)


def calls(df: pd.DataFrame, gid: np.ndarray, N: int) -> dict[str, np.ndarray]:
    """Directional call per bar for each conditioner: +1 up, -1 down, 0 no opinion.

    Every value is computed from closes up to and including bar t, and is compared against the
    move from bar t to bar t+N, so nothing here reads the future.
    """
    c = df["close"].to_numpy(float)
    e = S.ema(c, EMA_LEN)
    out: dict[str, np.ndarray] = {}

    out["close_vs_ema"] = np.sign(c - e)

    sl = np.r_[np.full(SLOPE_LOOKBACK, np.nan), e[SLOPE_LOOKBACK:] - e[:-SLOPE_LOOKBACK]]
    out["ema_slope"] = np.nan_to_num(np.sign(sl))

    mu, sg = vwap_bands(df, gid)
    z = band_index(mu, sg, c)
    out["vwap_zone"] = np.where(np.abs(z) >= 1, np.sign(z), 0.0)

    pr = np.r_[np.full(N, np.nan), c[N:] - c[:-N]]
    out["past_return"] = np.nan_to_num(np.sign(pr))

    s = pd.Series(pr)
    zz = ((s - s.rolling(Z_WIN).mean()) / s.rolling(Z_WIN).std()).to_numpy()
    out["past_return_z"] = np.where(np.abs(zz) >= 1.0, np.sign(zz), 0.0)

    # the paper's structure: long-horizon trend, short-horizon reversion against it
    lr = np.r_[np.full(4 * N, np.nan), c[4 * N :] - c[: -4 * N]]
    short = np.nan_to_num(np.sign(pr))
    long_ = np.nan_to_num(np.sign(lr))
    out["two_scale"] = np.where((long_ != 0) & (short == -long_), long_, 0.0)

    for k in out:
        out[k] = np.nan_to_num(out[k])
    return out


def cells_for(df: pd.DataFrame, gid: np.ndarray) -> dict:
    c = df["close"].to_numpy(float)
    n = len(c)
    res: dict[tuple, dict] = {}
    for N in HORIZONS:
        fwd_sign = np.zeros(n)
        fwd_sign[: n - N] = np.sign(c[N:] - c[: n - N])
        idx = np.arange(0, n - N, N)  # NON-OVERLAPPING
        cl = calls(df, gid, N)
        y = fwd_sign[idx]
        base = float((y > 0).mean())
        for name, call in cl.items():
            s = call[idx]
            m = (s != 0) & (y != 0)
            if m.sum() < 200:
                res[(name, N)] = dict(n=int(m.sum()), acc=np.nan, base=base, cov=0.0)
                continue
            acc = float((np.sign(s[m]) == np.sign(y[m])).mean())
            res[(name, N)] = dict(n=int(m.sum()), acc=acc, base=base, cov=float(m.mean()))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/direction_screen")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 104)
    log("  H19  DIRECTIONAL SCREEN: does any simple state move P(up) at 5-20 bars?")
    log("=" * 104)
    log(
        f"  family: {len(CONDITIONERS)} conditioners x {len(HORIZONS)} horizons x "
        f"{len(TFS)} timeframes x {len(SYMBOLS)} symbols = "
        f"{len(CONDITIONERS) * len(HORIZONS) * len(TFS) * len(SYMBOLS)} cells"
    )
    log(f"  permutations: {a.perms}   NON-OVERLAPPING windows   TRAIN only\n")
    log("  benchmarks fixed before the run:")
    log("    Takeuchi & Lee 2013 reach 53.36% on monthly stocks, 848k examples  -> +3.36 points")
    log("    the prop specification needs                                       -> +10   points\n")

    h = Hypothesis(
        name="direction_screen",
        instrument="MES and MNQ, 1m / 5m / 15m",
        session="all bars in the train segment; no session filter, so the base rate is the "
        "unconditional one",
        setup="six declared conditioners give a directional call at bar t: close above/below "
        "EMA(20); EMA(20) slope sign over 5 bars; signed anchored-VWAP band zone; sign of the "
        "past N-bar return; that return z-scored over 100 bars and thresholded at +/-1; and a "
        "two-scale rule taking the 4N-bar trend only when the N-bar return opposes it.",
        direction="both",
        exit_rule="none -- the label is the SIGN of the close-to-close move over N bars, with "
        "N in 5, 10, 15, 20 and non-overlapping windows at stride N",
        execution="every conditioner is built from closes up to and including bar t and scored "
        "against the move from t to t+N, so no value reads its own outcome",
        costs="not entered: the statistic is a directional accuracy, not a P&L. A positive "
        "result must be re-measured as a barrier race before any cost claim is made.",
        invalidation="abandon the channel if the family-max permutation p is >= 0.05. A pass "
        "below +4 points of accuracy is recorded as real-but-insufficient rather than as an "
        "edge, since the target specification needs about +10.",
        metric="directional accuracy against a per-cell permutation null; family-max and count "
        "statistics over the whole family",
        rationale="Sixteen hypotheses here tested specific signals. This tests the CHANNEL: "
        "whether any simple state moves the direction probability at all at these horizons. If "
        "it does not, no refinement of those signals will either. Takeuchi and Lee reach only "
        "+3.36 points on monthly stocks with a deep autoencoder and 848,000 examples, which is "
        "the scale this should be read against.",
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
            gid = anchor_ids(df["dt"], "ny")
            store[(sym, tf)] = (df, gid, session_groups(df))
            for key, r in cells_for(df, gid).items():
                obs[(sym, tf, *key)] = r
            log(
                f"  {sym} {tf}: {len(df):,} train bars   base rate up "
                f"{100 * cells_for(df, gid)[(CONDITIONERS[0], HORIZONS[0])]['base']:.2f}%"
            )

    keys = [k for k in sorted(obs) if np.isfinite(obs[k]["acc"]) and obs[k]["n"] >= 200]
    kidx = {k: i for i, k in enumerate(keys)}
    o = np.array([obs[k]["acc"] for k in keys])
    log(f"\n  {len(keys)} testable cells, {sum(obs[k]['n'] for k in keys):,} scored windows\n")

    log(f"  running {a.perms} grouped bar permutations ...")
    null = np.full((a.perms, len(keys)), np.nan)
    t0 = time.time()
    for p in range(a.perms):
        for (sym, tf), (df, gid, g) in store.items():
            pf = get_permutation_fast(
                df, start_index=0, seed=a.seed + p * 1009 + hash(sym + tf) % 997, groups=g
            )
            pf["dt"] = df["dt"]
            for key, r in cells_for(pf, gid).items():
                full = (sym, tf, *key)
                if full in kidx:
                    null[p, kidx[full]] = r["acc"]
        if p == 0 or (p + 1) % 25 == 0:
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
                cond=k[2],
                horizon=k[3],
                n=r["n"],
                acc=r["acc"],
                base=r["base"],
                cov=r["cov"],
                null_mean=mn[i],
                edge_pp=100 * (r["acc"] - mn[i]),
                t=t_o[i],
                p_cell=pc,
            )
        )
    d = pd.DataFrame(rows)
    d.to_csv(out / "cells.csv", index=False)
    np.save(out / "null.npy", null)

    log("\n" + "=" * 104)
    log("  RESULT")
    log("=" * 104)
    log(
        f"  family-max |t| = {fam:.2f}   null 95th {np.percentile(fam_n, 95):.2f}   p = {p_fam:.4f}"
    )
    log(
        f"  count {cnt} of {len(keys)} outside their own 5-95 band"
        f"   null median {np.median(cnt_n):.0f}   p = {p_cnt:.4f}"
    )
    log(
        f"  mean accuracy {100 * d['acc'].mean():.2f}%   mean null {100 * d['null_mean'].mean():.2f}%"
        f"   mean edge {d['edge_pp'].mean():+.2f} points"
    )

    log("\n  BY CONDITIONER (edge in percentage points over its own null):")
    log(
        f"  {'conditioner':<16}{'cells':>7}{'mean acc':>10}{'mean null':>11}"
        f"{'mean edge':>11}{'best edge':>11}{'p<0.05':>8}"
    )
    for c in CONDITIONERS:
        s = d[d["cond"] == c]
        if s.empty:
            continue
        log(
            f"  {c:<16}{len(s):>7}{100 * s['acc'].mean():>9.2f}%{100 * s['null_mean'].mean():>10.2f}%"
            f"{s['edge_pp'].mean():>+11.2f}{s['edge_pp'].max():>+11.2f}"
            f"{int((s['p_cell'] < 0.05).sum()):>8}"
        )

    log("\n  BY HORIZON:")
    log(f"  {'bars':>6}{'cells':>7}{'mean acc':>10}{'mean edge':>11}{'p<0.05':>8}")
    for N in HORIZONS:
        s = d[d["horizon"] == N]
        log(
            f"  {N:>6}{len(s):>7}{100 * s['acc'].mean():>9.2f}%{s['edge_pp'].mean():>+11.2f}"
            f"{int((s['p_cell'] < 0.05).sum()):>8}"
        )

    log("\n  THE TEN STRONGEST CELLS:")
    log(
        f"  {'sym':<4}{'tf':<4}{'conditioner':<16}{'N':>4}{'n':>8}{'acc':>8}{'null':>8}"
        f"{'edge pp':>9}{'t':>7}{'p':>8}"
    )
    for _, r in d.reindex(d["t"].abs().sort_values(ascending=False).index).head(10).iterrows():
        log(
            f"  {r['symbol']:<4}{r['tf']:<4}{r['cond']:<16}{int(r['horizon']):>4}{int(r['n']):>8}"
            f"{100 * r['acc']:>7.2f}%{100 * r['null_mean']:>7.2f}%{r['edge_pp']:>+9.2f}"
            f"{r['t']:>+7.2f}{r['p_cell']:>8.4f}"
        )

    best = d["edge_pp"].abs().max()
    log(f"\n  best edge anywhere: {best:.2f} points")
    log("    against Takeuchi & Lee at +3.36 and the specification at +10.00")
    verdict = "PASS" if min(p_fam, p_cnt) < 0.05 else "REJECT"
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
                mean_edge_pp=float(d["edge_pp"].mean()),
                best_edge_pp=float(best),
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
