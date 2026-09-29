#!/usr/bin/env python3
"""H21 stage 1. The structured candidate array: which indicator FAMILIES carry direction?

    python scripts/indicator_array.py --perms 300

SCOPE, AND WHY IT IS THIS SIZE

The full menu -- every family, every variant, every signal definition, every timeframe and
horizon -- is roughly 8,600 cells. A family-max correction over 8,600 draws raises the bar by
close to a full t-point, and scripts/direction_ceiling.py has already shown that real effects
here are worth about 1 point of accuracy. Running the whole menu at once would therefore
guarantee a null result through the size of the search alone.

So it is staged, and the stage-2 rule is written down HERE, before stage 1 is run, so the scope
cannot be chosen to fit the answer:

    STAGE 2 RULE, PRE-COMMITTED
    Only indicator families with at least two cells at uncorrected p < 0.05 AND a mean absolute
    edge of at least 0.5 points in stage 1 proceed. Those families are then expanded across
    variants and the 1m and 15m timeframes. If no family qualifies, there is no stage 2 and the
    directional-indicator channel is closed.

STAGE 1 GRID

  14 indicators x 6 signal definitions x 3 horizons x 2 symbols = 504 cells, on 5m only.

  Eleven of the indicators are OPEN -- non-linear in returns, so outside the linear-filter
  ceiling. Three (ema, macd, roc) are NEGATIVE CONTROLS: they are known to sit inside that bound,
  so they have an expected score of roughly nothing, and a harness that flatters them is broken.
  They are reported separately and excluded from the stage-2 decision.

THE QUESTION ASKED OF EVERY CANDIDATE, IDENTICALLY

From the first bar on which the condition is observable, what is the sign of the move over the
next 5, 10 and 20 bars, in non-overlapping windows, against that same cell's permutation null?

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
from quant.strategies.indicator_menu import BOUNDED, atr_for, build, signals

SYMBOLS = ("MES", "MNQ")
TF = "5m"
HORIZONS = (5, 10, 20)
DEFS = ("position", "fresh", "buffered", "slope", "sustained", "aligned")
MIN_N = 200


def log(m: str = "") -> None:
    print(m, flush=True)


def cells_for(df: pd.DataFrame) -> dict:
    c = df["close"].to_numpy(float)
    n = len(c)
    atr = atr_for(df)
    menu = build(df)
    sig = {name: signals(v, r, atr) for name, (v, r) in menu.items()}
    res: dict[tuple, dict] = {}
    for N in HORIZONS:
        fwd = np.zeros(n)
        fwd[: n - N] = np.sign(c[N:] - c[: n - N])
        idx = np.arange(200, n - N, N)  # non-overlapping, indicators warmed up
        y = fwd[idx]
        for name, defs in sig.items():
            for dname, s in defs.items():
                v = s[idx]
                m = (v != 0) & (y != 0)
                k = int(m.sum())
                res[(name, dname, N)] = dict(
                    n=k,
                    acc=float((np.sign(v[m]) == np.sign(y[m])).mean()) if k >= MIN_N else np.nan,
                    cov=float(m.mean()),
                )
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/indicator_array")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 108)
    log("  H21 stage 1  INDICATOR ARRAY: which families carry direction, and under which")
    log("  signal definition?")
    log("=" * 108)
    log(f"  {TF} bars, horizons {HORIZONS}, {len(DEFS)} signal definitions, both symbols")
    log("  11 open indicators (non-linear in returns) + 3 negative controls inside the")
    log("  linear-filter ceiling, reported separately\n")
    log("  STAGE 2 RULE, pre-committed before this run:")
    log("    a family proceeds only with >= 2 cells at p < 0.05 AND mean |edge| >= 0.5 points.")
    log("    If none qualifies there is no stage 2 and the channel is closed.\n")

    h = Hypothesis(
        name="indicator_array_stage1",
        instrument=f"MES and MNQ, {TF}",
        session="all train bars, no session filter",
        setup="14 indicators, each reduced to a value and a reference so that value>reference is "
        "bullish, crossed with six signal definitions: position, fresh cross, cross buffered by "
        "0.25 ATR, position with reference slope over 5 bars, position sustained 3 bars, and "
        "fast/slow alignment. Eleven indicators are non-linear in returns (KAMA, VIDYA, RSI, CMO, "
        "+DI/-DI, TSI, Donchian mid, Bollinger mid and %B, Keltner mid and position); three "
        "(EMA, MACD, ROC) are negative controls known to lie inside the linear-filter ceiling.",
        direction="both",
        exit_rule="none -- the label is the sign of the close-to-close move over N bars, "
        "N in 5, 10, 20, non-overlapping at stride N",
        execution="every indicator value is computed from bars up to and including bar t and "
        "scored against the move from t to t+N, so nothing reads its own outcome",
        costs="not entered: the statistic is directional accuracy. Any survivor is re-measured "
        "as a barrier race with costs before a tradeability claim.",
        invalidation="a family proceeds to stage 2 only with at least two cells at uncorrected "
        "p<0.05 and a mean absolute edge of at least 0.5 points. No qualifier closes the channel.",
        metric="directional accuracy against a per-cell permutation null; family-max and count "
        "statistics; negative controls excluded from the stage-2 decision",
        rationale="scripts/direction_ceiling.py bounded every linear filter of past returns at "
        "about +1.5 points in sample, which settles SMA, EMA, WMA, HMA, DEMA, TEMA, ROC, MACD, "
        "PPO, TRIX and regression slope together. What remains untested is the non-linear "
        "families, and they share one property: each normalises by volatility or range rather "
        "than filtering price. The open question is therefore whether volatility normalisation "
        "carries information that price filtering does not. The signal definition is treated as "
        "its own dimension because a line that hugs price makes crossings frequent, not useful.",
        author="research@example.invalid",
    )
    hh = h.register("out/hypotheses.jsonl")
    log(f"[pre-registered] {hh}\n")

    store, obs = {}, {}
    for sym in SYMBOLS:
        df = S.load_data(DATA[sym].format(tf=TF), "America/Chicago")
        d0 = df["dt"].dt.tz_localize(None).to_numpy()
        df = df.iloc[: make_split(sym, TF, d0).train.stop].reset_index(drop=True)
        store[sym] = (df, session_groups(df))
        for key, r in cells_for(df).items():
            obs[(sym, *key)] = r
        log(f"  {sym} {TF}: {len(df):,} train bars")

    keys = [k for k in sorted(obs) if np.isfinite(obs[k]["acc"])]
    kidx = {k: i for i, k in enumerate(keys)}
    o = np.array([obs[k]["acc"] for k in keys])
    log(f"\n  {len(keys)} of {len(obs)} cells have at least {MIN_N} scored windows\n")

    log(f"  running {a.perms} grouped bar permutations ...")
    null = np.full((a.perms, len(keys)), np.nan)
    t0 = time.time()
    for p in range(a.perms):
        for sym, (df, g) in store.items():
            pf = get_permutation_fast(
                df, start_index=0, seed=a.seed + p * 1009 + hash(sym) % 997, groups=g
            )
            for key, r in cells_for(pf).items():
                full = (sym, *key)
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
                ind=k[1],
                sig=k[2],
                horizon=k[3],
                n=r["n"],
                cov=r["cov"],
                acc=r["acc"],
                null_mean=mn[i],
                edge_pp=100 * (r["acc"] - mn[i]),
                t=t_o[i],
                p_cell=pc,
                control=BOUNDED.get(k[1], False),
            )
        )
    d = pd.DataFrame(rows)
    d.to_csv(out / "cells.csv", index=False)
    np.save(out / "null.npy", null)

    log("\n" + "=" * 108)
    log("  RESULT")
    log("=" * 108)
    log(
        f"  family-max |t| = {fam:.2f}   null 95th {np.percentile(fam_n, 95):.2f}   p = {p_fam:.4f}"
    )
    log(f"  count {cnt} of {len(keys)}   null median {np.median(cnt_n):.0f}   p = {p_cnt:.4f}")

    log("\n  NEGATIVE CONTROLS (inside the linear bound -- these should be near nothing):")
    log(f"  {'indicator':<16}{'cells':>7}{'mean edge':>11}{'best |edge|':>13}{'p<0.05':>8}")
    for name in sorted(d[d["control"]]["ind"].unique()):
        s = d[d["ind"] == name]
        log(
            f"  {name:<16}{len(s):>7}{s['edge_pp'].mean():>+11.2f}"
            f"{s['edge_pp'].abs().max():>13.2f}{int((s['p_cell'] < 0.05).sum()):>8}"
        )

    op = d[~d["control"]]
    log("\n  OPEN FAMILIES, by indicator (edge is signed: + means the rule as stated,")
    log("  - means its inverse is the informative side):")
    log(
        f"  {'indicator':<16}{'cells':>7}{'mean acc':>10}{'mean edge':>11}{'best |edge|':>13}"
        f"{'p<0.05':>8}{'stage 2':>9}"
    )
    qual = []
    for name in sorted(op["ind"].unique()):
        s = op[op["ind"] == name]
        nsig = int((s["p_cell"] < 0.05).sum())
        mabs = s["edge_pp"].abs().mean()
        ok = (nsig >= 2) and (mabs >= 0.5)
        if ok:
            qual.append(name)
        log(
            f"  {name:<16}{len(s):>7}{100 * s['acc'].mean():>9.2f}%{s['edge_pp'].mean():>+11.2f}"
            f"{s['edge_pp'].abs().max():>13.2f}{nsig:>8}{'YES' if ok else '-':>9}"
        )

    log("\n  BY SIGNAL DEFINITION (open families only):")
    log(
        f"  {'definition':<12}{'cells':>7}{'coverage':>10}{'mean acc':>10}{'mean edge':>11}"
        f"{'best |edge|':>13}{'p<0.05':>8}"
    )
    for dn in DEFS:
        s = op[op["sig"] == dn]
        if s.empty:
            continue
        log(
            f"  {dn:<12}{len(s):>7}{100 * s['cov'].mean():>9.1f}%{100 * s['acc'].mean():>9.2f}%"
            f"{s['edge_pp'].mean():>+11.2f}{s['edge_pp'].abs().max():>13.2f}"
            f"{int((s['p_cell'] < 0.05).sum()):>8}"
        )

    log("\n  BY HORIZON (open families only):")
    log(f"  {'bars':>6}{'cells':>7}{'mean acc':>10}{'mean edge':>11}{'p<0.05':>8}")
    for N in HORIZONS:
        s = op[op["horizon"] == N]
        log(
            f"  {N:>6}{len(s):>7}{100 * s['acc'].mean():>9.2f}%{s['edge_pp'].mean():>+11.2f}"
            f"{int((s['p_cell'] < 0.05).sum()):>8}"
        )

    log("\n  THE TWELVE STRONGEST CELLS:")
    log(
        f"  {'sym':<4}{'indicator':<16}{'signal':<11}{'N':>4}{'n':>8}{'cov':>7}{'acc':>8}"
        f"{'null':>8}{'edge':>8}{'t':>7}{'p':>8}"
    )
    for _, r in d.reindex(d["t"].abs().sort_values(ascending=False).index).head(12).iterrows():
        tag = "*" if r["control"] else " "
        log(
            f"  {r['symbol']:<4}{r['ind'] + tag:<16}{r['sig']:<11}{int(r['horizon']):>4}"
            f"{int(r['n']):>8}{100 * r['cov']:>6.1f}%{100 * r['acc']:>7.2f}%"
            f"{100 * r['null_mean']:>7.2f}%{r['edge_pp']:>+8.2f}{r['t']:>+7.2f}{r['p_cell']:>8.4f}"
        )
    log("  (* = negative control)")

    log(f"\n  STAGE 2 QUALIFIERS: {', '.join(qual) if qual else 'NONE -- the channel is closed'}")
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
                best_abs_edge_pp=float(op["edge_pp"].abs().max()),
                stage2_qualifiers=qual,
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
