#!/usr/bin/env python3
"""Price the two Q5 findings: the London reversion, and the acceptance discriminator.

    python scripts/vwap_session_economics.py

Runs no permutation -- the p-values are fixed in out/vwap_band_by_session. This recomputes the
trade geometry (distance to the mean, distance to the next band out, sigma in points) for the
cells that passed, so the reversion rate can be set against the rate the trade actually needs.

The Q1 baseline established that on a driftless walk the reversion rate EQUALS the zero-cost
breakeven, so the whole question is whether a cell beats its null by more than cost demands. At
the 1 sigma band cost demands roughly 9.6 percentage points of win rate on these instruments; at
4 sigma roughly 3.4. Those are the bars to clear.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant import strategy as S
from quant.costs import POINT_VALUE, CostModel
from quant.data_splits import DATA, make_split
from quant.strategies.vwap_bands import anchor_ids, race, vwap_bands

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.vwap_band_by_session import (
    HORIZON,
    bucket_mask,
    first_per_anchor,
)

CELLS = pathlib.Path("out/vwap_band_by_session/cells.csv")


def geometry(sym: str, tf: str, bucket: str, k: int, sign: int, arm: str) -> dict:
    df = S.load_data(DATA[sym].format(tf=tf), "America/Chicago")
    d0 = df["dt"].dt.tz_localize(None).to_numpy()
    df = df.iloc[: make_split(sym, tf, d0).train.stop].reset_index(drop=True)
    gid = anchor_ids(df["dt"], "ny")
    mins = df["dt"].dt.hour.to_numpy() * 60 + df["dt"].dt.minute.to_numpy()
    mu, sg = vwap_bands(df, gid)
    hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
    level = mu + sign * k * sg
    reach = ((hi >= level) if sign > 0 else (lo <= level)) & np.isfinite(level)
    if arm == "accept":
        reach &= (cl > level) if sign > 0 else (cl < level)
    elif arm == "wick":
        reach &= ~((cl > level) if sign > 0 else (cl < level))
    ev = first_per_anchor(np.flatnonzero(reach & bucket_mask(mins, bucket)), gid)
    hor = HORIZON[tf]
    r = race(df, mu, sg, ev, k, sign, hor)
    ev = ev[(ev + hor) < len(df)]
    if len(ev) == 0:
        return dict(nA=0)
    s = sg[ev]
    dm = np.abs(mu[ev] - cl[ev]) / s
    de = np.abs(cl[ev] - (mu[ev] + sign * (k + 1) * s)) / s
    cost = CostModel.for_prop(sym, "average", contracts=1).round_turn_points
    win = float((dm * s).mean())
    risk = float((de * s).mean())
    return dict(
        nA=r["nA"],
        pA=r["pA"],
        win=win,
        risk=risk,
        cost=cost,
        be=(risk + cost) / (win + risk),
        be0=risk / (win + risk),
        ev_rev=r["pA"] * win - (1 - r["pA"]) * risk - cost,
        ev_usd=(r["pA"] * win - (1 - r["pA"]) * risk - cost) * POINT_VALUE[sym],
        sigma=float(s.mean()),
    )


def main() -> int:
    d = pd.read_csv(CELLS)

    print("=" * 108)
    print("  THE LONDON REVERSION, PRICED   (02:00-08:30 CT, VWAP still anchored at the")
    print("  PREVIOUS New York open -- so sigma is 8+ hours mature)")
    print("=" * 108)
    lon = d[(d["arm"] == "all") & (d["bucket"] == "london")].sort_values("t", ascending=False)
    print(
        f"  {'sym':<4}{'tf':<4}{'band':>5}{'nA':>6}{'pA':>7}{'null':>7}{'beats':>7}"
        f"{'t':>7}{'p':>8}{'win':>7}{'risk':>7}{'BE':>7}{'needs':>7}{'EV pt':>8}{'EV $':>8}"
    )
    rows = []
    for _, r in lon.iterrows():
        g = geometry(r["symbol"], r["tf"], "london", int(r["band"]), int(r["sign"]), "all")
        if not g["nA"]:
            continue
        beats = r["pA"] - r["null_mean"]
        needs = g["be"] - g["be0"]
        rows.append(dict(**r, **{f"g_{k}": v for k, v in g.items()}, beats=beats, needs=needs))
        print(
            f"  {r['symbol']:<4}{r['tf']:<4}{int(r['signed']):>+5}{int(r['nA']):>6}"
            f"{r['pA']:>7.3f}{r['null_mean']:>7.3f}{beats:>+7.3f}{r['t']:>+7.2f}"
            f"{r['p_cell']:>8.4f}{g['win']:>7.1f}{g['risk']:>7.1f}{g['be']:>7.3f}"
            f"{needs:>+7.3f}{g['ev_rev']:>+8.2f}{g['ev_usd']:>+8.0f}"
        )
    L = pd.DataFrame(rows)
    if len(L):
        print(
            f"\n  London beats its null by a mean of {L['beats'].mean():+.3f}; "
            f"cost demands a mean of {L['needs'].mean():+.3f}. "
            f"{int((L['g_ev_rev'] > 0).sum())} of {len(L)} cells have positive expectancy."
        )

    print("\n" + "=" * 108)
    print("  THE ACCEPTANCE DISCRIMINATOR, PRICED")
    print("  Does conditioning on a CLOSE beyond the band make any cell tradeable?")
    print("=" * 108)
    acc = pd.read_csv("out/vwap_band_by_session/acceptance.csv")
    top = acc.reindex(acc["t"].sort_values(ascending=False).index).head(14)
    print(
        f"  {'sym':<4}{'tf':<4}{'bucket':<10}{'band':>5}{'t':>7}"
        f"{'nA_acc':>8}{'pA_acc':>8}{'BE_acc':>8}{'EV pt':>8}{'EV $':>8}"
        f"{'nA_wick':>9}{'pA_wick':>9}{'EV$ wick':>10}"
    )
    best = []
    for _, r in top.iterrows():
        ga = geometry(r["symbol"], r["tf"], r["bucket"], int(r["band"]), int(r["sign"]), "accept")
        gw = geometry(r["symbol"], r["tf"], r["bucket"], int(r["band"]), int(r["sign"]), "wick")
        if not ga["nA"] or not gw["nA"]:
            continue
        best.append(ga["ev_usd"])
        print(
            f"  {r['symbol']:<4}{r['tf']:<4}{r['bucket']:<10}"
            f"{int(r['sign']) * int(r['band']):>+5}{r['t']:>+7.2f}"
            f"{int(ga['nA']):>8}{ga['pA']:>8.3f}{ga['be']:>8.3f}"
            f"{ga['ev_rev']:>+8.2f}{ga['ev_usd']:>+8.0f}"
            f"{int(gw['nA']):>9}{gw['pA']:>9.3f}{gw['ev_usd']:>+10.0f}"
        )
    if best:
        print(
            f"\n  best accept-arm expectancy in the top 14 by t: {max(best):+.0f} dollars a trade;"
            f" {sum(1 for x in best if x > 0)} of {len(best)} positive"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
