#!/usr/bin/env python3
"""Why can't we just follow the cross? The most direct version of the question, measured.

    python scripts/follow_the_cross.py

Always in, flip on every cross, flat outside RTH, NO STOPS. This is the purest form of "follow
price until it turns" and it is materially different from everything tested in hypotheses 1-10,
all of which imposed a stop or an exit grid. The 45-cell grid chops you out; pure exposure to
the moving-average state rides.

WHAT THE DECOMPOSITION SHOWS

  gross    following with costs switched off
  cost     every position change, including the daily open and close, charged a round turn
  null     the same strategy on 300 grouped bar permutations
  drawdown peak-to-trough in points and dollars, against the prop trailing limit

THE ACCOUNTING BUG THIS SCRIPT EXISTS TO AVOID

The first version counted a flip only when the cross happened DURING RTH. A state change
overnight then silently repositioned the book for free, and the strategy was never charged for
entering at the open or exiting at the close. That understated costs and turned several losing
configurations into winners. Every position change is a trade.
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
from quant.permutation import get_permutation_fast, session_groups
from quant.strategy import ema

CONFIGS = (
    ("MNQ", "5m", 9, 21),
    ("MNQ", "15m", 20, 50),
    ("MES", "15m", 20, 50),
    ("MES", "5m", 9, 21),
)
N_PERM = 300
MLL = 2000.0  # Topstep $50K trailing maximum loss limit


def steps(c, inwin, fast, slow, cost):
    """Per-bar P&L of always-in, flip-on-cross, flat outside RTH, costs on every turn."""
    a = ema(c, fast) > ema(c, slow)
    pos = np.where(inwin, np.where(a, 1.0, -1.0), 0.0)
    r = np.r_[0.0, np.diff(c)]
    turn = (np.abs(np.diff(np.r_[0.0, pos])) > 0)[:-1]
    return pos[:-1] * r[1:] - turn * cost / 2


def main() -> int:
    for sym, tf, f, s in CONFIGS:
        cost = CostModel.for_prop(sym, "average", contracts=1).round_turn_points
        pv = POINT_VALUE[sym]
        df = S.load_data(DATA[sym].format(tf=tf), "America/Chicago")
        d0 = df["dt"].dt.tz_localize(None).to_numpy()
        df = df.iloc[: make_split(sym, tf, d0).train.stop].reset_index(drop=True)
        c = df["close"].to_numpy(float)
        mins = df["dt"].dt.hour.to_numpy() * 60 + df["dt"].dt.minute.to_numpy()
        inwin = (mins >= 510) & (mins < 900)

        st = steps(c, inwin, f, s, cost)
        eq = np.cumsum(st)
        dd = float((eq - np.maximum.accumulate(eq)).min())
        day = (df["dt"] + pd.Timedelta(hours=7)).dt.strftime("%Y%m%d").to_numpy()[:-1]
        daily = pd.Series(st).groupby(day).sum()
        year = df["dt"].dt.year.to_numpy()[:-1]
        by_year = pd.Series(st).groupby(year).sum()

        g = session_groups(df)
        null = np.array(
            [
                steps(
                    get_permutation_fast(df, seed=k, groups=g)["close"].to_numpy(float),
                    inwin,
                    f,
                    s,
                    cost,
                ).sum()
                for k in range(N_PERM)
            ]
        )
        p = (1 + int((null >= eq[-1]).sum())) / (1 + N_PERM)

        print("=" * 88)
        print(f"  {sym} {tf}  EMA {f}/{s}   always in, flip on cross, no stops")
        print("=" * 88)
        print(f"  net           {eq[-1]:+,.0f} pts = ${eq[-1] * pv:+,.0f} over {len(daily)} days")
        print("  by year       " + "  ".join(f"{int(y)}:{v:+,.0f}" for y, v in by_year.items()))
        print(
            f"  permutation   null median {np.median(null):+,.0f}   "
            f"95th pct {np.percentile(null, 95):+,.0f}   p = {p:.4f}"
        )
        print(
            f"  drawdown      {dd:,.0f} pts = ${dd * pv:,.0f}   "
            f"= {abs(dd) * pv / MLL:.1f}x the ${MLL:,.0f} trailing limit"
        )
        print(
            f"  daily         sd ${daily.std() * pv:,.0f}   "
            f"worst ${daily.min() * pv:,.0f}   best ${daily.max() * pv:,.0f}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
