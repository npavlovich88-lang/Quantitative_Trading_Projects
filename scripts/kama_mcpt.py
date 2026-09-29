"""In-sample MCPT for the KAMA/EMA strategy, with 1m execution.

The subtlety: signals come from 5m bars but trades execute on 1m bars, and the two frames must
stay consistent under permutation. Permuting them independently would pair a 5m signal with 1m
price action from a different shuffled world.

So we permute the 1m frame and RE-DERIVE the 5m bars from it. That is how the real bars were
built, so the permuted pair is internally consistent by construction.

No parameter search was performed here -- the configuration was specified a priori -- so the
MCPT re-runs the FIXED rule on each permutation rather than re-running a search. That is the
correct form when there is nothing to select.
"""

import json
import pathlib
import sys
import time
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from quant import strategy as S
from quant.bar_integrity import derive_timeframe
from quant.costs import CostModel
from quant.data_splits import DATA, make_split
from quant.permutation import get_permutation, session_groups
from quant.strategies.kama_ema import signals, simulate_1m

SYM, N = "MNQ", int(sys.argv[1]) if len(sys.argv) > 1 else 300
P = dict(stop=40.0, tp=125.0, ta=25.0, td=10.0, sep=0.035)
cp = CostModel.for_prop(SYM, "average", contracts=1).round_turn_points

df5 = S.load_data(DATA[SYM].format(tf="5m"), "America/Chicago")
df1 = S.load_data(DATA[SYM].format(tf="1m"), "America/Chicago")
d5 = df5["dt"].dt.tz_localize(None).to_numpy()
sp5 = make_split(SYM, "5m", d5)
lo = int(df5["ts"].iloc[sp5.train.start])
hi = int(df5["ts"].iloc[sp5.train.stop - 1])
one = df1[(df1.ts >= lo) & (df1.ts <= hi + 300)].reset_index(drop=True)
groups = session_groups(one)


def pf_of(df1_frame):
    f5 = derive_timeframe(df1_frame[["ts", "open", "high", "low", "close", "volume"]], "5m")
    f5["dt"] = pd.to_datetime(f5["ts"], unit="s", utc=True).dt.tz_convert("America/Chicago")
    f1 = df1_frame.copy()
    sig = signals(f5, kama_n=20, ema_n=50, min_sep_atr=P["sep"])
    tr = simulate_1m(
        sig,
        f1,
        cost_pts=cp,
        stop_pts=P["stop"],
        tp_pts=P["tp"],
        trail_after=P["ta"],
        trail_dist=P["td"],
    )
    if len(tr) < 20:
        return np.nan, 0
    w = tr.pnl_pts > 0
    gl = -tr.pnl_pts[~w].sum()
    return (tr.pnl_pts[w].sum() / gl if gl > 0 else np.nan), len(tr)


real, n_real = pf_of(one)
print(f"REAL: PF {real:.4f} on {n_real} trades (train, 1m execution)", flush=True)

better, perms, t0 = 1, [], time.time()
for i in range(1, N):
    p1 = get_permutation(one, seed=i, groups=groups)
    v, _ = pf_of(p1)
    if np.isfinite(v):
        perms.append(v)
        if v >= real:
            better += 1
    if i % 20 == 0:
        el = time.time() - t0
        print(
            f"  {i}/{N}  beaten {better - 1}x  p~{better / (i + 1):.3f}  "
            f"({el / i:.1f}s/perm, {(N - i) * el / i / 60:.0f}min left)",
            flush=True,
        )
pv = better / N
q = np.percentile(perms, [50, 90, 99])
print(
    f"\nMCPT p = {pv:.4f}  ({better - 1} of {len(perms)} permutations matched or beat {real:.4f})",
    flush=True,
)
print(f"permutation PF: median {q[0]:.4f}  p90 {q[1]:.4f}  p99 {q[2]:.4f}", flush=True)
print(f"GATE 5 (p < 0.01): {'PASS' if pv < 0.01 else 'FAIL'}", flush=True)
pathlib.Path("out/kama_MNQ").mkdir(parents=True, exist_ok=True)
with open("out/kama_MNQ/mcpt.json", "w") as fh:
    json.dump(dict(real=real, p=pv, perms=perms), fh)
