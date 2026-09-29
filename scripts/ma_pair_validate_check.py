"""ONE pre-committed cell, on the VALIDATE segment. No optimisation, no alternatives.

The train run nominally passed its pre-registered primary gate at p=0.042, carried entirely by
MNQ 10m tema55/ema21 at h=6. Two standing project rules disagree with that pass: the winner is a
lone spike on the parameter axis, and it has no cross-market support. Rather than argue the
result away post hoc, this spends one validate look on exactly that configuration.

Family size 1, so the per-cell permutation p-value IS the correct p-value. Nothing is selected
here -- the cell was fixed before this file existed.
"""

import sys

import numpy as np

sys.path.insert(0, "src")
from quant import strategy as S
from quant.data_splits import DATA, make_split
from quant.information import atr_shifted
from quant.permutation import get_permutation_fast, session_groups
from quant.strategies.ma_pairs import PairSpec, build_pair, cross_masks

SPEC, H, WARMUP, NPERM = PairSpec("tema", 55, 21), 6, 1000, 2000


def tstat(df, base):
    c = df["close"].to_numpy(float)
    atr = atr_shifted(df)
    n = len(c)
    up, dn = cross_masks(*build_pair(SPEC, df, {}))
    fr = np.full(n, np.nan)
    fr[: n - H] = (c[H:] - c[: n - H]) / atr[: n - H]
    ok = np.isfinite(fr)
    a, b = fr[up & base & ok], fr[dn & base & ok]
    if len(a) < 30 or len(b) < 30:
        return np.nan, len(a), len(b), np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    d = a.mean() - b.mean()
    return (d / se if se > 0 else np.nan), len(a), len(b), d


for sym in ("MNQ", "MES"):
    df = S.load_data(DATA[sym].format(tf="10m"), "America/Chicago")
    dates = df["dt"].dt.tz_localize(None).to_numpy()
    sp = make_split(sym, "10m", dates)
    lo, hi = sp.validate.start, sp.validate.stop
    seg = df.iloc[max(lo - WARMUP, 0) : hi].reset_index(drop=True)
    n = len(seg)
    off = lo - max(lo - WARMUP, 0)
    mins = seg["dt"].dt.hour.to_numpy() * 60 + seg["dt"].dt.minute.to_numpy()
    atr = atr_shifted(seg)
    base = (mins >= 510) & (mins < 900) & (np.arange(n) >= off) & np.isfinite(atr) & (atr > 0)
    t, na, nb, d = tstat(seg, base)
    if not np.isfinite(t):
        print(f"{sym} 10m validate: only {na} up / {nb} down events -- too few to test")
        continue
    g = session_groups(seg)
    null = np.array(
        [
            tstat(get_permutation_fast(seg, start_index=off, seed=k, groups=g), base)[0]
            for k in range(NPERM)
        ]
    )
    null = null[np.isfinite(null)]
    p = (1 + int((np.abs(null) >= abs(t)).sum())) / (1 + len(null))
    print(
        f"{sym} 10m tema55/ema21 h=6 on VALIDATE ({seg['dt'].iloc[off].date()} -> "
        f"{seg['dt'].iloc[-1].date()}):"
    )
    print(f"   t = {t:+.3f}   d = {d:+.4f} ATR   {na} up / {nb} down events")
    print(f"   permutation p (family size 1, {len(null)} draws) = {p:.4f}")
    print(
        f"   null |t|: median {np.median(np.abs(null)):.2f}, 95th pct {np.percentile(np.abs(null), 95):.2f}"
    )
    print("   TRAIN was t = +3.98\n")
