"""Bar-file integrity checks. Run on every dataset before it is trusted for a backtest.

A backtest inherits every defect in its bars silently. A single impossible bar (high < close)
lets a stop fill at a price that never traded; a duplicated timestamp double-counts a trade; a
timeframe that does not aggregate from its own 1m source means two backtests on "the same data"
disagree for reasons nobody can find later.

None of these announce themselves. They show up as an unusually good equity curve.

CHECKS
    schema          required columns, types, no nulls in OHLC
    monotonic       timestamps strictly increasing, no duplicates
    ohlc_sane       high >= max(open, close), low <= min(open, close), high >= low, all > 0
    spacing         bar spacing matches the nominal timeframe
    volume          no negative volume; zero-volume bars counted (legal but worth knowing)
    coverage        calendar gaps (delegated to diagnostics.data_coverage)
    jumps           implausible bar-to-bar moves, which flag roll/back-adjust splices
    cross_tf        does the coarse timeframe actually aggregate from the 1m file?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .diagnostics import data_coverage

REQUIRED = ("ts", "open", "high", "low", "close", "volume")
NOMINAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "10m": 600,
    "15m": 900,
    "20m": 1200,
    "30m": 1800,
    "45m": 2700,
    "60m": 3600,
}


def _v(fail: bool, warn: bool) -> str:
    return "fail" if fail else ("warn" if warn else "pass")


def check_schema(df: pd.DataFrame) -> dict:
    missing = [c for c in REQUIRED if c not in df.columns]
    nulls = {c: int(df[c].isna().sum()) for c in REQUIRED if c in df.columns}
    bad = {k: v for k, v in nulls.items() if v}
    return dict(
        verdict=_v(bool(missing) or bool(bad), False),
        missing=missing,
        nulls=bad,
        rows=len(df),
        columns=list(df.columns),
    )


def check_monotonic(df: pd.DataFrame) -> dict:
    """Duplicate and out-of-order timestamps.

    Count duplicates with `duplicated()`, NOT with `np.diff(ts) == 0`. The diff version only
    sees duplicates that are ADJACENT in file order, and a file assembled from per-day chunks
    can repeat a timestamp far apart. On MNQ 1m the diff version reported 1 duplicate; the real
    count was 145. An integrity check that undercounts by 145x is worse than none, because it
    is believed.
    """
    ts = df["ts"]
    dupe_rows = int(ts.duplicated(keep=False).sum())
    dupe_stamps = int(ts[ts.duplicated(keep=False)].nunique())
    adjacent = int((np.diff(ts.to_numpy()) == 0).sum())
    backwards = int((np.diff(ts.to_numpy()) < 0).sum())
    return dict(
        verdict=_v(dupe_stamps > 0 or backwards > 0, False),
        duplicate_timestamps=dupe_stamps,
        duplicate_rows=dupe_rows,
        adjacent_duplicates=adjacent,
        backwards_steps=backwards,
        note="a duplicate timestamp double-counts a bar; repair with merge_duplicate_bars(), "
        "which aggregates them rather than dropping an arbitrary one",
    )


def check_ohlc_sane(df: pd.DataFrame) -> dict:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    hi_bad = int((h < np.maximum(o, c) - 1e-9).sum())
    lo_bad = int((l > np.minimum(o, c) + 1e-9).sum())
    hl_bad = int((h < l - 1e-9).sum())
    nonpos = int((np.minimum.reduce([o, h, l, c]) <= 0).sum())
    total = hi_bad + lo_bad + hl_bad + nonpos
    return dict(
        verdict=_v(total > 0, False),
        high_below_body=hi_bad,
        low_above_body=lo_bad,
        high_below_low=hl_bad,
        non_positive_prices=nonpos,
        note="an impossible bar lets a stop or target fill at a price that never traded",
    )


def check_spacing(df: pd.DataFrame, timeframe: str) -> dict:
    want = NOMINAL_SECONDS.get(timeframe)
    d = np.diff(df["ts"].to_numpy())
    if want is None or d.size == 0:
        return dict(verdict="warn", reason="unknown timeframe or too few bars")
    on_grid = int((d == want).sum())
    # gaps are expected (sessions, weekends); what matters is that the MODAL step is nominal
    # and that nothing is SHORTER than nominal, which would mean overlapping bars
    short = int((d < want).sum())
    vals, counts = np.unique(d, return_counts=True)
    modal = int(vals[np.argmax(counts)])
    return dict(
        verdict=_v(short > 0 or modal != want, False),
        nominal_seconds=want,
        modal_step_seconds=modal,
        on_grid=on_grid,
        on_grid_pct=float(on_grid / d.size),
        shorter_than_nominal=short,
        note="a step shorter than nominal means overlapping bars",
    )


def check_volume(df: pd.DataFrame) -> dict:
    v = df["volume"].to_numpy(float)
    neg = int((v < 0).sum())
    zero = int((v == 0).sum())
    return dict(
        verdict=_v(neg > 0, zero > len(v) * 0.02),
        negative=neg,
        zero_volume_bars=zero,
        zero_pct=float(zero / max(len(v), 1)),
        median=float(np.median(v)),
        max=float(v.max()) if v.size else 0.0,
    )


def check_jumps(df: pd.DataFrame, *, atr_mult: float = 10.0) -> dict:
    """Bar-to-bar moves too large to be real -- roll splices, bad ticks, back-adjustment."""
    c = df["close"].to_numpy(float)
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    step = np.abs(np.diff(c, prepend=c[0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(atr > 0, step / atr, 0.0)
    idx = np.flatnonzero(z > atr_mult)
    return dict(
        verdict=_v(False, idx.size > 0),
        n_jumps=int(idx.size),
        threshold_atr=atr_mult,
        largest_atr=float(np.nanmax(z)) if z.size else 0.0,
        worst=[dict(i=int(i), atr=round(float(z[i]), 1)) for i in idx[np.argsort(-z[idx])][:5]],
    )


def check_cross_timeframe(
    df_1m: pd.DataFrame, df_tf: pd.DataFrame, timeframe: str, *, sample: int = 500, seed: int = 0
) -> dict:
    """Does the coarse file actually aggregate from the 1m file?

    The check people skip, and the one that catches a stale or differently-built file. If a 10m
    bar's high is not the max of its ten 1m highs, the two files disagree about reality and
    every result depending on which one was loaded is unreproducible.
    """
    secs = NOMINAL_SECONDS.get(timeframe)
    if secs is None:
        return dict(verdict="warn", reason=f"unknown timeframe {timeframe}")
    one = df_1m.set_index("ts")
    rng = np.random.default_rng(seed)
    cand = df_tf["ts"].to_numpy()
    pick = rng.choice(len(cand), size=min(sample, len(cand)), replace=False)
    checked = mismatch = empty = 0
    examples = []
    for i in pick:
        t0 = int(cand[i])
        seg = one.loc[(one.index >= t0) & (one.index < t0 + secs)]
        if seg.empty:
            empty += 1
            continue
        checked += 1
        row = df_tf.iloc[i]
        ok = (
            np.isclose(row["open"], seg["open"].iloc[0])
            and np.isclose(row["close"], seg["close"].iloc[-1])
            and np.isclose(row["high"], seg["high"].max())
            and np.isclose(row["low"], seg["low"].min())
            and np.isclose(row["volume"], seg["volume"].sum())
        )
        if not ok:
            mismatch += 1
            if len(examples) < 3:
                examples.append(
                    dict(
                        ts=t0,
                        tf_high=float(row["high"]),
                        from_1m_high=float(seg["high"].max()),
                        tf_vol=float(row["volume"]),
                        from_1m_vol=float(seg["volume"].sum()),
                    )
                )
    return dict(
        verdict=_v(mismatch > 0, empty > checked * 0.05),
        sampled=len(pick),
        checked=checked,
        mismatches=mismatch,
        no_1m_data=empty,
        examples=examples,
        note="a mismatch means the coarse file was not derived from this 1m file",
    )


def check_file(
    df: pd.DataFrame,
    timeframe: str,
    *,
    df_1m: pd.DataFrame | None = None,
    dates: np.ndarray | None = None,
) -> dict:
    out = {
        "schema": check_schema(df),
        "monotonic": check_monotonic(df),
        "ohlc_sane": check_ohlc_sane(df),
        "spacing": check_spacing(df, timeframe),
        "volume": check_volume(df),
        "jumps": check_jumps(df),
    }
    if dates is not None:
        out["coverage"] = data_coverage(dates)
    if df_1m is not None and timeframe != "1m":
        out["cross_tf"] = check_cross_timeframe(df_1m, df, timeframe)
    fails = [k for k, v in out.items() if v.get("verdict") == "fail"]
    warns = [k for k, v in out.items() if v.get("verdict") == "warn"]
    out["_summary"] = dict(verdict=_v(bool(fails), bool(warns)), failed=fails, warned=warns)
    return out


# --------------------------------------------------------------------- repair
def merge_duplicate_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse duplicate-timestamp bars by AGGREGATING them, not by dropping one.

    Day-file boundaries can emit the same minute twice: the tail of one day's file and the head
    of the next. `drop_duplicates` keeps whichever happened to sort first, which is arbitrary --
    on MNQ it kept a 1-contract fragment and discarded the 150-contract bar for the same minute,
    and that wrong bar then propagated into every coarser timeframe.

    The correct collapse is the same aggregation used to build a bar in the first place:
    open = first, high = max, low = min, close = last, volume = sum.

    Returns (repaired, n_timestamps_merged).
    """
    dupe_mask = df["ts"].duplicated(keep=False)
    n = int(df.loc[dupe_mask, "ts"].nunique())
    if n == 0:
        return df.sort_values("ts").reset_index(drop=True), 0
    merged = (
        df.sort_values("ts", kind="stable")
        .groupby("ts", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
    )
    return merged.sort_values("ts").reset_index(drop=True), n


def derive_timeframe(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate a clean 1m frame to a coarser timeframe, matching the builder's convention
    (label='left', closed='left')."""
    secs = NOMINAL_SECONDS[timeframe]
    d = df_1m.copy()
    d["dt"] = pd.to_datetime(d["ts"], unit="s", utc=True)
    agg = (
        d.set_index("dt")
        .resample(f"{secs // 60}min", label="left", closed="left")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open"])
        .reset_index()
    )
    agg["ts"] = (agg["dt"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    return agg[["ts", "open", "high", "low", "close", "volume"]]
