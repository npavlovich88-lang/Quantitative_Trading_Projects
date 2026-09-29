"""The vectorised permutation must agree with the reference implementation.

The fast path exists only to make a 2,400-cell family test affordable. If it disagrees with
`get_permutation` in any way that could change a signal, the speed is worthless -- so this
checks both the numbers and the thing the numbers are used for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.permutation import get_permutation, get_permutation_fast, session_groups
from quant.strategy import epoch_seconds


def synthetic(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 4000 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    spread = np.abs(rng.normal(0, 1.5, n)) + 0.25
    o = c + rng.normal(0, 1.0, n)
    h = np.maximum(o, c) + spread
    lo = np.minimum(o, c) - spread
    dt = pd.date_range("2022-01-03 00:00", periods=n, freq="5min", tz="America/Chicago")
    return pd.DataFrame(
        dict(
            dt=dt,
            ts=epoch_seconds(dt),
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=rng.integers(1, 500, n).astype(float),
        )
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("grouped", [False, True])
def test_fast_matches_reference_in_log_space(seed: int, grouped: bool) -> None:
    df = synthetic()
    g = session_groups(df) if grouped else None
    a = get_permutation(df, start_index=100, seed=seed, groups=g)
    b = get_permutation_fast(df, start_index=100, seed=seed, groups=g)
    for col in ("open", "high", "low", "close"):
        la = np.log(a[col].to_numpy())
        lb = np.log(b[col].to_numpy())
        assert np.max(np.abs(la - lb)) < 1e-12, col
    assert np.array_equal(a["volume"].to_numpy(), b["volume"].to_numpy())


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_fast_produces_identical_cross_events(seed: int) -> None:
    """The downstream property: where price crosses a mid-Donchian line must not move."""
    df = synthetic()
    a = get_permutation(df, start_index=100, seed=seed)
    b = get_permutation_fast(df, start_index=100, seed=seed)

    def crosses(d: pd.DataFrame, n: int = 52) -> np.ndarray:
        h = pd.Series(d["high"]).rolling(n).max().to_numpy()
        lo = pd.Series(d["low"]).rolling(n).min().to_numpy()
        mid = (h + lo) / 2.0
        c = d["close"].to_numpy()
        above = c > mid
        return np.flatnonzero(above & ~np.r_[False, above[:-1]])

    assert np.array_equal(crosses(a), crosses(b))


def test_epoch_seconds_survives_the_pandas_resolution_change() -> None:
    """pandas 3 infers datetime resolution; `astype("int64") // 10**9` silently returns
    epoch-seconds/1000 when the unit is microseconds. This is the guard for that."""
    for unit in ("s", "ms", "us", "ns"):
        dt = pd.to_datetime(pd.Series(["2022-01-03 06:00:00"]), utc=True).astype(
            f"datetime64[{unit}, UTC]"
        )
        assert int(epoch_seconds(dt).iloc[0]) == 1641189600, unit
    dr = pd.date_range("2022-01-03 06:00", periods=3, freq="5min", tz="America/Chicago")
    assert list(epoch_seconds(dr)) == [1641211200, 1641211500, 1641211800]
