"""HMA, TEMA and session VWAP must be causal and correct.

The prefix test is the one that matters: f(x[:i+1])[i] must equal f(x)[i]. A boolean signal can
hide a small leak, so the lines themselves are checked, not the crosses derived from them. This
is the test that caught a one-bar peek in mid_donchian earlier in the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.information import evaluate, verify
from quant.strategies.ma_pairs import (
    build_pair,
    cross_masks,
    hma,
    pair_family,
    session_vwap,
    wma,
)
from quant.strategy import ema, epoch_seconds, tema


def synthetic(n: int = 4000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 4000 * np.exp(np.cumsum(rng.normal(0, 4e-4, n)))
    spread = np.abs(rng.normal(0, 1.5, n)) + 0.25
    o = c + rng.normal(0, 1.0, n)
    dt = pd.date_range("2022-01-03 00:00", periods=n, freq="30min", tz="America/Chicago")
    return pd.DataFrame(
        dict(
            dt=dt,
            ts=epoch_seconds(dt),
            open=o,
            high=np.maximum(o, c) + spread,
            low=np.minimum(o, c) - spread,
            close=c,
            volume=rng.integers(1, 5000, n).astype(float),
        )
    )


@pytest.mark.parametrize("n", [3, 5, 9, 21])
def test_wma_matches_direct_definition(n: int) -> None:
    x = np.arange(60, dtype=float) ** 1.3
    got = wma(x, n)
    w = np.arange(1, n + 1, dtype=float)
    for i in range(n - 1, len(x)):
        assert np.isclose(got[i], (x[i - n + 1 : i + 1] * w).sum() / w.sum())
    assert np.all(np.isnan(got[: n - 1]))


def test_wma_weights_the_newest_bar_most() -> None:
    """A spike in the last bar must move a WMA more than the same spike n-1 bars back."""
    n = 10
    a = np.ones(40)
    b = a.copy()
    b[39] += 1.0
    late = wma(b, n)[39] - wma(a, n)[39]
    c = a.copy()
    c[30] += 1.0
    early = wma(c, n)[39] - wma(a, n)[39]
    assert late > early > 0


@pytest.mark.parametrize(("fn", "arg"), [(hma, 21), (hma, 55), (tema, 14), (ema, 50)])
@pytest.mark.parametrize("i", [500, 1500, 3000])
def test_line_is_causal(fn, arg, i: int) -> None:
    """Truncating the series after bar i must not change the line's value at bar i."""
    c = synthetic()["close"].to_numpy(float)
    assert np.isclose(fn(c, arg)[i], fn(c[: i + 1], arg)[i], rtol=1e-9)


@pytest.mark.parametrize("i", [500, 1500, 3000])
def test_session_vwap_is_causal(i: int) -> None:
    df = synthetic()
    assert np.isclose(session_vwap(df)[i], session_vwap(df.iloc[: i + 1])[i], rtol=1e-12)


def test_session_vwap_resets_each_session() -> None:
    """On the first bar of a session, VWAP must equal that bar's own typical price."""
    df = synthetic()
    v = session_vwap(df)
    day = (df["dt"] + pd.Timedelta(hours=7)).dt.strftime("%Y-%m-%d").to_numpy()
    first = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    tp = (df["high"] + df["low"] + df["close"]).to_numpy() / 3.0
    assert np.allclose(v[first], tp[first])
    assert len(first) > 5


def test_session_vwap_sits_inside_the_session_range() -> None:
    df = synthetic()
    v = session_vwap(df)
    day = (df["dt"] + pd.Timedelta(hours=7)).dt.strftime("%Y-%m-%d").to_numpy()
    for d in np.unique(day)[:20]:
        m = day == d
        assert v[m].min() >= df["low"].to_numpy()[m].min() - 1e-9
        assert v[m].max() <= df["high"].to_numpy()[m].max() + 1e-9


def test_family_is_the_declared_size_and_has_no_inverted_duplicates() -> None:
    fam = pair_family()
    assert len(fam) == 61
    assert len({s.label for s in fam}) == 61
    ema_pairs = [s for s in fam if s.fast_kind == "ema"]
    assert all(s.fast_n < s.slow_n for s in ema_pairs)
    assert {s.family for s in fam} == {"hma", "tema", "vwap", "ema"}


def test_cross_masks_are_exclusive_and_never_fire_in_warmup() -> None:
    df = synthetic()
    cache: dict = {}
    for spec in pair_family()[:8]:
        f, s = build_pair(spec, df, cache)
        up, dn = cross_masks(f, s)
        assert not (up & dn).any()
        warm = ~(np.isfinite(f) & np.isfinite(s))
        assert not (up & warm).any()
        assert not (dn & warm).any()


def test_information_fast_path_matches_exact_recomputation() -> None:
    """The float32 matmul in information.evaluate must agree with a float64 reduction."""
    from quant.information import DEFAULT_HORIZONS, atr_shifted

    df = synthetic(n=20000)
    c = df["close"].to_numpy(float)
    atr = atr_shifted(df)
    cache: dict = {}
    masks = [cross_masks(*build_pair(s, df, cache)) for s in pair_family()]
    base = (np.arange(len(c)) >= 600) & np.isfinite(atr) & (atr > 0)
    cells = evaluate(c, atr, masks, base)
    assert np.isfinite(cells.t).any()
    worst = verify(c, atr, masks, base, cells, DEFAULT_HORIZONS, n_check=10)
    assert worst < 1e-3, worst
