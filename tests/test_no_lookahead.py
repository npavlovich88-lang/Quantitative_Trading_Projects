"""The single most important test in this repo.

Look-ahead bias is the failure mode that makes a worthless strategy look excellent, and it is
invisible in the output -- an equity curve built on leaked information looks exactly like an
equity curve built on edge. So we do not inspect the code for it; we prove its absence.

METHOD
  Take real bars. Compute every feature. Now corrupt the future -- multiply all prices from bar
  k onward by 1.5 -- and recompute. If any feature at a bar before k moves, that feature read
  data it could not have had. The test is indifferent to how the indicator is implemented,
  which is the point: it would catch a look-ahead introduced by a future refactor just as well.

WHY 1.5x AND NOT NOISE
  A large, structured corruption. Additive noise can cancel; a 50% level shift cannot be
  absorbed by any causal filter without showing up.
"""

import numpy as np
import pandas as pd
import pytest

from quant import strategy as S

BARS = 4000
K = 2500  # corrupt everything from here on


def synthetic_bars(n=BARS, seed=7):
    """Deterministic OHLCV with a real intraday session structure, so the RTH logic engages."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.0012, n)
    close = 15000 * np.exp(np.cumsum(r))
    spread = np.abs(rng.normal(0, 0.0008, n)) * close
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0003, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    # 10-minute bars starting at 17:00 CT so a full session cycle is covered
    ts = pd.date_range("2024-01-02 17:00", periods=n, freq="10min", tz="America/Chicago")
    return pd.DataFrame(
        {
            "ts": ts.view("int64") // 10**9,
            "dt": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100, 5000, n).astype(float),
        }
    )


def corrupt_future(df, k, factor=1.5):
    d = df.copy()
    for col in ("open", "high", "low", "close"):
        d.loc[d.index >= k, col] = d.loc[d.index >= k, col] * factor
    d.loc[d.index >= k, "volume"] = d.loc[d.index >= k, "volume"] * 3.0
    return d


FEATURES = ["atr", "el", "es", "ok_long", "ok_short", "st_bull", "inwin"]


@pytest.fixture(scope="module")
def pair():
    df = synthetic_bars()
    return S.build_features(df, "08:30-15:00"), S.build_features(
        corrupt_future(df, K), "08:30-15:00"
    )


@pytest.mark.parametrize("name", FEATURES)
def test_feature_is_causal(pair, name):
    """No feature before bar K may react to prices at or after bar K."""
    real, fake = pair
    a, b = np.asarray(real[name])[:K], np.asarray(fake[name])[:K]
    if a.dtype == bool:
        bad = np.flatnonzero(a != b)
    else:
        bad = np.flatnonzero(~(np.isclose(a, b, equal_nan=True)))
    assert bad.size == 0, (
        f"{name} leaked future data at {bad.size} bars before K={K}; "
        f"first at index {bad[0] if bad.size else '-'}"
    )


def test_indicator_primitives_are_causal():
    """The corruption test above is necessary but not sufficient: the exported features are
    mostly BOOLEAN, so a small leak can move a threshold without flipping it, and the test
    passes on a strategy that peeks. (Found exactly that way -- a one-bar peek inserted into
    mid_donchian survived the corruption test.)

    This is the sharp version. For a causal function, the value at bar i must not depend on
    anything after i, so computing it on the PREFIX x[:i+1] must give the same answer at i as
    computing it on the whole series. Continuous output, no threshold to hide behind.
    """
    df = synthetic_bars(n=600, seed=3)
    h, l, c, v = (df[k].to_numpy(float) for k in ("high", "low", "close", "volume"))

    cases = {
        "ema": lambda h, l, c, v: S.ema(c, 20),
        "tema": lambda h, l, c, v: S.tema(c, 14),
        "mid_donchian(9)": lambda h, l, c, v: S.mid_donchian(h, l, 9),
        "mid_donchian(52)": lambda h, l, c, v: S.mid_donchian(h, l, 52),
        "true_range": lambda h, l, c, v: S.true_range(h, l, c),
        "wilder_smooth": lambda h, l, c, v: S.wilder_smooth(S.true_range(h, l, c), 14),
        "efficiency_ratio": lambda h, l, c, v: S.efficiency_ratio(c, 14),
        "adx": lambda h, l, c, v: S.adx_dmi(h, l, c, 14)[0],
        "choppiness": lambda h, l, c, v: S.choppiness(h, l, c, 14),
        "supertrend": lambda h, l, c, v: S.supertrend(h, l, c, 10, 3.0),
    }
    probes = [199, 300, 401, 500, 599]
    for name, fn in cases.items():
        full = np.asarray(fn(h, l, c, v), dtype=float)
        for i in probes:
            prefix = np.asarray(fn(h[: i + 1], l[: i + 1], c[: i + 1], v[: i + 1]), dtype=float)
            a, b = full[i], prefix[i]
            if np.isnan(a) and np.isnan(b):
                continue
            assert np.isclose(a, b, rtol=1e-9, atol=1e-9), (
                f"{name} is NOT causal at bar {i}: whole-series value {a!r} != "
                f"prefix-only value {b!r}. It is reading data from after bar {i}."
            )


def test_atr_is_shifted_one_bar(pair):
    """The R-unit must not know the current bar's range -- it sizes the stop on that bar."""
    real, _ = pair
    tr = S.true_range(real["h"], real["l"], real["c"])
    unshifted = pd.Series(tr).ewm(alpha=1 / S.ATR_N, adjust=False).mean().to_numpy()
    atr = np.asarray(real["atr"])
    ok = ~np.isnan(atr[1:])
    assert np.allclose(atr[1:][ok], unshifted[:-1][ok]), (
        "features['atr'] is not the previous bar's ATR; a stop sized from the current bar's "
        "own range is look-ahead"
    )
    assert np.isnan(atr[0]), "the first bar cannot have a previous-bar ATR"


def test_trades_closed_before_k_are_unaffected(pair):
    """A trade that both opened and closed before K must be identical under a corrupted future."""
    real, fake = pair
    a = S.simulate(real, "fixed", 6.0, 2.0, 2.5)
    b = S.simulate(fake, "fixed", 6.0, 2.0, 2.5)
    a = a[a["exit_i"] < K].reset_index(drop=True)
    b = b[b["exit_i"] < K].reset_index(drop=True)
    assert len(a) == len(b), f"trade count before K changed: {len(a)} vs {len(b)}"
    if len(a):
        assert np.allclose(a["pnl_pts"], b["pnl_pts"]), (
            "a closed trade's P&L moved when the future changed"
        )
        assert (a["entry_i"].to_numpy() == b["entry_i"].to_numpy()).all()
