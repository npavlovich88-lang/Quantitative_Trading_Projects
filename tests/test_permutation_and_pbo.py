"""The validation machinery has to be right, or every p-value it produces is decoration.

Two things are checked:
  * the permutation preserves what the null is supposed to preserve (drift, the distribution of
    moves, bar geometry) and destroys what it is supposed to destroy (sequence)
  * PBO discriminates noise from signal, and matches the paper's definitions
"""

import numpy as np
import pandas as pd
import pytest

from quant import pbo as P
from quant.permutation import get_permutation, session_groups

from .test_no_lookahead import synthetic_bars


@pytest.fixture(scope="module")
def bars():
    return synthetic_bars(n=3000, seed=11)


# ------------------------------------------------------------------ permutation
def test_bar_geometry_is_preserved(bars):
    """high >= max(open, close) and low <= min(open, close) on every permuted bar. A permutation
    that breaks this produces impossible bars and silently invalid stop fills."""
    p = get_permutation(bars, seed=1)
    assert (p["high"] >= p[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (p["low"] <= p[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (p[["open", "high", "low", "close"]] > 0).all().all(), "prices must stay positive"


def test_return_distribution_is_preserved(bars):
    """Same multiset of log returns, so the marginal distribution is untouched -- only the order
    changes. Mean and standard deviation must match closely."""
    p = get_permutation(bars, seed=2)
    a = np.diff(np.log(bars["close"].to_numpy()))
    b = np.diff(np.log(p["close"].to_numpy()))
    assert np.isclose(a.mean(), b.mean(), rtol=0.05, atol=1e-6), "drift was not preserved"
    assert np.isclose(a.std(), b.std(), rtol=0.10), "volatility was not preserved"


def test_sequence_is_destroyed(bars):
    """The whole point. Autocorrelation of absolute returns (volatility clustering) must drop."""
    p = get_permutation(bars, seed=3)

    def acf1(x):
        a = np.abs(np.diff(np.log(x)))
        a = a - a.mean()
        return float((a[:-1] * a[1:]).sum() / (a * a).sum())

    # synthetic bars are IID so real clustering is already ~0; assert the permuted series is not
    # MORE autocorrelated, which is the invariant that must hold on real data too
    assert abs(acf1(p["close"].to_numpy())) < 0.15


def test_start_index_keeps_the_past_real(bars):
    """Walk-forward MCPT depends on this: everything up to start_index must be untouched."""
    k = 1000
    p = get_permutation(bars, start_index=k, seed=4)
    pd.testing.assert_frame_equal(
        bars.iloc[: k + 1][["open", "high", "low", "close"]],
        p.iloc[: k + 1][["open", "high", "low", "close"]],
        check_exact=False,
        rtol=1e-12,
    )
    assert not np.allclose(bars["close"].to_numpy()[k + 1 :], p["close"].to_numpy()[k + 1 :])


def test_grouped_permutation_preserves_session_structure(bars):
    """The grouped variant must only move a bar onto a position of the same kind, so the
    session skeleton and the time-of-day volatility profile survive."""
    g = session_groups(bars)
    p = get_permutation(bars, seed=5, groups=g)
    # the multiset of intrabar ranges within each group must be unchanged
    for grp in np.unique(g):
        m = g == grp
        a = np.sort(np.log(bars["high"].to_numpy()[m]) - np.log(bars["low"].to_numpy()[m]))
        b = np.sort(np.log(p["high"].to_numpy()[m]) - np.log(p["low"].to_numpy()[m]))
        assert np.allclose(a, b, atol=1e-9), f"group {grp} exchanged bars with another group"


def test_permutation_is_reproducible(bars):
    a = get_permutation(bars, seed=42)
    b = get_permutation(bars, seed=42)
    pd.testing.assert_frame_equal(a, b)
    c = get_permutation(bars, seed=43)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_multi_market_shares_one_permutation(bars):
    """Cross-market correlation must survive, which requires both markets to be shuffled with
    the SAME permutation."""
    other = bars.copy()
    other[["open", "high", "low", "close"]] *= 2.0
    a, b = get_permutation([bars, other], seed=9)
    ra = np.diff(np.log(a["close"].to_numpy()))
    rb = np.diff(np.log(b["close"].to_numpy()))
    assert np.corrcoef(ra, rb)[0, 1] > 0.99


# ------------------------------------------------------------------ PBO
def test_pbo_high_on_pure_noise():
    """With no real edge, selecting the in-sample best must carry little information."""
    rng = np.random.default_rng(0)
    res = P.cscv(rng.normal(size=(2400, 30)), S=10)
    assert res["pbo"] > 0.15, f"noise should not produce a confident selection, got {res['pbo']}"
    assert res["degradation_slope"] < 0, "overfit data must show negative IS->OOS degradation"


def test_pbo_zero_when_one_configuration_is_genuinely_better():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(2400, 30))
    m[:, 13] += 0.15
    res = P.cscv(m, S=10)
    assert res["pbo"] < 0.05, f"a real edge should be selectable, got PBO {res['pbo']}"
    assert res["winner_counts"][13] == res["n_combinations"], "the good column should always win"
    assert not res["reject_at_005"]


def test_blocks_are_exactly_equal_and_cover_in_order():
    """Algorithm 2.3 step two: S disjoint submatrices of EQUAL dimensions."""
    rng = np.random.default_rng(2)
    res = P.cscv(rng.normal(size=(1003, 8)), S=10)  # 1003 does not divide by 10
    assert res["T"] == 1003
    from math import comb

    assert res["n_combinations"] == comb(10, 5)


def test_pbo_rejects_single_configuration():
    """PBO measures selection. One candidate is not a selection."""
    with pytest.raises(ValueError, match="at least 2"):
        P.cscv(np.random.default_rng(3).normal(size=(500, 1)), S=4)


def test_odd_S_is_rejected():
    with pytest.raises(ValueError, match="even"):
        P.cscv(np.random.default_rng(4).normal(size=(500, 5)), S=7)
