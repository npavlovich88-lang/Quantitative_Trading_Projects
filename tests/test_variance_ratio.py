"""The variance ratio has to be right on series whose answer is known in advance.

This is the one statistic in the project with an analytic null, so it can be validated against
constructed processes rather than only against itself: a random walk must give VR = 1, an AR(1)
with positive phi must give VR > 1, negative phi must give VR < 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant.variance_ratio import (
    conditional_returns,
    variance_ratio,
    variance_ratio_segmented,
    vr_edge_points,
    vr_profile,
)


def ar1(n: int, phi: float, seed: int = 0, sigma: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    r = np.empty(n)
    r[0] = e[0]
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    return r


@pytest.mark.parametrize("q", [2, 4, 8, 16])
def test_random_walk_gives_vr_near_one(q: int) -> None:
    rng = np.random.default_rng(7)
    r = rng.normal(0, 1, 200_000)
    res = variance_ratio(r, q)
    assert abs(res["vr"] - 1.0) < 0.03, res
    assert abs(res["m2"]) < 3.0, res


@pytest.mark.parametrize("phi", [0.10, 0.20])
def test_positive_autocorrelation_gives_vr_above_one(phi: float) -> None:
    r = ar1(200_000, phi, seed=1)
    res = variance_ratio(r, 8)
    assert res["vr"] > 1.05, res
    assert res["m2"] > 3.0, res


@pytest.mark.parametrize("phi", [-0.10, -0.20])
def test_negative_autocorrelation_gives_vr_below_one(phi: float) -> None:
    r = ar1(200_000, phi, seed=2)
    res = variance_ratio(r, 8)
    assert res["vr"] < 0.95, res
    assert res["m2"] < -3.0, res


def test_vr_matches_its_theoretical_value_for_ar1() -> None:
    """For AR(1), VR(q) has a closed form. The estimator must land on it."""
    phi, q, n = 0.15, 16, 400_000
    r = ar1(n, phi, seed=3)
    k = np.arange(1, q)
    theory = 1 + 2 * np.sum((1 - k / q) * phi**k)
    got = variance_ratio(r, q)["vr"]
    assert abs(got - theory) < 0.05, (got, theory)


def test_heteroskedasticity_does_not_break_the_statistic() -> None:
    """M2 is the robust version; a GARCH-like series with NO autocorrelation must still pass."""
    rng = np.random.default_rng(5)
    n = 200_000
    vol = np.exp(rng.normal(0, 0.5, n))
    vol = np.convolve(vol, np.ones(50) / 50, mode="same")  # clustered volatility
    r = rng.normal(0, 1, n) * vol
    res = variance_ratio(r, 8)
    assert abs(res["vr"] - 1.0) < 0.05, res
    assert abs(res["m2"]) < 4.0, res


def test_short_series_returns_nan_rather_than_a_number() -> None:
    assert np.isnan(variance_ratio(np.random.default_rng(0).normal(size=20), 16)["vr"])
    assert np.isnan(variance_ratio(np.random.default_rng(0).normal(size=1000), 1)["vr"])


def test_profile_covers_every_horizon() -> None:
    r = ar1(50_000, 0.1, seed=4)
    prof = vr_profile(r, qs=(2, 4, 8))
    assert set(prof) == {2, 4, 8}
    assert all(np.isfinite(v["vr"]) for v in prof.values())


def test_edge_points_is_zero_at_a_random_walk_and_positive_above_it() -> None:
    assert abs(vr_edge_points(1.0, 2.0, 16)) < 1e-12
    assert vr_edge_points(1.21, 2.0, 16) > 0
    assert vr_edge_points(0.81, 2.0, 16) < 0


def test_conditional_returns_never_splices_across_an_inactive_gap() -> None:
    """The trap this guards: joining returns from either side of a gap manufactures a move."""
    r = np.arange(100, dtype=float)
    active = np.zeros(100, bool)
    active[10:30] = True  # a 20-bar run
    active[50:55] = True  # a 5-bar run, too short for q=8
    out = conditional_returns(r, active, q=8)
    assert len(out) == 20
    assert out[0] == 10.0
    assert out[-1] == 29.0
    assert 50.0 not in out


def test_conditional_returns_is_empty_when_nothing_is_long_enough() -> None:
    r = np.arange(100, dtype=float)
    active = np.zeros(100, bool)
    active[::3] = True  # isolated bars, no run reaches q
    assert len(conditional_returns(r, active, q=8)) == 0


def test_segmented_vr_never_spans_a_boundary() -> None:
    """A huge jump placed exactly at a session join must not enter any q-period window."""
    rng = np.random.default_rng(11)
    r = rng.normal(0, 1, 6000)
    active = np.ones(6000, bool)
    active[2000:2010] = False  # a gap
    r[2000:2010] = 500.0  # an overnight jump the naive version would splice in
    seg = variance_ratio_segmented(r, active, 8)
    assert np.isfinite(seg["vr"])
    assert abs(seg["vr"] - 1.0) < 0.15, seg
    assert seg["n_segments"] == 2


def test_segmented_matches_plain_vr_when_there_are_no_gaps() -> None:
    r = ar1(60_000, 0.12, seed=12)
    plain = variance_ratio(r, 8)["vr"]
    seg = variance_ratio_segmented(r, np.ones(len(r), bool), 8)["vr"]
    assert abs(plain - seg) < 0.02, (plain, seg)


def test_segmented_counts_only_windows_inside_runs() -> None:
    r = np.zeros(100)
    active = np.zeros(100, bool)
    active[0:20] = True
    active[50:70] = True
    res = variance_ratio_segmented(r, active, 8)
    assert res["n_segments"] == 2
    assert res["n_windows"] == 2 * (20 - 8 + 1)
