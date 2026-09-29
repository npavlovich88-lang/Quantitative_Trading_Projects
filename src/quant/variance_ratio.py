"""Lo-MacKinlay variance ratio: is there serial dependence at all, and where?

WHY THIS ONE TEST ANSWERS A WHOLE FAMILY

The proposal behind hypothesis 11 is that we should stop trying to predict direction and instead
follow price until stopped out -- the edge coming from asymmetry rather than from forecasting.

That claim has a theorem under it. Under a driftless random walk, EVERY stopping rule has
expectancy exactly zero before costs and negative after them. This is the optional stopping
theorem, and no trail, state machine or acceptance rule escapes it. So a trend-following system
can only work if the series is NOT a random walk at the horizons its trail operates on.

The variance ratio measures precisely that:

    VR(q) = Var(q-period return) / (q * Var(1-period return))

    VR > 1   positively autocorrelated -- moves extend, a trail can capture
    VR = 1   random walk -- no stopping rule has positive expectancy
    VR < 1   mean reverting -- moves retrace, a trail bleeds on every whipsaw

It is the only component in the framework spec that carries its own significance test, and it
was marked MISSING by the spec audit.

WHY BOTH AN ANALYTIC AND A PERMUTATION NULL

Lo & MacKinlay (RFS 1988) give a heteroskedasticity-robust statistic, M2, that is asymptotically
standard normal under the random walk null. Financial returns are heteroskedastic, so the
homoskedastic version is not usable; M2 corrects for it, which is why it is the one implemented.

But this project has learned twice that analytic nulls are miscalibrated on overlapping intraday
data -- the Ichimoku run found a per-cell null 95th percentile of 1.377 rather than 1.960, and
hypothesis 7 found the opposite, a null best-cell median of 9.343. So M2 is reported for
reference and the grouped bar permutation is what decides. Where they disagree, the permutation
is right, because it is the only one that knows about this data's overlap structure.

ECONOMIC SIGNIFICANCE IS NOT STATISTICAL SIGNIFICANCE

VR can be distinguishable from 1 and still far too small to pay a round turn. `vr_edge_points`
converts a variance ratio into the extra points a q-bar move is expected to carry relative to a
random walk, so the number can be put next to the measured cost rather than admired on its own.
"""

from __future__ import annotations

import numpy as np


def variance_ratio(r: np.ndarray, q: int) -> dict:
    """VR(q) with the Lo-MacKinlay heteroskedasticity-robust statistic M2.

    `r` is a 1-period return series (log returns), NaNs removed by the caller. Uses overlapping
    q-period windows and the unbiased estimators from the 1988 paper.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4 * q or q < 2:
        return dict(vr=np.nan, m2=np.nan, n=n, q=q)

    mu = r.mean()
    dev = r - mu
    # 1-period variance, unbiased
    var_a = (dev**2).sum() / (n - 1)

    # q-period variance from OVERLAPPING windows, with the paper's bias correction
    cs = np.concatenate([[0.0], np.cumsum(r)])
    qsum = cs[q:] - cs[:-q]  # every overlapping q-period return
    m = q * (n - q + 1) * (1 - q / n)
    var_c = ((qsum - q * mu) ** 2).sum() / m
    if var_a <= 0 or not np.isfinite(var_c):
        return dict(vr=np.nan, m2=np.nan, n=n, q=q)
    vr = var_c / var_a

    # heteroskedasticity-robust variance of (VR - 1): the delta_j / theta construction
    d2 = dev**2
    denom = d2.sum() ** 2
    theta = 0.0
    for j in range(1, q):
        num = (d2[j:] * d2[:-j]).sum()
        delta_j = num / denom if denom > 0 else np.nan
        theta += ((2.0 * (q - j) / q) ** 2) * delta_j
    # M2 = (VR - 1) / sqrt(theta). NO sqrt(n) factor: delta_j already carries the 1/n, since
    # its denominator is a SQUARED sum. Including sqrt(n) as well inflated the statistic by
    # exactly that factor and turned a textbook random walk into M2 = -104.
    m2 = (vr - 1.0) / np.sqrt(theta) if theta > 0 else np.nan
    return dict(vr=float(vr), m2=float(m2), n=n, q=q)


def vr_edge_points(vr: float, sigma_1: float, q: int) -> float:
    """Extra points a q-bar move carries relative to a random walk, given VR(q).

    Under a random walk the expected absolute q-bar move scales as sigma * sqrt(q). A variance
    ratio of vr scales the q-period standard deviation by sqrt(vr), so the excess travel is

        sigma * sqrt(q) * (sqrt(vr) - 1)

    in the same price units as sigma. This is the number to put beside the round-turn cost. It is
    an upper bound on what a perfect trail could extract from the serial dependence alone, not an
    expectancy -- a real system captures a fraction of it and pays costs on every flip.
    """
    if not np.isfinite(vr) or vr <= 0:
        return np.nan
    return float(sigma_1 * np.sqrt(q) * (np.sqrt(vr) - 1.0))


def vr_profile(r: np.ndarray, qs=(2, 4, 8, 16, 32, 64)) -> dict[int, dict]:
    """VR across several horizons at once. The SHAPE matters more than any single value.

    A series that is mean-reverting at short q and trending at long q is a different animal from
    one that is flat everywhere, and only the profile shows which.
    """
    return {q: variance_ratio(r, q) for q in qs}


def conditional_returns(r: np.ndarray, active: np.ndarray, q: int) -> np.ndarray:
    """1-period returns restricted to windows lying ENTIRELY inside an active regime.

    This is what makes the conditional test honest. Computing VR over a gate-active subsample
    naively would splice together returns from either side of an inactive stretch, manufacturing
    a q-period "move" out of two unrelated episodes. Only runs of at least q consecutive active
    bars contribute, and the windows never straddle a gap.
    """
    r = np.asarray(r, dtype=float)
    a = np.asarray(active, dtype=bool) & np.isfinite(r)
    if not a.any():
        return np.array([])
    edges = np.flatnonzero(np.diff(np.r_[False, a, False]))
    out = []
    for s, e in zip(edges[0::2], edges[1::2], strict=True):
        if e - s >= q:
            out.append(r[s:e])
    return np.concatenate(out) if out else np.array([])


def segment_bounds(active: np.ndarray, q: int) -> list[tuple[int, int]]:
    """(start, stop) of every contiguous active run long enough to hold a q-period window."""
    a = np.asarray(active, dtype=bool)
    if not a.any():
        return []
    e = np.flatnonzero(np.diff(np.r_[False, a, False]))
    return [(s, t) for s, t in zip(e[0::2], e[1::2], strict=True) if t - s > q]


def variance_ratio_segmented(r: np.ndarray, active: np.ndarray, q: int) -> dict:
    """VR(q) with q-period windows confined to contiguous runs. The correct intraday version.

    WHY THE NAIVE VERSION IS WRONG, found by reading its own output

    Restricting to RTH and then handing the surviving returns to `variance_ratio` looks right and
    is not. The estimator forms OVERLAPPING q-period sums across whatever array it is given, so a
    window happily spans the join between two sessions -- splicing 17:00 to 08:30 as though no
    time passed. At 15m, RTH is 26 bars, so nearly every q=64 window straddles two boundaries and
    the overnight gap is silently deleted from the series.

    `conditional_returns` has the same defect: it drops runs shorter than q and then concatenates
    the survivors, which only moves the joins around.

    Here the 1-period variance pools every active return, and the q-period variance pools only
    windows lying entirely inside one run. Nothing crosses a boundary.
    """
    r = np.asarray(r, dtype=float)
    segs = segment_bounds(np.asarray(active, dtype=bool) & np.isfinite(r), q)
    if not segs or q < 2:
        return dict(vr=np.nan, m2=np.nan, n=0, q=q, n_windows=0, n_segments=0)

    pooled = np.concatenate([r[s:t] for s, t in segs])
    n = len(pooled)
    if n < 4 * q:
        return dict(vr=np.nan, m2=np.nan, n=n, q=q, n_windows=0, n_segments=len(segs))
    mu = pooled.mean()
    dev = pooled - mu
    var_a = (dev**2).sum() / (n - 1)

    qsums = []
    for s, t in segs:
        x = r[s:t]
        cs = np.concatenate([[0.0], np.cumsum(x)])
        qsums.append(cs[q:] - cs[:-q])
    qsum = np.concatenate(qsums)
    nw = len(qsum)
    if var_a <= 0 or nw < 2:
        return dict(vr=np.nan, m2=np.nan, n=n, q=q, n_windows=nw, n_segments=len(segs))
    var_c = ((qsum - q * mu) ** 2).sum() / (q * nw)
    vr = var_c / var_a

    d2 = dev**2
    denom = d2.sum() ** 2
    theta = 0.0
    for j in range(1, q):
        theta += ((2.0 * (q - j) / q) ** 2) * ((d2[j:] * d2[:-j]).sum() / denom)
    m2 = (vr - 1.0) / np.sqrt(theta) if theta > 0 else np.nan
    return dict(vr=float(vr), m2=float(m2), n=n, q=q, n_windows=nw, n_segments=len(segs))
