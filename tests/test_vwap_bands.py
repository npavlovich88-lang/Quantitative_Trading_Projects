"""The two things that can silently invalidate a VWAP band measurement.

1. CAUSALITY. A band computed from bars up to and including bar i is partly a function of bar
   i, so "bar i touched the band" becomes arithmetic rather than evidence. Checked against a
   slow explicit reference that sums only bars strictly before i.
2. A NULL THAT IS NOT 0.5. Race A asks whether price reaches the VWAP (k sigma away) before
   the next band out (1 sigma away). The near target wins most of the time on any path at all,
   so the test that matters is that a random walk also shows it. If a test suite only asserted
   pA > 0.5 on real data it would pass on noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.vwap_bands import (
    ANCHORS,
    anchor_ids,
    band_index,
    first_touches,
    race,
    vwap_bands,
)


def walk(n: int = 6000, seed: int = 0, sd: float = 3.0, freq: str = "5min") -> pd.DataFrame:
    r = np.random.default_rng(seed)
    dt = pd.date_range("2024-01-02 08:30", periods=n, freq=freq)
    c = 20000.0 + np.cumsum(r.normal(0.0, sd, n))
    w = np.abs(r.normal(0.0, sd, n))
    return pd.DataFrame(
        dict(
            dt=dt,
            open=c,
            high=c + w,
            low=c - w,
            close=c,
            volume=r.integers(50, 500, n).astype(float),
        )
    )


def test_bands_use_no_information_from_their_own_bar() -> None:
    """mu[i] and sigma[i] must equal the volume-weighted mean and dispersion of bars
    [anchor_start, i-1] -- never including bar i."""
    df = walk(1200)
    gid = anchor_ids(df["dt"], "ny")
    mu, sg = vwap_bands(df, gid, min_bars=5)
    tp = (df["high"].to_numpy() + df["low"].to_numpy() + df["close"].to_numpy()) / 3.0
    v = df["volume"].to_numpy(float)

    checked = 0
    for i in range(1, len(df)):
        if not np.isfinite(mu[i]):
            continue
        s = np.flatnonzero(gid == gid[i])[0]
        w, p = v[s:i], tp[s:i]  # STRICTLY before i
        m_ref = (w * p).sum() / w.sum()
        s_ref = np.sqrt(max((w * p * p).sum() / w.sum() - m_ref**2, 0.0))
        assert abs(mu[i] - m_ref) < 1e-6 * max(1.0, abs(m_ref)), f"mu leaked at {i}"
        assert abs(sg[i] - s_ref) < 1e-5 * max(1.0, s_ref), f"sigma leaked at {i}"
        checked += 1
    assert checked > 500, f"reference check was near-vacuous, only {checked} bars compared"


def test_every_anchor_produces_bands_and_the_reset_count_is_right() -> None:
    df = walk(6000)
    counts = {}
    for kind in ANCHORS:
        gid = anchor_ids(df["dt"], kind)
        _mu, sg = vwap_bands(df, gid)
        counts[kind] = len(np.unique(gid))
        assert np.isfinite(sg).mean() > 0.5, f"{kind}: bands mostly undefined"
        assert (sg[np.isfinite(sg)] > 0).all(), f"{kind}: non-positive sigma"
    # 6000 5m bars is about 20.8 days of 24h clock time
    assert counts["continuous"] == 1
    assert counts["ny"] > counts["weekly"] > counts["monthly"] >= counts["quarterly"]
    assert 18 <= counts["ny"] <= 23, counts["ny"]


def test_the_race_engine_reproduces_gamblers_ruin_on_a_driftless_walk() -> None:
    """The engine check, against a closed form, with the distances the race ACTUALLY has.

    The naive prediction is 1/(k+1): the VWAP sits k sigma from the touched level and the next
    band out sits 1 sigma from it. That is wrong, and the reason is worth keeping. The event
    fires when the bar LOW reaches the level, but the forward scan resumes on the NEXT bar, so
    the walk restarts from the touch bar CLOSE -- which on noise already sits up to 0.9 sigma
    back inside the band at k=4, purely from the size of the wick that got there.

    Using the real distances from that close, ruin = d_ext / (d_mean + d_ext) must track the
    measured pA. If it does, the machinery has nothing unaccounted for; if it drifts apart,
    something in the race is wrong and no result from it can be read.
    """
    for k in (1, 2, 3, 4):
        pa, ruin, tot = [], [], 0
        for seed in range(4):
            df = walk(40000, seed=100 + seed)
            gid = anchor_ids(df["dt"], "ny")
            mu, sg = vwap_bands(df, gid)
            hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
            ev = first_touches(mu, sg, hi, lo, gid, k, -1)
            ev = ev[(ev + 78) < len(df)]
            if len(ev) < 30:
                continue
            m, s = mu[ev], sg[ev]
            d_mean = ((m - cl[ev]) / s).mean()
            d_ext = ((cl[ev] - (m - (k + 1) * s)) / s).mean()
            r = race(df, mu, sg, ev, k, -1, horizon=78)
            pa.append(r["pA"])
            ruin.append(d_ext / (d_mean + d_ext))
            tot += r["nA"]
        assert tot > 200, f"k={k}: only {tot} resolved races, test is near-vacuous"
        assert abs(np.mean(pa) - np.mean(ruin)) < 0.05, (
            f"k={k}: measured pA={np.mean(pa):.3f} against ruin={np.mean(ruin):.3f} from the "
            "actual distances. The race engine and the geometry disagree."
        )


def test_the_wick_back_inside_the_band_grows_with_the_band() -> None:
    """On noise alone, the further out the band, the more the touch bar closes back inside it.
    That wick is what a chart reads as a rejection candle, and it is free."""
    over = {}
    for k in (1, 2, 3, 4):
        df = walk(40000, seed=100)
        gid = anchor_ids(df["dt"], "ny")
        mu, sg = vwap_bands(df, gid)
        ev = first_touches(
            mu, sg, df["high"].to_numpy(float), df["low"].to_numpy(float), gid, k, -1
        )
        cl = df["close"].to_numpy(float)
        over[k] = float(((cl[ev] - (mu[ev] - k * sg[ev])) / sg[ev]).mean())
    assert over[1] < over[2] < over[3] < over[4], over
    assert over[4] > 0.5, f"expected a large free wick at 4 sigma, got {over[4]:.2f}"


def test_race_c_is_much_rarer_than_race_a() -> None:
    """Traversing to the mirror band is a longer trip than reaching the mean, so pC must sit
    well below pA. If they were similar the levels would not be what they claim to be."""
    df = walk(40000, seed=11)
    gid = anchor_ids(df["dt"], "ny")
    mu, sg = vwap_bands(df, gid)
    ev = first_touches(mu, sg, df["high"].to_numpy(float), df["low"].to_numpy(float), gid, 2, -1)
    r = race(df, mu, sg, ev, 2, -1, horizon=78)
    assert r["nC"] > 50
    assert r["pC"] < r["pA"] - 0.1, f"pC={r['pC']:.3f} pA={r['pA']:.3f}"


def test_band_index_is_zero_where_bands_are_undefined() -> None:
    df = walk(400)
    gid = anchor_ids(df["dt"], "ny")
    mu, sg = vwap_bands(df, gid)
    z = band_index(mu, sg, df["close"].to_numpy(float))
    assert (z[~np.isfinite(sg)] == 0).all()
    assert np.abs(z).max() >= 1, "no bar ever left the first sigma, sample is unusable"


def test_touches_are_one_per_anchor_period() -> None:
    df = walk(6000)
    gid = anchor_ids(df["dt"], "ny")
    mu, sg = vwap_bands(df, gid)
    ev = first_touches(mu, sg, df["high"].to_numpy(float), df["low"].to_numpy(float), gid, 1, -1)
    assert len(ev) == len(np.unique(gid[ev])), "more than one event in some anchor period"
