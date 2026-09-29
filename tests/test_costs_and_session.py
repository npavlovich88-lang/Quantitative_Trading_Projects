"""Costs are in the P&L, and the session rules hold.

Both are things a backtest can silently stop doing during a refactor while still producing a
plausible equity curve.
"""

import numpy as np
import pytest

from quant import strategy as S

from .test_no_lookahead import synthetic_bars


@pytest.fixture(scope="module")
def feats():
    return S.build_features(synthetic_bars(), "08:30-15:00")


def test_every_trade_is_charged_exactly_one_round_turn(feats):
    """Doubling the cost must reduce total P&L by exactly n_trades x the increment -- no more
    (double charging), no less (a path that skips the fee)."""
    c0, c1 = 2.5, 5.0
    a = S.simulate(feats, "fixed", 6.0, 2.0, c0)
    b = S.simulate(feats, "fixed", 6.0, 2.0, c1)
    assert len(a) == len(b), "changing the cost changed which trades were taken"
    delta = a["pnl_pts"].sum() - b["pnl_pts"].sum()
    assert np.isclose(delta, len(a) * (c1 - c0)), (
        f"expected {len(a)} x {c1 - c0} = {len(a) * (c1 - c0)} points of extra cost, got {delta}"
    )


def test_zero_cost_is_strictly_better(feats):
    """Sanity: a free backtest always beats a costed one. If this fails the sign is wrong."""
    free = S.simulate(feats, "fixed", 6.0, 2.0, 0.0)
    paid = S.simulate(feats, "fixed", 6.0, 2.0, 2.5)
    assert free["pnl_pts"].sum() > paid["pnl_pts"].sum()


def test_no_entry_outside_rth(feats):
    """Entries are RTH-only. An entry outside the window is a spec violation, not a preference."""
    tr = S.simulate(feats, "fixed", 6.0, 2.0, 2.5)
    inwin = np.asarray(feats["inwin"])
    outside = tr[~inwin[tr["entry_i"].to_numpy().astype(int)]]
    assert len(outside) == 0, f"{len(outside)} trades entered outside RTH"


def test_flat_at_session_close(feats):
    """No position is carried past the last RTH bar of a day."""
    tr = S.simulate(feats, "fixed", 6.0, 2.0, 2.5)
    inwin = np.asarray(feats["inwin"])
    last_rth = inwin & np.r_[~inwin[1:], True]
    for ei, xi in zip(tr["entry_i"], tr["exit_i"], strict=True):
        span = last_rth[int(ei) : int(xi)]
        assert not span.any(), (
            f"trade {ei}->{xi} was held across a session close at "
            f"{int(ei) + int(np.flatnonzero(span)[0])}"
        )


def test_entry_price_is_the_signal_bar_close(feats):
    """Entry at the close of the signal bar is the stated rule. If entry ever became the NEXT
    bar's open, or the same bar's low, this catches it."""
    tr = S.simulate(feats, "fixed", 6.0, 2.0, 0.0)
    c = np.asarray(feats["c"])
    same_bar = tr[tr["entry_i"] == tr["exit_i"]]
    for _, row in same_bar.iterrows():
        i = int(row["entry_i"])
        # P&L on a same-bar trade is direction * (exit - close_at_entry); recover the exit and
        # check it lies inside the bar.
        exit_px = c[i] + row["pnl_pts"] * row["direction"]
        assert feats["l"][i] - 1e-6 <= exit_px <= feats["h"][i] + 1e-6, (
            f"same-bar exit {exit_px:.2f} is outside bar {i} range "
            f"[{feats['l'][i]:.2f}, {feats['h'][i]:.2f}]"
        )


def test_stop_is_never_more_favourable_than_the_stop_price(feats):
    """A stop fill must be at the stop or worse (gap), never better. Better-than-stop fills are
    the classic way a backtest invents money."""
    stop_atr = 6.0
    tr = S.simulate(feats, "fixed", stop_atr, 2.0, 0.0)
    atr = np.asarray(feats["atr"])
    stops = tr[tr["reason"] == "stop"]
    for _, row in stops.iterrows():
        i, d = int(row["entry_i"]), int(row["direction"])
        loss_pts = row["pnl_pts"]
        expected_worst = -stop_atr * atr[i]
        assert loss_pts <= expected_worst + 1e-6, (
            f"stop fill at {loss_pts:.2f} pts is better than the {expected_worst:.2f} pt stop "
            f"(trade entered bar {i}, direction {d})"
        )
