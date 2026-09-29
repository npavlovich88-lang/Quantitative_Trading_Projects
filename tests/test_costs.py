"""Costs must be decomposable and defensible, not a single unexamined number."""

import pathlib

import pytest

from quant.data_splits import DATA_ROOT

from quant.costs import (
    EXCHANGE_FEE_PER_SIDE,
    MEASURED_SPREAD_POINTS,
    NFA_FEE_PER_SIDE,
    POINT_VALUE,
    CostModel,
    measure_spread,
)


def test_commission_has_no_default():
    """Choosing a broker rate for the user would recreate the assumption this module removes."""
    with pytest.raises(TypeError):
        CostModel(symbol="MNQ")  # type: ignore[call-arg]


def test_components_sum_to_the_total():
    c = CostModel(symbol="MNQ", commission_per_side=0.50, slippage_ticks=1.0)
    assert c.round_turn_usd == pytest.approx(c.fees_usd + c.spread_usd + c.slippage_usd)
    assert c.round_turn_points == pytest.approx(c.round_turn_usd / POINT_VALUE["MNQ"])


def test_spread_is_crossed_once_per_round_turn():
    """Buy the ask, sell the bid: one crossing, not two. Double-counting it would roughly
    double the modelled cost on MES."""
    c = CostModel(
        symbol="MES", commission_per_side=0.0, exchange_fee_per_side=0.0, nfa_fee_per_side=0.0
    )
    assert c.spread_usd == pytest.approx(MEASURED_SPREAD_POINTS["MES"] * POINT_VALUE["MES"])
    assert c.round_turn_usd == pytest.approx(1.25)


def test_fees_are_charged_on_both_sides():
    c = CostModel(symbol="MNQ", commission_per_side=1.00)
    assert c.fees_usd == pytest.approx(2 * (1.00 + EXCHANGE_FEE_PER_SIDE + NFA_FEE_PER_SIDE))


def test_measured_spreads_are_whole_ticks_and_mes_is_tighter():
    """MES is a one-tick market, MNQ a two-tick market -- the substantive measurement."""
    for sym, s in MEASURED_SPREAD_POINTS.items():
        assert abs(s / 0.25 - round(s / 0.25)) < 1e-9, f"{sym} spread is not a whole tick"
    assert MEASURED_SPREAD_POINTS["MES"] < MEASURED_SPREAD_POINTS["MNQ"]


def test_the_old_assumption_was_conservative_not_optimistic():
    """The retired 2.5-pt MNQ number overstated cost at every plausible broker rate. That is
    the safe direction to have been wrong in: prior results were pessimistic, not flattering.
    If this ever flips, every earlier conclusion needs revisiting."""
    for comm in (0.25, 0.50, 1.00, 1.50):
        c = CostModel(symbol="MNQ", commission_per_side=comm)
        assert c.round_turn_points <= 2.5, (
            f"at ${comm}/side the measured cost is {c.round_turn_points:.3f} pts, which EXCEEDS "
            "the old 2.5-pt assumption -- earlier results were optimistic, not conservative"
        )


def test_slippage_is_extra_and_per_side():
    base = CostModel(symbol="MNQ", commission_per_side=0.50)
    slipped = CostModel(symbol="MNQ", commission_per_side=0.50, slippage_ticks=1.0)
    assert slipped.round_turn_usd - base.round_turn_usd == pytest.approx(2 * 0.25 * 2.0)


def test_breakdown_names_every_component():
    text = CostModel(symbol="MES", commission_per_side=0.75, slippage_ticks=0.5).breakdown()
    for token in ("commission", "exchange", "NFA", "spread", "slippage", "TOTAL"):
        assert token in text


@pytest.mark.slow
def test_measure_spread_reproduces_the_stored_constant():
    """Re-derive the measurement from raw ticks. Guards against the constant drifting away
    from the data it claims to come from."""
    p = (
        pathlib.Path(DATA_ROOT)
        / "Future_MES_T2"
        / "Future_MES_T2_20250200"
        / "20250211.csv"
    )
    if not p.exists():
        pytest.skip("raw tick file not available")
    r = measure_spread(str(p), rth_only=True, nrows=1_500_000)
    assert r["n"] > 10_000
    assert r["weekday"] not in ("Saturday", "Sunday"), (
        "spread measured on a weekend session is not representative of RTH trading"
    )
    assert r["median_points"] == pytest.approx(MEASURED_SPREAD_POINTS["MES"], abs=0.125)


# ------------------------------------------------------------------ prop firm rates
def test_fee_model_reproduces_topsteps_published_round_turn():
    """Topstep publishes $1.22 round turn, itemised NFA $0.02 + exchange $0.70 + commission
    $0.50. Our fee components must reconstruct that exactly, for both symbols. If this drifts,
    either a rate changed or the model is wrong -- both worth stopping for."""
    from quant.costs import CostModel

    for sym in ("MES", "MNQ"):
        c = CostModel.for_prop(sym, "topstep")
        assert c.fees_usd == pytest.approx(1.22, abs=0.005), (
            f"{sym}: fee model gives ${c.fees_usd:.2f}, Topstep publishes $1.22"
        )


def test_the_published_fee_is_not_the_cost_of_trading():
    """The broker bills fees; the market charges the spread. A backtest costed at the published
    fee alone understates real cost by roughly half on these contracts."""
    from quant.costs import CostModel

    for sym in ("MES", "MNQ"):
        c = CostModel.for_prop(sym, "topstep")
        assert c.round_turn_usd > c.fees_usd * 1.7, (
            f"{sym}: spread should be a substantial share of true cost"
        )
        assert c.spread_usd > 0


def test_average_sits_between_the_two_firms():
    from quant.costs import PROP_COMMISSION_PER_SIDE, CostModel

    lo = CostModel.for_prop("MNQ", "topstep").round_turn_usd
    hi = CostModel.for_prop("MNQ", "lucid").round_turn_usd
    avg = CostModel.for_prop("MNQ", "average").round_turn_usd
    assert lo < avg < hi
    assert PROP_COMMISSION_PER_SIDE["topstep"] < PROP_COMMISSION_PER_SIDE["lucid"]


def test_lucid_is_read_conservatively():
    """Lucid does not disclose whether its $0.50/side is all-in. We read it as commission-only,
    which is the higher-cost reading. Overstating cost is the safe direction."""
    from quant.costs import PROP_COMMISSION_PER_SIDE, CostModel

    assert PROP_COMMISSION_PER_SIDE["lucid"] == 0.50
    c = CostModel.for_prop("MES", "lucid")
    assert c.fees_usd > 1.22, "the conservative reading must cost more than Topstep's all-in"


# ------------------------------------------------------------------ slippage
def test_slippage_is_measured_and_size_dependent():
    """Not a constant. A flat slippage assumption is wrong in both directions at once:
    too high for one lot, far too low for twenty."""
    from quant.costs import CostModel

    one = CostModel.for_prop("MNQ", "average", contracts=1)
    twenty = CostModel.for_prop("MNQ", "average", contracts=20)
    assert one.slippage_usd == pytest.approx(0.0, abs=0.01)
    assert twenty.slippage_usd > 0.9
    assert twenty.round_turn_points > one.round_turn_points * 1.3


def test_mes_slippage_is_negligible_at_every_size_we_would_trade():
    """MES rests ~82 contracts at the touch. The hypothesis 'slippage is not an issue' is
    simply true here, and the test records that as a measured fact."""
    from quant.costs import CostModel

    for n in (1, 5, 10, 20, 32, 50):
        c = CostModel.for_prop("MES", "average", contracts=n)
        assert c.slippage_usd < 0.15, f"MES at {n} lots: ${c.slippage_usd:.2f}"


def test_mnq_slippage_becomes_material_above_five_lots():
    """MNQ rests only ~6 contracts at the touch, so the same hypothesis fails there. Our prop
    Monte Carlo swept contract counts to 32 -- this is the cost that was missing from it."""
    from quant.costs import CostModel

    small = CostModel.for_prop("MNQ", "average", contracts=5)
    big = CostModel.for_prop("MNQ", "average", contracts=32)
    assert small.slippage_usd < 0.15
    assert big.slippage_usd > 1.0
    assert big.round_turn_points > 1.6 * small.round_turn_points * 0.9


def test_slippage_is_charged_on_top_of_the_spread_not_instead_of_it():
    """Double-counting the first tick is the classic error: the spread already covers crossing
    the touch, slippage only covers walking past it."""
    from quant.costs import CostModel

    c = CostModel.for_prop("MNQ", "average", contracts=20)
    assert c.round_turn_usd == pytest.approx(c.fees_usd + c.spread_usd + c.slippage_usd)
    assert c.spread_usd > 0, "the spread must still be charged"
    assert c.slippage_usd > 0, "and the walk cost on top of it"


def test_explicit_slippage_overrides_the_measured_table():
    from quant.costs import CostModel

    c = CostModel(symbol="MES", commission_per_side=0.25, contracts=1, slippage_ticks=2.0)
    assert c.slippage_per_side_ticks == 2.0
    assert c.slippage_usd == pytest.approx(2 * 2.0 * 0.25 * 5.0)


def test_slippage_interpolates_and_extrapolates_monotonically():
    from quant.costs import slippage_ticks_for_size

    prev = -1.0
    for n in (1, 3, 7, 12, 25, 40, 60, 100):
        v = slippage_ticks_for_size("MNQ", n)
        assert v >= prev, f"slippage must not decrease with size (at {n})"
        prev = v
