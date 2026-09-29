"""The declared splits are a contract. These tests are what makes it one.

If a boundary moves, one of these fails and the diff shows exactly what changed. A split you
can quietly edit is not a pre-registration.
"""

import itertools
import pathlib

import numpy as np
import pytest

from quant import strategy as S
from quant.data_splits import (
    BARS_BUILT,
    DATA,
    MNQ_HOLE,
    RAW_SPAN,
    SESSION_BARS,
    TRAIN_END,
    VALIDATE_END,
    make_split,
)
from quant.splits import HoldoutLocked


def test_boundaries_are_exactly_these_and_shared_by_both_symbols():
    """Locks the dates. One rule, both symbols -- there is no per-symbol boundary to diverge."""
    assert TRAIN_END == "2024-07-01"
    assert VALIDATE_END == "2025-09-01"


def test_both_symbols_get_identical_boundaries():
    """The point of the design: cross-market agreement is only meaningful if the windows match.

    Built on synthetic date arrays so this holds regardless of which bars exist on disk.
    """
    dates = np.arange("2020-01-01", "2026-07-01", dtype="datetime64[D]").astype("datetime64[ns]")
    a = make_split("MNQ", "10m", dates)
    b = make_split("MES", "10m", dates)
    assert (a.train_end, a.validate_end) == (b.train_end, b.validate_end), (
        "identical input dates must produce identical boundaries for both symbols"
    )


def test_recorded_raw_spans_are_the_measured_ones():
    assert RAW_SPAN["MNQ"] == ("2021-09-01", "2026-08-31", 1308)
    assert RAW_SPAN["MES"] == ("2020-01-01", "2026-06-28", 2169)
    assert MNQ_HOLE == ("2023-05-31", "2024-06-30", 396)


def test_both_symbols_have_built_bars():
    """MES built 2026-09-22. The distinction this file got wrong once was thin BARS vs thin
    DATA; both are now built and both pass the integrity battery."""
    assert BARS_BUILT["MNQ"] is True
    assert BARS_BUILT["MES"] is True
    assert RAW_SPAN["MES"][2] > RAW_SPAN["MNQ"][2], (
        "MES has MORE raw days than MNQ -- it is not the weaker dataset"
    )


@pytest.mark.slow
def test_mes_split_populates_and_holdout_is_untouched():
    """MES now has real data behind every segment, and its holdout has never been opened."""
    path = pathlib.Path(DATA["MES"].format(tf="10m"))
    if not path.exists():
        pytest.skip("no MES data")
    df = S.load_data(str(path), "America/Chicago")
    d = df["dt"].dt.tz_localize(None).to_numpy()
    sp = make_split("MES", "10m", d)
    assert sp.train.stop - sp.train.start > 100_000, "MES train should hold 4+ years"
    assert sp.validate.stop - sp.validate.start > 20_000
    h0, h1 = sp.holdout_bounds
    assert h1 - h0 > 20_000
    assert len(sp.holdout_looks()) == 0, "the MES holdout has never been opened"
    with pytest.raises(HoldoutLocked):
        _ = sp.holdout


def test_embargo_covers_a_full_session_at_every_timeframe():
    """The embargo must be >= the longest holding period. Flat at every RTH close, so one RTH
    session (390 minutes) is the bound."""
    for tf, bars in SESSION_BARS.items():
        minutes = int(tf.rstrip("m"))
        assert bars * minutes >= 390 - minutes, f"{tf}: {bars} bars does not span an RTH session"


@pytest.mark.slow
@pytest.mark.parametrize("tf", ["5m", "10m", "30m", "60m"])
def test_mnq_boundary_lands_in_the_hole_and_holdout_is_locked(tf):
    path = pathlib.Path(DATA["MNQ"].format(tf=tf))
    if not path.exists():
        pytest.skip(f"no data at {path}")
    df = S.load_data(str(path), "America/Chicago")
    d = df["dt"].dt.tz_localize(None).to_numpy()
    sp = make_split("MNQ", tf, d)

    last_train = np.datetime64(d[sp.train.stop - 1], "D")
    first_val = np.datetime64(d[sp.validate.start], "D")
    assert last_train <= np.datetime64("2023-05-31")
    assert first_val >= np.datetime64("2024-06-30")
    assert (first_val - last_train).astype(int) > 300, "expected the 396-day hole at the boundary"

    with pytest.raises(HoldoutLocked):
        _ = sp.holdout
    assert sp.train.stop > sp.train.start
    assert sp.validate.stop > sp.validate.start
    h0, h1 = sp.holdout_bounds
    assert h1 > h0


@pytest.mark.slow
def test_mnq_holdout_records_its_prior_exposure():
    """The holdout was traversed by the 2026-09-21 walk-forward smoke test. That is logged, so
    the count starts at 1. Asserting 0 would be asserting a comfortable fiction."""
    path = pathlib.Path(DATA["MNQ"].format(tf="10m"))
    if not path.exists():
        pytest.skip("no data")
    df = S.load_data(str(path), "America/Chicago")
    sp = make_split("MNQ", "10m", df["dt"].dt.tz_localize(None).to_numpy())
    looks = sp.holdout_looks()
    assert len(looks) >= 1, "the prior exposure must be on the record"
    assert any("PRIOR EXPOSURE" in e["reason"] for e in looks)
    assert not sp.is_holdout_spent(), "one prior look is exposure; a second makes it spent"


# ------------------------------------------------------------------ walk-forward
def test_wf_folds_never_overlap_and_tile_the_validate_segment():
    """Non-overlapping test blocks covering validate exactly. An overlap would double-count
    out-of-sample bars and inflate the sample size behind the estimate."""
    from quant.splits import Split, WalkForwardPlan

    sp = Split.by_fraction(12000, 0.5, 0.35, embargo=20, name="wf")
    folds = WalkForwardPlan(train_lookback=2000, step=500).folds(sp)
    assert len(folds) > 3
    for a, b in itertools.pairwise(folds):
        assert a.test.stop == b.test.start, "test blocks must tile without gaps or overlap"
    assert folds[0].test.start == sp.validate.start
    assert folds[-1].test.stop == sp.validate.stop


def test_wf_optimisation_window_never_touches_its_own_test_block():
    """The whole point of walk-forward: optimise strictly on the past, with an embargo."""
    from quant.splits import Split, WalkForwardPlan

    sp = Split.by_fraction(12000, 0.5, 0.35, embargo=25, name="wf")
    for f in WalkForwardPlan(train_lookback=2000, step=500).folds(sp):
        assert f.opt.stop <= f.test.start - sp.embargo, (
            f"fold {f.i}: optimisation window ends at {f.opt.stop} but its test block starts "
            f"at {f.test.start} with embargo {sp.embargo}"
        )


def test_wf_never_optimises_on_the_holdout():
    from quant.splits import Split, WalkForwardPlan

    sp = Split.by_fraction(12000, 0.5, 0.35, embargo=20, name="wf")
    h0, _ = sp.holdout_bounds
    for f in WalkForwardPlan(train_lookback=2000, step=500).folds(sp):
        assert f.opt.stop <= h0, "an optimisation window reached into the holdout"
        assert f.test.stop <= h0, "a test block reached into the holdout"


def test_anchored_plan_expands_while_rolling_plan_does_not():
    from quant.splits import Split, WalkForwardPlan

    sp = Split.by_fraction(12000, 0.5, 0.35, embargo=0, name="wf")
    roll = WalkForwardPlan(train_lookback=2000, step=500).folds(sp)
    anch = WalkForwardPlan(train_lookback=2000, step=500, anchored=True).folds(sp)
    assert roll[0].n_opt == roll[-1].n_opt, "a rolling window keeps a fixed lookback"
    assert anch[-1].n_opt > anch[0].n_opt, "an anchored window expands"
    assert all(f.opt.start == 0 for f in anch)


def test_bars_per_month_is_measured_not_assumed():
    """These are FULL SESSION files. An RTH-derived constant is ~3.5x too small, which once
    produced 43 folds of nine days each labelled 'monthly'."""
    import numpy as np

    from quant.data_splits import bars_per_month

    # 23h/day of 10-minute bars across three whole months
    dates = np.arange(
        "2024-01-01T00:00", "2024-04-01T00:00", np.timedelta64(10, "m"), dtype="datetime64[m]"
    )
    n = bars_per_month(dates)
    assert 3800 < n < 4600, f"expected ~4300 full-session 10m bars/month, measured {n}"
