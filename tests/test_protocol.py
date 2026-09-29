"""The framework must reject a strategy that has no edge.

A validation protocol that passes noise is worse than none, because it launders noise into
confidence. So the headline test builds a strategy on a pure random walk -- where the true
answer is known to be "nothing here" -- and asserts the protocol says so.
"""

import numpy as np
import pytest

from quant import diagnostics as diag
from quant.protocol import Protocol
from quant.register import TrialRegister
from quant.splits import HoldoutLocked, Split


# ------------------------------------------------------------------ splits
def test_holdout_is_not_reachable_by_accident():
    s = Split.by_fraction(1000, 0.5, 0.25, name="t")
    assert s.train.stop == 500
    assert s.validate.start == 500
    with pytest.raises(HoldoutLocked, match="frozen"):
        _ = s.holdout
    # bounds are readable without spending it, so reports and plots stay honest
    assert s.holdout_bounds == (750, 1000)


def test_unlocking_requires_a_real_reason_and_is_logged(tmp_path):
    led = tmp_path / "ledger.jsonl"
    s = Split.by_fraction(1000, 0.5, 0.25, name="t", ledger=led)
    with pytest.raises(ValueError, match="real reason"):
        s.unlock_holdout("because")
    s.unlock_holdout("final evaluation of the frozen preset, ladder fully passed")
    assert s.holdout == slice(750, 1000)
    assert len(s.holdout_looks()) == 1
    assert not s.is_holdout_spent()

    again = Split.by_fraction(1000, 0.5, 0.25, name="t", ledger=led)
    again.unlock_holdout("second look at the same holdout, which should mark it spent")
    assert again.is_holdout_spent(), "a second look must mark the holdout as spent"


def test_embargo_creates_a_gap_between_segments():
    s = Split.by_fraction(1000, 0.5, 0.25, embargo=40, name="t")
    assert s.train.stop == 460
    assert s.validate.start == 500, "the gap is dropped, not reassigned"
    assert s.validate.stop == 710


# ------------------------------------------------------------------ register
def test_register_counts_unique_trials_not_reruns(tmp_path):
    r = TrialRegister(tmp_path / "trials.jsonl")
    for _ in range(3):
        r.record({"stop": 6, "tp": 2}, segment="train", metrics={"pf": 1.1})
    r.record({"stop": 3, "tp": 2}, segment="train", metrics={"pf": 1.2})
    assert r.n_trials("train") == 2, "re-running the same cell is not a new trial"


def test_register_matrix_is_column_aligned(tmp_path):
    r = TrialRegister(tmp_path / "trials.jsonl")
    rng = np.random.default_rng(0)
    for i in range(5):
        r.record({"k": i}, segment="train", metrics={"pf": 1.0}, pnl=rng.normal(size=200))
    r.record({"k": 99}, segment="train", metrics={"pf": 1.0}, pnl=rng.normal(size=50))
    m, cfgs = r.matrix("train")
    assert m.shape == (200, 5), "the odd-length trial must be dropped, not padded"
    assert len(cfgs) == 5


# ------------------------------------------------------------------ diagnostics
def test_luck_battery_flags_a_result_carried_by_a_few_trades():
    pnl = np.r_[np.full(200, -1.0), [400.0, 300.0]]
    res = diag.luck_battery(pnl, top_k=5)
    assert res["verdict"] == "fail"
    assert res["total"] > 0, "the raw backtest looks profitable"
    assert res["total_without_top_k"] < 0, "but only because of the top few trades"


def test_luck_battery_passes_a_broad_edge():
    rng = np.random.default_rng(1)
    res = diag.luck_battery(rng.normal(0.4, 1.0, 500), top_k=5)
    assert res["verdict"] == "pass"


def test_runs_test_detects_alternation():
    alternating = np.array([1.0, -1.0] * 60)
    assert diag.runs_test(alternating)["z"] > 2
    streaky = np.r_[np.ones(60), -np.ones(60)]
    assert diag.runs_test(streaky)["z"] < -2


def test_plateau_vs_spike():
    flat = {(a, b): 1.2 for a in range(5) for b in range(5)}
    assert diag.parameter_plateau(flat)["verdict"] == "pass"
    spike = {(a, b): 1.0 for a in range(5) for b in range(5)}
    spike[(2, 2)] = 3.0
    assert diag.parameter_plateau(spike)["verdict"] == "warn"


def test_coverage_finds_a_hole():
    days = np.r_[
        np.arange("2024-01-01", "2024-03-01", dtype="datetime64[D]"),
        np.arange("2024-09-01", "2024-10-01", dtype="datetime64[D]"),
    ]
    res = diag.data_coverage(days)
    assert res["n_gaps"] == 1
    assert res["largest_gap_days"] > 180
    assert res["verdict"] == "fail"


def test_regime_balance_flags_single_regime_edge():
    pnl = np.r_[np.full(100, 1.0), np.full(100, -1.0)]
    regime = np.r_[np.zeros(100), np.ones(100)]
    res = diag.regime_balance(pnl, regime)
    assert res["verdict"] == "fail"
    assert res["n_profitable_regimes"] == 1


def test_cost_sensitivity_finds_breakeven():
    rng = np.random.default_rng(2)
    base = rng.normal(3.0, 10.0, 400)
    res = diag.cost_sensitivity(lambda c: base - c, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert res["breakeven_cost_pts"] is not None
    assert 2.0 < res["breakeven_cost_pts"] < 5.0


# ------------------------------------------------------------------ end to end
def test_protocol_rejects_a_no_edge_strategy(tmp_path):
    """The headline test. Random-walk P&L, 40 configurations, no edge anywhere.

    The protocol must halt and return REJECTED. If this ever passes, the framework is
    laundering noise and nothing else it says can be trusted.
    """
    rng = np.random.default_rng(7)
    n = 2400
    split = Split.by_fraction(
        n, 0.5, 0.25, embargo=30, name="noise", ledger=tmp_path / "ledger.jsonl"
    )
    reg = TrialRegister(tmp_path / "trials.jsonl")
    for i in range(40):
        pnl = rng.normal(0.0, 1.0, n // 2)  # zero expectancy, by construction
        reg.record({"cfg": i}, segment="train", metrics={"pf": 1.0}, pnl=pnl)

    p = Protocol("noise strategy", tmp_path / "out", split, reg)
    p.rung_insample(metric=1.03, n_trades=120)
    p.rung_selection("train", S=8)
    # a p-value consistent with noise
    p.rung_mcpt(real=1.03, perms=list(rng.normal(1.02, 0.05, 200)), p_value=0.42)

    assert p.halted, "the protocol must stop once a rung fails"
    assert p.verdict() == "REJECTED"
    out = p.save()
    assert (out / "report.txt").exists()
    assert (out / "result.json").exists()
    report = (out / "report.txt").read_text(encoding="utf-8")
    assert "REJECTED" in report
    assert "has NOT been shown to have an edge" in report
    assert not split.is_holdout_spent(), "a rejected strategy must never have cost the holdout"
    assert (out / "figures").exists()
    assert list((out / "figures").glob("*.png")), "a rejection still gets its figures"


def test_protocol_reports_incomplete_when_rungs_are_skipped(tmp_path):
    split = Split.by_fraction(1000, 0.5, 0.25, name="x")
    reg = TrialRegister(tmp_path / "t.jsonl")
    p = Protocol("partial", tmp_path / "out2", split, reg)
    p.rung_insample(metric=1.5, n_trades=200)
    p.skip_rest("4. walk-forward")
    assert p.verdict() == "INCOMPLETE"


def test_describe_handles_a_segment_the_data_never_reaches():
    """MES found this: every built MES bar predates train_end, so validate and holdout are
    empty. An empty segment is a normal state (the data stops early), not a crash."""
    import numpy as np

    dates = np.arange("2020-01-01", "2020-06-01", dtype="datetime64[D]").astype("datetime64[ns]")
    s = Split.by_date(dates, train_end="2024-07-01", validate_end="2025-09-01", name="short")
    text = s.describe(dates)
    assert "EMPTY" in text
    assert "2020-01-01" in text, "the non-empty train segment must still report its real span"
