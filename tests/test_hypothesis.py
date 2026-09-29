"""Pre-registration must be impossible to fake retroactively."""

import pytest

from quant.hypothesis import Hypothesis, PreRegistrationError

GOOD = dict(
    name="test_h",
    instrument="MES 10m",
    session="08:30-15:00 America/Chicago",
    setup="price trades below the prior session's low and reclaims it within 30 minutes, "
    "with the reclaim bar closing above the prior low",
    direction="long",
    exit_rule="2xATR(14) stop, 2xATR target, flat at 15:00 CT",
    execution="enter at the close of the reclaim bar",
    costs="CostModel.for_prop('MES','average',contracts=1)",
    invalidation="abandon if in-sample MCPT p >= 0.01 or fewer than 30 trades in train",
    metric="profit factor on per-bar P&L, train segment",
)


def test_every_required_field_is_mandatory():
    for field in Hypothesis.REQUIRED:
        bad = dict(GOOD)
        bad[field] = ""
        with pytest.raises(PreRegistrationError, match=r"missing|too short"):
            Hypothesis(**bad)


def test_vague_fields_are_rejected():
    """'RSI below 30 looks bullish' must not pass as a setup."""
    with pytest.raises(PreRegistrationError, match="too short"):
        Hypothesis(**{**GOOD, "setup": "RSI below 30"})
    with pytest.raises(PreRegistrationError, match="too short"):
        Hypothesis(**{**GOOD, "invalidation": "if it fails"})


def test_direction_is_constrained():
    with pytest.raises(PreRegistrationError, match="long/short/both"):
        Hypothesis(**{**GOOD, "direction": "up"})


def test_hash_changes_when_any_field_changes():
    a = Hypothesis(**GOOD)
    for field in ("setup", "exit_rule", "session", "metric"):
        b = Hypothesis(**{**GOOD, field: GOOD[field] + " (edited)"})
        assert a.hash != b.hash, f"editing {field} must change the hash"


def test_registering_twice_is_a_noop_but_an_edit_is_recorded(tmp_path):
    reg = tmp_path / "hypotheses.jsonl"
    h = Hypothesis(**GOOD)
    h.register(reg)
    h.register(reg)
    assert len(Hypothesis.registry(reg)) == 1, "identical re-registration must not duplicate"

    edited = Hypothesis(**{**GOOD, "exit_rule": "3xATR stop, 1xATR target, flat at 15:00 CT"})
    edited.register(reg)
    revs = Hypothesis.revisions(reg, "test_h")
    assert len(revs) == 2, "a changed hypothesis must appear as a separate, visible entry"
    assert revs[1]["revision_of"] == [revs[0]["hash"]]
    assert revs[0]["hash"] != revs[1]["hash"]


def test_registry_preserves_registration_time(tmp_path):
    reg = tmp_path / "h.jsonl"
    Hypothesis(**GOOD).register(reg)
    e = Hypothesis.registry(reg)[0]
    assert "registered" in e
    assert len(e["registered"]) >= 19


def test_describe_lists_every_required_field():
    text = Hypothesis(**GOOD).describe()
    for f in Hypothesis.REQUIRED:
        if f != "name":
            assert f in text
