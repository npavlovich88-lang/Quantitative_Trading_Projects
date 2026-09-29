"""The verdict block must refuse to be misleading.

The failure mode it exists to prevent is a "best configuration" printed next to a "no edge"
conclusion, which reads as a recommendation to anyone skimming the tail of a log.
"""

from __future__ import annotations

import pytest

from quant.verdict import EDGE, INCONCLUSIVE, NO_EDGE, SIGNAL_ONLY, Verdict, VerdictError

REASON = "The in-sample permutation test returned p = 0.13, an ordinary draw from the null."


def test_no_edge_refuses_to_name_a_config() -> None:
    with pytest.raises(VerdictError, match="only be named"):
        Verdict(NO_EDGE, "x", REASON, config="MES 5m ema8/ema21")


def test_signal_only_refuses_to_name_a_config() -> None:
    with pytest.raises(VerdictError, match="only be named"):
        Verdict(SIGNAL_ONLY, "x", REASON, config="MNQ 30m vwap/ema50")


def test_edge_must_name_a_config_and_its_expectancy() -> None:
    with pytest.raises(VerdictError, match="name the configuration"):
        Verdict(EDGE, "x", REASON)
    with pytest.raises(VerdictError, match="AFTER costs"):
        Verdict(EDGE, "x", REASON, config="MES 30m mid(120)")


def test_reason_must_be_a_sentence() -> None:
    with pytest.raises(VerdictError, match="sentence, not a label"):
        Verdict.no_edge("x", "failed")


def test_unknown_state_is_rejected() -> None:
    with pytest.raises(VerdictError, match="state must be"):
        Verdict("MAYBE", "x", REASON)


def test_from_gates_takes_the_first_failure_as_binding() -> None:
    gates = [
        ("1. pre-registered", "pass", "hash abc"),
        ("3. in-sample quality", "fail", "PF 0.98 on 40 trades"),
        ("5. luck (in-sample MCPT)", "fail", "p = 0.41"),
    ]
    v = Verdict.from_gates("ema8/21 MES 5m", gates)
    assert v.state == NO_EDGE
    assert "3. in-sample quality" in v.reason
    assert "PF 0.98" in v.reason
    assert any("5. luck" in e for e in v.evidence)
    assert "p = 0.41" not in v.reason  # downstream of a dead signal, so not binding


def test_from_gates_refuses_when_nothing_failed() -> None:
    with pytest.raises(VerdictError, match="must name a configuration"):
        Verdict.from_gates("x", [("1. pre-registered", "pass", "")])


@pytest.mark.parametrize("state", [NO_EDGE, SIGNAL_ONLY, INCONCLUSIVE])
def test_render_says_there_is_nothing_to_trade(state: str) -> None:
    v = Verdict(state, "some test", REASON, evidence=("2,711 live cells",))
    txt = v.render()
    assert f"EDGE:  {state}" in txt
    assert "CONFIG TO TRADE" in txt
    assert "None" in txt
    assert "BINDING REASON" in txt
    assert "2,711 live cells" in txt


def test_render_of_an_edge_names_config_and_net() -> None:
    v = Verdict.edge(
        "some test",
        config="MNQ 10m tema55/ema21, stop 3 ATR / tp 6 ATR",
        net="+2.10 pts per trade after 1.23 pts round turn",
        reason="Slow lines detect the 30m trend persistence that survives the round turn.",
        evidence=("walk-forward MCPT p = 0.004", "PBO 0.031"),
        caveat="A regime with no 30m persistence removes it entirely.",
    )
    txt = v.render()
    assert "EDGE:  EDGE" in txt
    assert "tema55/ema21" in txt
    assert "after 1.23 pts round turn" in txt
    assert "WHY IT WORKS" in txt
    assert "WHAT WOULD BREAK IT" in txt


def test_as_dict_carries_a_plain_boolean() -> None:
    assert Verdict.no_edge("x", REASON).as_dict()["edge"] is False
    v = Verdict.edge("x", "cfg", "+1 pt net", REASON)
    assert v.as_dict()["edge"] is True


def test_long_reason_wraps_without_losing_words() -> None:
    long = " ".join(["permutation"] * 40)
    txt = Verdict.no_edge("x", long).render()
    assert txt.count("permutation") == 40
    assert max(len(line) for line in txt.splitlines()) <= 92
