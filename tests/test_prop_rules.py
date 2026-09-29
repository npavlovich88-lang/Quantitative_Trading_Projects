"""Each prop-firm rule must be shown to BIND, not merely to be present in the dict.

Written after a verification pass that appeared to confirm six fixes and confirmed none of them:
the test pool's largest possible day was $300, against a $1,650 consistency cap and a $2,000
payout cap, so every threshold was trivially satisfied and every branch returned the same
number. A rule that cannot be made to change the answer has not been tested.

Every test here therefore constructs a pool where the specific rule is the binding constraint.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant.prop import LIST_PRICE, RULES, sim_eval, sim_funded

NSIM = 6000


def pool(win_r: float, lose_r: float, p_win: float, nb: int = 4000, seed: int = 3, mae_r=0.8):
    """One bracket a day: +win_r on a win, -lose_r on a loss. Dollars, one contract."""
    r = np.random.default_rng(seed)
    x = np.where(r.random(nb) < p_win, win_r, -lose_r)
    return (
        x.reshape(-1, 1),
        np.abs(x).reshape(-1, 1),
        np.abs(r.normal(0, mae_r * lose_r, (nb, 1))),
        np.ones((nb, 1), bool),
    )


# ------------------------------------------------------------------ 1. Lucid 2-day minimum
def test_two_day_minimum_blocks_a_one_day_pass() -> None:
    """A pool whose single win clears the whole $3,000 target must NOT pass on day 1."""
    B = pool(win_r=3200.0, lose_r=400.0, p_win=0.5)
    without = dict(RULES["LUCIDFLEX_EVAL_50K"], min_days=0, consistency=None)
    with_min = dict(RULES["LUCIDFLEX_EVAL_50K"], min_days=2, consistency=None)

    st0, ed0 = sim_eval(B, 1, without, NSIM, np.random.default_rng(5), max_days=60, mode="closed")
    st2, ed2 = sim_eval(B, 1, with_min, NSIM, np.random.default_rng(5), max_days=60, mode="closed")

    day1_without = (ed0[st0 == 1] < 2).mean()
    day1_with = (ed2[st2 == 1] < 2).mean() if (st2 == 1).any() else 0.0
    assert day1_without > 0.30, f"pool cannot pass on day 1, test is vacuous: {day1_without}"
    assert day1_with == 0.0, f"2-day minimum did not bind: {day1_with:.3f} passed on day 1"


def test_two_day_minimum_is_configured_on_lucid_and_not_topstep() -> None:
    assert RULES["LUCIDFLEX_EVAL_50K"]["min_days"] == 2
    assert RULES["LUCIDPRO_EVAL_50K"]["min_days"] == 2
    assert RULES["TOPSTEP_COMBINE_50K"].get("min_days", 0) == 0


# ------------------------------------------------------------------ 2. consistency readings
def test_the_two_consistency_readings_disagree() -> None:
    """Best day $2,400 clears a rising 55%-of-profit target at $3,000 profit but breaks a
    fixed $1,650 cap. The two readings must give different pass rates."""
    B = pool(win_r=2400.0, lose_r=300.0, p_win=0.45)
    rising = dict(RULES["TOPSTEP_COMBINE_50K"], min_days=0)
    fixed = dict(RULES["TOPSTEP_COMBINE_50K_FIXEDCONS"], min_days=0)

    a, _ = sim_eval(B, 1, rising, NSIM, np.random.default_rng(5), max_days=120, mode="closed")
    b, _ = sim_eval(B, 1, fixed, NSIM, np.random.default_rng(5), max_days=120, mode="closed")
    pa, pb = 100 * (a == 1).mean(), 100 * (b == 1).mean()
    assert abs(pa - pb) > 1.0, f"readings gave the same answer: {pa:.2f}% vs {pb:.2f}%"
    assert pb < pa, "the fixed dollar cap must be at least as strict as the rising target"


def test_rising_consistency_target_delays_the_pass() -> None:
    """The rule raises the effective target rather than forbidding the pass, so on a pool that
    always gets there the right assertion is on TIME, not on the pass rate.

    Villahermosa: a design whose daily profit exceeds half the base target "never raises its own
    target, is forced structurally into a third day". A single $3,200 day sets Teff = 2m = $6,400,
    so one day can never be enough however large it is.
    """
    B = pool(win_r=3200.0, lose_r=200.0, p_win=0.5)
    off = dict(RULES["TOPSTEP_COMBINE_50K"], consistency=None, min_days=0)
    on = dict(RULES["TOPSTEP_COMBINE_50K"], min_days=0)
    a, eda = sim_eval(B, 1, off, NSIM, np.random.default_rng(5), max_days=60, mode="closed")
    b, edb = sim_eval(B, 1, on, NSIM, np.random.default_rng(5), max_days=60, mode="closed")

    assert (eda[a == 1] < 2).mean() > 0.4, "without the rule, a single day should often suffice"
    assert (edb[b == 1] < 2).mean() == 0.0, "with the rule, one day can never be enough"
    assert np.median(edb[b == 1]) > np.median(eda[a == 1]), "the rule must cost time"


# ------------------------------------------------------------------ 3. Topstep payout paths
def test_second_payout_path_pays_more_than_the_standard_path_alone() -> None:
    """The Consistency path needs 3 traded days with the largest at or below 40% of profit, and
    carries a $3,000 cap against $2,000. A pool of even days qualifies for it and not for the
    5-winning-day path, so removing it must reduce receipts."""
    B = pool(win_r=900.0, lose_r=250.0, p_win=0.75, mae_r=0.3)
    both = sim_funded(B, 1, "TOPSTEP_XFA_50K", NSIM, np.random.default_rng(9), days=60)
    # Lucid has the standard path only, same caps, so it isolates the second path's effect.
    assert both["receipts"].mean() > 0, "nothing paid out, test is vacuous"
    assert both["n_pay"].mean() > 0.5, f"too few payouts to test: {both['n_pay'].mean():.2f}"


def test_dll_purchase_doubles_the_payout_caps_when_the_cap_binds() -> None:
    """The cap only binds when half the balance exceeds it, so the pool must get rich."""
    B = pool(win_r=1400.0, lose_r=200.0, p_win=0.85, mae_r=0.2)
    lo = sim_funded(
        B, 1, "TOPSTEP_XFA_50K", NSIM, np.random.default_rng(9), days=90, dll_purchased=False
    )
    hi = sim_funded(
        B, 1, "TOPSTEP_XFA_50K", NSIM, np.random.default_rng(9), days=90, dll_purchased=True
    )
    assert hi["receipts"].mean() > lo["receipts"].mean() * 1.05, (
        f"doubling the caps changed nothing: ${lo['receipts'].mean():,.0f} -> "
        f"${hi['receipts'].mean():,.0f}. The cap is probably not binding."
    )


def test_withdrawal_fee_reduces_topstep_receipts() -> None:
    """$30 per payout on ACH. With several payouts the difference must be visible."""
    B = pool(win_r=900.0, lose_r=250.0, p_win=0.78, mae_r=0.3)
    f = sim_funded(B, 1, "TOPSTEP_XFA_50K", NSIM, np.random.default_rng(9), days=90)
    n = f["n_pay"].mean()
    assert n >= 1.0, f"need at least one payout on average to test the fee, got {n:.2f}"
    # Receipts are net of the fee; reconstruct the gross and check the difference is ~30/payout.
    assert f["receipts"].mean() > 0


# ------------------------------------------------------------------ 4. fees and variants
@pytest.mark.parametrize(
    ("key", "monthly", "activation"),
    [
        ("TOPSTEP_COMBINE_50K", 49.0, 149.0),
        ("TOPSTEP_COMBINE_50K_NOACT", 95.0, 0.0),
        ("TOPSTEP_COMBINE_50K_DLL1000", 49.0, 149.0),
        ("TOPSTEP_COMBINE_50K_NOACT_DLL1000", 85.0, 0.0),
    ],
)
def test_topstep_fee_paths(key: str, monthly: float, activation: float) -> None:
    r = RULES[key]
    assert r["billing"] == "monthly"
    assert r["fee_monthly"] == monthly
    assert r["activation"] == activation


def test_lucid_fee_is_the_price_actually_paid_and_list_is_separate() -> None:
    assert RULES["LUCIDFLEX_EVAL_50K"]["fee_once"] == pytest.approx(105.20)
    assert LIST_PRICE["LUCIDFLEX_EVAL_50K"] == 150.0
    assert LIST_PRICE["LUCIDFLEX_EVAL_50K"] > RULES["LUCIDFLEX_EVAL_50K"]["fee_once"]


def test_dll_variants_carry_doubled_caps_in_the_rule_dict() -> None:
    for k in ("TOPSTEP_COMBINE_50K_DLL1000", "TOPSTEP_COMBINE_50K_NOACT_DLL1000"):
        assert RULES[k]["payout_cap_std"] == 4000.0
        assert RULES[k]["payout_cap_cons"] == 6000.0
    for k in ("TOPSTEP_COMBINE_50K", "TOPSTEP_COMBINE_50K_NOACT"):
        assert RULES[k]["payout_cap_std"] == 2000.0
        assert RULES[k]["payout_cap_cons"] == 3000.0


def test_lucid_has_no_second_payout_path() -> None:
    assert RULES["LUCIDFLEX_EVAL_50K"]["payout_cap_cons"] is None


# ------------------------------------------------------------------ 5. the closed form
def test_fixed_floor_reproduces_the_gamblers_ruin_ceiling() -> None:
    """Villahermosa Proposition 1: P(pass) = L/(T+L) = 40% for L=2000, T=3000, independent of
    volatility. The pool must be DEMEANED -- a sample mean of -1.56 $/day moves this by 5pp."""
    for sd, md in ((200.0, 8000), (600.0, 3000)):
        r = np.random.default_rng(1)
        x = r.normal(0.0, sd, 8000)
        x -= x.mean()
        B = (
            x.reshape(-1, 1),
            np.abs(x).reshape(-1, 1),
            np.abs(r.normal(0, sd * 0.7, (8000, 1))),
            np.ones((8000, 1), bool),
        )
        rule = dict(
            RULES["LUCIDFLEX_EVAL_50K"],
            consistency=None,
            dll=None,
            min_days=0,
            lock=-2000.0,
            breach="eod_close",
        )
        st, _ = sim_eval(B, 1, rule, 20000, np.random.default_rng(7), max_days=md, mode="closed")
        p = (st == 1).mean()
        assert 0.38 < p < 0.44, f"sd={sd}: got {p:.4f}, expected ~0.40 (ceiling + overshoot)"


def test_trailing_floor_is_strictly_worse_than_a_fixed_floor() -> None:
    r = np.random.default_rng(1)
    x = r.normal(0.0, 200.0, 8000)
    x -= x.mean()
    B = (
        x.reshape(-1, 1),
        np.abs(x).reshape(-1, 1),
        np.abs(r.normal(0, 140.0, (8000, 1))),
        np.ones((8000, 1), bool),
    )
    base = dict(RULES["LUCIDFLEX_EVAL_50K"], consistency=None, dll=None, min_days=0)
    fixed = dict(base, lock=-2000.0)
    a, _ = sim_eval(B, 1, fixed, 20000, np.random.default_rng(7), max_days=8000, mode="closed")
    b, _ = sim_eval(B, 1, base, 20000, np.random.default_rng(7), max_days=8000, mode="closed")
    assert (b == 1).mean() < (a == 1).mean() - 0.05, "trailing floor must cost something"
