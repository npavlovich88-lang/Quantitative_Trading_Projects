#!/usr/bin/env python3
"""
Prop-firm evaluation + funded-account Monte Carlo built on the firms' PUBLISHED rules
(researched 2026-09-21; see PROP_FIRM_HANDOFF.md for sources and confidence tags).
Standalone: numpy + pandas only. Input is a "block pool":

    B = (pnl, mfe, mae, valid)     each shaped [n_days, K]
        pnl/mfe/mae = dollars for ONE contract, per trade, left-packed per day
        (mfe/mae = best / worst open-trade excursion, positive numbers)
        valid = bool mask.  n_days must include EVERY real trading day (most are empty).

Every simulated day is drawn with replacement from the pool (day-block bootstrap).
Equity is measured relative to the starting balance (start = 0).

Two accounting modes for real-time-monitored limits:
    mode="closed": limits checked only when a trade closes (optimistic)
    mode="open"  : also checks the trade's worst open excursion (pessimistic)
"""

import numpy as np

# ----------------------------------------------------------------------------
# Rule sets ($50K tier only -- the only tier verified from official pages)
# ----------------------------------------------------------------------------
RULES = {
    # Topstep Trading Combine $50K [OFFICIAL help.topstep.com]
    "TOPSTEP_COMBINE_50K": dict(
        target=3000.0,
        mll=2000.0,
        lock=0.0,  # floor trails EOD balance highs, locks at starting balance
        breach="realtime",  # monitored in real time incl. unrealized P&L
        # NOTE ON THE CONSISTENCY READING. The operator's wording is "best day below 55% of the
        # profit target", which reads as a FIXED $1,650 cap. What is implemented is best day <=
        # 55% of TOTAL net profit, i.e. an effective target that RISES: Teff = max(T, m/c).
        # Villahermosa (SSRN 7445798) documents the second form with dashboard evidence -- a
        # $644.50 best day displaying a $1,289.00 target against a $1,250 base. The two differ
        # materially, so CONSISTENCY_FIXED_TARGET below carries the other reading for sensitivity.
        consistency=0.55,
        dll=None,  # optional add-on ($1K/$2K/$3K), soft; off by default
        min_days=0,
        fee_monthly=49.0,
        reset_fee=49.0,
        activation=149.0,
        billing="monthly",
        payout_cap_std=2000.0,
        payout_cap_cons=3000.0,
        payout_fee=30.0,  # ACH; Aeropay is free, so this is the conservative branch
    ),
    # Same Combine on the No Activation Fee path: $95/month, nothing due on passing.
    "TOPSTEP_COMBINE_50K_NOACT": dict(
        target=3000.0,
        mll=2000.0,
        lock=0.0,
        breach="realtime",
        consistency=0.55,
        dll=None,
        min_days=0,
        fee_monthly=95.0,
        reset_fee=95.0,
        activation=0.0,
        billing="monthly",
        payout_cap_std=2000.0,
        payout_cap_cons=3000.0,
        payout_fee=30.0,
    ),
    # Adding the DLL at purchase DOUBLES the funded payout caps under a limited-time offer,
    # and takes $10/month off the No Activation Fee Combine. Both are modelled.
    "TOPSTEP_COMBINE_50K_DLL1000": dict(
        target=3000.0,
        mll=2000.0,
        lock=0.0,
        breach="realtime",
        consistency=0.55,
        dll=1000.0,
        min_days=0,
        fee_monthly=49.0,
        reset_fee=49.0,
        activation=149.0,
        billing="monthly",
        payout_cap_std=4000.0,
        payout_cap_cons=6000.0,
        payout_fee=30.0,
    ),
    "TOPSTEP_COMBINE_50K_NOACT_DLL1000": dict(
        target=3000.0,
        mll=2000.0,
        lock=0.0,
        breach="realtime",
        consistency=0.55,
        dll=1000.0,
        min_days=0,
        fee_monthly=85.0,  # $95 less the $10 DLL discount
        reset_fee=85.0,
        activation=0.0,
        billing="monthly",
        payout_cap_std=4000.0,
        payout_cap_cons=6000.0,
        payout_fee=30.0,
    ),
    # LucidFlex evaluation $50K [OFFICIAL support.lucidtrading.com]
    "LUCIDFLEX_EVAL_50K": dict(
        target=3000.0,
        mll=2000.0,
        lock=100.0,  # trails highest CLOSING balance, locks at start+$100
        breach="eod_close",  # official page silent on intraday; third-party: EOD only [ASSUMED]
        consistency=0.50,
        dll=None,
        min_days=2,  # Lucid requires a 2-day minimum in the evaluation
        fee_once=105.20,  # price actually paid via the MATCH code; see _LIST for list price
        reset_fee=105.0,
        billing="once",
        payout_cap_std=2000.0,
        payout_cap_cons=None,  # Lucid has no second payout path
        payout_fee=0.0,
    ),
    # LucidPro evaluation $50K [OFFICIAL]
    "LUCIDPRO_EVAL_50K": dict(
        target=3000.0,
        mll=2000.0,
        lock=100.0,
        breach="eod_close",
        consistency=None,
        dll=1200.0,  # soft: no more trades until next session
        min_days=2,
        fee_once=185.0,
        reset_fee=185.0,
        billing="once",
        payout_cap_std=2000.0,
        payout_cap_cons=None,
        payout_fee=0.0,
    ),  # third-party list price
}
# Sensitivity: same rules but an intraday breach check on Lucid (if the real behaviour is stricter)
RULES["LUCIDFLEX_EVAL_50K_INTRADAY"] = dict(RULES["LUCIDFLEX_EVAL_50K"], breach="realtime")
RULES["LUCIDPRO_EVAL_50K_INTRADAY"] = dict(RULES["LUCIDPRO_EVAL_50K"], breach="realtime")

# The other reading of Topstep's consistency rule: a FIXED cap at 55% of the BASE target
# ($1,650) rather than an effective target that rises. Run both; they are not equivalent.
RULES["TOPSTEP_COMBINE_50K_FIXEDCONS"] = dict(
    RULES["TOPSTEP_COMBINE_50K"], consistency=None, consistency_fixed=0.55 * 3000.0
)

# List prices, for the EV-at-list-price column the papers report alongside EV-at-fee-paid.
# Lim (SSRN 7184138) finds 29 of 31 contracts negative at list, so the distinction matters.
LIST_PRICE = {
    "LUCIDFLEX_EVAL_50K": 150.0,  # MATCH code gives 30% off; list is not shown in page text
    "LUCIDPRO_EVAL_50K": 264.0,
}


def sim_eval(B, nc, rule, n_sims, rng, max_days=260, mode="open"):
    """Returns (status, end_day). status: 1 pass, 2 fail, 0 unresolved after max_days."""
    pnl, _mfe, mae, valid = B
    nb, K = pnl.shape
    mll, lock, target = rule["mll"], rule["lock"], rule["target"]
    cons, dll = rule["consistency"], rule["dll"]
    cons_fixed = rule.get("consistency_fixed")  # the alternative reading: a fixed dollar cap
    min_days = rule.get("min_days", 0)  # Lucid requires 2 days minimum
    realtime = rule["breach"] == "realtime"
    eq = np.zeros(n_sims)
    maxeod = np.zeros(n_sims)
    floor = np.full(n_sims, -mll)
    best_day = np.zeros(n_sims)
    status = np.zeros(n_sims, np.int8)
    end_day = np.full(n_sims, np.inf)
    for day in range(1, max_days + 1):
        if not (status == 0).any():
            break
        idx = rng.integers(0, nb, size=n_sims)
        dp = np.zeros(n_sims)
        stopped = np.zeros(n_sims, bool)
        for k in range(K):
            v = valid[idx, k] & (status == 0)
            if dll is not None:
                v &= ~stopped
            if not v.any():
                break
            p = pnl[idx, k] * nc
            fail = np.zeros(n_sims, bool)
            if realtime and mode == "open":
                fail |= v & (eq - mae[idx, k] * nc <= floor)
            eq = np.where(v, eq + p, eq)
            dp = np.where(v, dp + p, dp)
            if realtime:
                fail |= v & (eq <= floor)
            status = np.where(fail, 2, status)
            end_day = np.where(fail & np.isinf(end_day), day, end_day)
            if dll is not None:
                stopped |= dp <= -dll
        act = status == 0
        if not realtime:
            f = act & (eq <= floor)
            status = np.where(f, 2, status)
            end_day = np.where(f & np.isinf(end_day), day, end_day)
            act = status == 0
        best_day = np.where(act, np.maximum(best_day, dp), best_day)
        maxeod = np.where(act, np.maximum(maxeod, eq), maxeod)
        floor = np.where(act, np.minimum(maxeod - mll, lock), floor)
        ok = eq >= target
        if cons is not None:
            ok &= best_day <= cons * eq  # equivalently Teff = max(T, best_day / cons)
        if cons_fixed is not None:
            ok &= best_day <= cons_fixed  # the fixed-dollar reading
        if day < min_days:
            ok &= False  # a 2-day minimum cannot be passed on day 1 however large the gain
        passed = act & ok
        status = np.where(passed, 1, status)
        end_day = np.where(passed & np.isinf(end_day), day, end_day)
    return status, end_day


# ----------------------------------------------------------------------------
# Funded phase (for EV).  Returns dict of arrays.
# ----------------------------------------------------------------------------
def sim_funded(
    B,
    nc,
    kind,
    n_sims,
    rng,
    days=126,
    mode="open",
    first10k_full=False,
    dll_purchased=False,
    withdraw_at=0.0,
):
    """kind: 'TOPSTEP_XFA_50K' or 'LUCIDFLEX_FUNDED_50K'.

    Scaling plans are ignored: both allow >= 20 micros from day one, and this project trades
    2-8 micros. `dll_purchased` doubles Topstep's payout caps, which is the limited-time offer
    attached to buying the Combine with a daily loss limit.
    """
    pnl, _mfe, mae, valid = B
    nb, K = pnl.shape
    if (
        kind == "TOPSTEP_XFA_50K"
    ):  # [OFFICIAL] MLL -2000 trailing EOD, locks at 0; resets to 0 permanently after a payout
        mll, lock, realtime = 2000.0, 0.0, True
        win_day, min_pay, split = 150.0, 125.0, 0.90
        # TWO payout paths, and the second one has the HIGHER cap. Modelling only the first
        # understated Topstep's funded-stage value.
        #   Standard    : 5 winning days of $150+          cap $2,000
        #   Consistency : 3 days traded, largest <= 40%    cap $3,000
        # Both caps DOUBLE to $4,000 / $6,000 if the DLL was added at Combine purchase.
        cap_std = 4000.0 if dll_purchased else 2000.0
        cap_cons = 6000.0 if dll_purchased else 3000.0
        cons_share, min_cons_days = 0.40, 3
        payout_fee = 30.0  # ACH; Aeropay is free, so this is the conservative branch
    else:  # LUCIDFLEX funded [OFFICIAL]: EOD trail, locks at +100; 5 days>=150, min 500, max 50% up to 2000
        mll, lock, realtime = 2000.0, 100.0, False
        win_day, min_pay, split = 150.0, 500.0, 0.90
        cap_std, cap_cons = 2000.0, None  # Lucid has no second payout path
        cons_share, min_cons_days = None, None
        payout_fee = 0.0
    eq = np.zeros(n_sims)
    maxeod = np.zeros(n_sims)
    floor = np.full(n_sims, -mll)
    alive = np.ones(n_sims, bool)
    cnt = np.zeros(n_sims, int)
    eq_last = np.zeros(n_sims)
    paid_any = np.zeros(n_sims, bool)
    receipts = np.zeros(n_sims)
    gross_paid = np.zeros(n_sims)
    n_pay = np.zeros(n_sims, int)
    n_capped = np.zeros(n_sims, int)
    n_traded = np.zeros(n_sims, int)
    best_cyc = np.zeros(n_sims)
    bust_day = np.full(n_sims, np.inf)
    first_pay_day = np.full(n_sims, np.inf)
    for day in range(1, days + 1):
        if not alive.any():
            break
        idx = rng.integers(0, nb, size=n_sims)
        dp = np.zeros(n_sims)
        for k in range(K):
            v = valid[idx, k] & alive
            if not v.any():
                break
            p = pnl[idx, k] * nc
            fail = np.zeros(n_sims, bool)
            if realtime and mode == "open":
                fail |= v & (eq - mae[idx, k] * nc <= floor)
            eq = np.where(v, eq + p, eq)
            dp = np.where(v, dp + p, dp)
            if realtime:
                fail |= v & (eq <= floor)
            alive &= ~fail
            bust_day = np.where(fail & np.isinf(bust_day), day, bust_day)
        if not realtime:
            f = alive & (eq <= floor)
            alive &= ~f
            bust_day = np.where(f & np.isinf(bust_day), day, bust_day)
        act = alive
        maxeod = np.where(act, np.maximum(maxeod, eq), maxeod)
        floor = np.where(act, np.where(paid_any, lock, np.minimum(maxeod - mll, lock)), floor)
        cnt = np.where(act & (dp >= win_day), cnt + 1, cnt)
        traded = np.where(act & (dp != 0.0), n_traded + 1, n_traded)
        n_traded = traded
        best_cyc = np.where(act, np.maximum(best_cyc, dp), best_cyc)
        # Path 1, Standard: five winning days in the cycle.
        e1 = act & (cnt >= 5)
        # Path 2, Consistency: three days traded with the largest at or below 40% of profit.
        if cap_cons is None:
            e2 = np.zeros(n_sims, bool)
            cap_now = np.full(n_sims, cap_std)
        else:
            e2 = act & (n_traded >= min_cons_days) & (eq > 0) & (best_cyc <= cons_share * eq)
            cap_now = np.where(e2, cap_cons, cap_std)  # take the better path when both qualify
        elig = (e1 | e2) & (eq > eq_last) & (eq > 0)
        gross = np.where(elig, np.minimum(0.5 * eq, cap_now), 0.0)
        do = elig & (gross >= min_pay) & (gross >= withdraw_at * cap_now)
        if kind == "TOPSTEP_XFA_50K" and first10k_full:
            full = np.maximum(0.0, np.minimum(gross, 10000.0 - gross_paid))
            got = full + (gross - full) * split
        else:
            got = gross * split
        got = np.maximum(got - payout_fee, 0.0)  # withdrawal fee, charged per payout
        # A payout is CAPPED when the firm's ceiling bound rather than the 50%-of-balance rule,
        # i.e. the balance was at least twice the cap. That is the "max payout" a trader means,
        # and it is a far higher bar than being merely eligible: on the 50K it needs a $12,000
        # balance to draw the $6,000 Consistency cap.
        capped = do & (0.5 * eq >= cap_now - 1e-9)
        n_capped = np.where(capped, n_capped + 1, n_capped)
        eq = np.where(do, eq - gross, eq)
        receipts = np.where(do, receipts + got, receipts)
        gross_paid = np.where(do, gross_paid + gross, gross_paid)
        n_pay = np.where(do, n_pay + 1, n_pay)
        first_pay_day = np.where(do & np.isinf(first_pay_day), day, first_pay_day)
        paid_any |= do
        eq_last = np.where(do, eq, eq_last)
        cnt = np.where(do, 0, cnt)
        n_traded = np.where(do, 0, n_traded)  # both payout paths count per CYCLE
        best_cyc = np.where(do, 0.0, best_cyc)
        floor = np.where(do, lock, floor)
    return dict(
        receipts=receipts,
        n_pay=n_pay,
        n_capped=n_capped,
        bust_day=bust_day,
        alive=alive,
        first_pay_day=first_pay_day,
    )


# ----------------------------------------------------------------------------
# EV assembly
# ----------------------------------------------------------------------------
def campaign_economics(status, end_day, rule, receipts_mean, activation_on_pass=True):
    """EV of 'keep buying evaluations until one passes, then trade the funded account for the
    horizon used to compute receipts_mean'.  All numbers in USD."""
    res = status != 0
    n_res = res.sum()
    if n_res == 0 or (status == 1).sum() == 0:
        return dict(
            p_pass_resolved=0.0,
            expected_attempts=np.inf,
            campaign_cost=np.inf,
            ev_campaign=-np.inf,
            ev_single=np.nan,
            months_pass=np.nan,
            months_fail=np.nan,
            p_unresolved=1 - n_res / len(status),
        )
    p = (status == 1).sum() / n_res
    ed = np.where(np.isinf(end_day), 260, end_day)
    if rule["billing"] == "monthly":
        months = np.maximum(1, np.ceil(ed / 21.0))
        m_p = months[status == 1].mean()
        m_f = months[status == 2].mean() if (status == 2).any() else 0.0
        cost_attempt_pass = rule["fee_monthly"] * m_p + (
            rule["activation"] if activation_on_pass else 0.0
        )
        cost_attempt_fail = rule["fee_monthly"] * m_f + rule["reset_fee"]
        cost_camp = cost_attempt_pass + (1 - p) / p * cost_attempt_fail
        single_cost = rule["fee_monthly"] * (p * m_p + (1 - p) * m_f)
    else:
        m_p = m_f = 0.0
        cost_camp = rule["fee_once"] / p  # every attempt buys a new evaluation
        single_cost = rule["fee_once"]
    ev_single = p * receipts_mean - single_cost  # buy ONE evaluation, walk away if it fails
    return dict(
        p_pass_resolved=float(p),
        expected_attempts=float(1 / p),
        campaign_cost=float(cost_camp),
        ev_campaign=float(receipts_mean - cost_camp),
        ev_single=float(ev_single),
        months_pass=float(m_p),
        months_fail=float(m_f),
        median_days_to_pass=float(np.median(ed[status == 1])),
        p_unresolved=float(1 - n_res / len(status)),
    )


def zero_edge(B):
    """Same trade-size distribution, zero expectancy: subtract the pooled mean P&L per trade."""
    pnl, mfe, mae, valid = B
    mean = pnl[valid].mean()
    out = pnl.copy()
    out[valid] = out[valid] - mean
    return out, mfe, mae, valid
