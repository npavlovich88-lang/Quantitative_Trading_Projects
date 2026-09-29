#!/usr/bin/env python3
"""
Self-contained reference implementation (numpy + pandas only) of the
"Tenkan/SenkouA + ADX-regime" intraday futures strategy family and the
corrected prop-firm Monte Carlo. This file is the GROUND TRUTH for the
spec in STRATEGY_HANDOFF.md -- if prose and code disagree, the code wins.

Input CSV columns: ts (epoch seconds UTC = bar START time), open, high, low,
close, volume.  (A 'datetime'/'timestamp'/'time' column also works.)

Example (MNQ, 10-minute bars):
  python strategy_reference.py --csv MNQ_10m.csv --point-value 2 --cost-usd 5 \
      --tz America/Chicago --rth 08:30-15:00 --presets A B C --json out_mnq.json
"""

import argparse
import json

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ presets
PRESETS = {
    # A = the "negative RR" strategy: risk 6 ATR to make 2 ATR (RR 0.33)
    "A": dict(name="A_negRR_stop6_tp2", exit="fixed", stop_atr=6.0, tp_atr=2.0),
    # B = positive-RR "ride to the close": 3 ATR stop, no target, flat at RTH close
    "B": dict(name="B_posRR_stop3_notarget", exit="fixed", stop_atr=3.0, tp_atr=None),
    # C = positive-RR trend exit: exit when SuperTrend(10,3) flips against you (close-based)
    "C": dict(name="C_supertrend_flip", exit="st_flip", stop_atr=None, tp_atr=None),
    # D = 1:1 with the wide stop (comparison cell)
    "D": dict(name="D_posRR_stop6_tp6", exit="fixed", stop_atr=6.0, tp_atr=6.0),
}

# ------------------------------------------------- fixed strategy parameters
ER_N, ADX_N, CHOP_N = 14, 14, 14
ER_THRESH, ADX_THRESH, CHOP_THRESH, PERSIST = 0.35, 30.0, 45.0, 3
TEMA_LEN, EMA_LEN, VWMA_LEN = 14, 20, 12
TENKAN_N, SENKOU_N, DISP = 9, 52, 26
ATR_N = 14
ST_N, ST_MULT = 10, 3.0
TRAIN_FRAC = 0.70


# ------------------------------------------------------------------ loading
def epoch_seconds(dt):
    """Seconds since the epoch, independent of the datetime's storage resolution.

    `dt.astype("int64") // 10**9` is the obvious idiom and it is WRONG from pandas 3 onward.
    pandas 2 stored every datetime as nanoseconds, so dividing by 1e9 gave seconds. pandas 3
    infers the unit, and to_datetime / date_range now commonly produce microseconds -- the same
    expression then returns epoch-seconds/1000, silently, with no error and no warning. Measured
    on pandas 3.0.6: 1,641,189 where 1,641,189,600 was meant.

    Subtracting an epoch Timestamp and dividing by a Timedelta is resolution-agnostic, so it
    keeps working whatever unit the caller's data happens to carry.
    """
    return (dt - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)


def load_data(path, tz):
    df = pd.read_csv(path)
    if "ts" in df.columns:
        dt = pd.to_datetime(df["ts"], unit="s", utc=True)
    else:
        col = next(c for c in df.columns if c.lower() in ("datetime", "timestamp", "time", "date"))
        dt = pd.to_datetime(df[col], utc=True)
        df["ts"] = epoch_seconds(dt)
    df["dt"] = dt.dt.tz_convert(tz)
    df = df.sort_values("dt").reset_index(drop=True)
    return df


# --------------------------------------------------------------- indicators
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def tema(x, n):
    e1 = ema(x, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    return 3.0 * (e1 - e2) + e3


def mid_donchian(h, l, n):
    return (
        pd.Series(h).rolling(n).max().to_numpy() + pd.Series(l).rolling(n).min().to_numpy()
    ) / 2.0


def true_range(h, l, c):
    pc = pd.Series(c).shift(1).to_numpy()
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def wilder_smooth(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.nanmean(x[:n])
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def efficiency_ratio(c, n):
    net = np.abs(c - np.roll(c, n))
    net[:n] = np.nan
    total = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        er = net / total
    er[total == 0] = 0.0
    return er


def adx_dmi(h, l, c, n):
    up = h - np.roll(h, 1)
    dn = np.roll(l, 1) - l
    up[0] = dn[0] = 0
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(h, l, c)
    trs, pds, mds = wilder_smooth(tr, n), wilder_smooth(pdm, n), wilder_smooth(mdm, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = 100 * pds / trs
        mdi = 100 * mds / trs
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return wilder_smooth(dx, n), pdi, mdi


def choppiness(h, l, c, n):
    tr = true_range(h, l, c)
    s = pd.Series(tr).rolling(n).sum().to_numpy()
    rng = pd.Series(h).rolling(n).max().to_numpy() - pd.Series(l).rolling(n).min().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * np.log10(s / rng) / np.log10(n)


def confirm(raw, persist):
    return (pd.Series(raw.astype(float)).rolling(persist).min() == 1).to_numpy()


def supertrend(h, l, c, n=10, mult=3.0):
    """Project-specific SuperTrend (Wilder/RMA ATR, sticky bands). NOT guaranteed
    bit-identical to TradingView's ta.supertrend -- this file is the reference."""
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    hl2 = (h + l) / 2
    upper, lower = hl2 + mult * atr, hl2 - mult * atr
    st = np.full(len(c), np.nan)
    direction = np.ones(len(c))
    st[0] = lower[0]
    for i in range(1, len(c)):
        if c[i - 1] > st[i - 1]:
            st[i] = max(lower[i], st[i - 1]) if direction[i - 1] == 1 else lower[i]
            direction[i] = 1
        else:
            st[i] = min(upper[i], st[i - 1]) if direction[i - 1] == -1 else upper[i]
            direction[i] = -1
        if c[i] > st[i] and direction[i] == -1:
            direction[i] = 1
            st[i] = lower[i]
        elif c[i] < st[i] and direction[i] == 1:
            direction[i] = -1
            st[i] = upper[i]
    return st


def build_features(df, rth):
    o, h, l, c, v = (
        df[k].to_numpy().astype(float) for k in ("open", "high", "low", "close", "volume")
    )
    tr = true_range(h, l, c)
    # R-unit: PREVIOUS bar's Wilder ATR(14). shift(1) => no current-bar info.
    atr = pd.Series(tr).ewm(alpha=1 / ATR_N, adjust=False).mean().shift(1).to_numpy()
    tema_v, ema_v = tema(c, TEMA_LEN), ema(c, EMA_LEN)
    vwma_v = (
        pd.Series(c * v).rolling(VWMA_LEN).sum() / pd.Series(v).rolling(VWMA_LEN).sum()
    ).to_numpy()
    tenkan = mid_donchian(h, l, TENKAN_N)
    senkou_disp = pd.Series(mid_donchian(h, l, SENKOU_N)).shift(DISP).to_numpy()
    er = efficiency_ratio(c, ER_N)
    ser = er * np.sign(c - np.roll(c, ER_N))
    adx, pdi, mdi = adx_dmi(h, l, c, ADX_N)
    chop = choppiness(h, l, c, CHOP_N)
    with np.errstate(invalid="ignore"):
        raw_chop = (np.abs(ser) < 0.20) | (chop > 55) | (adx < (ADX_THRESH - 7))
        raw_tl = (er > ER_THRESH) & (adx > ADX_THRESH) & (pdi > mdi) & (chop < CHOP_THRESH)
        raw_ts = (er > ER_THRESH) & (adx > ADX_THRESH) & (mdi > pdi) & (chop < CHOP_THRESH)
    tl, ts_ = confirm(raw_tl, PERSIST), confirm(raw_ts, PERSIST)
    chop_state = confirm(raw_chop, PERSIST) & ~tl & ~ts_
    regime = np.where(
        tl, 1, np.where(ts_, -1, np.where(chop_state, 0, 2))
    )  # 1 up-trend, -1 down-trend, 0 chop, 2 neutral
    with np.errstate(invalid="ignore"):
        bull_tenkan = c > tenkan
        bear_stack = (tema_v < ema_v) & (ema_v < vwma_v)
        bull_senkou = c > senkou_disp
        bear_senkou = c < senkou_disp
    el = bull_tenkan & bull_senkou  # LONG  raw condition
    es = bear_stack & bear_senkou  # SHORT raw condition (note: different ingredients)
    st = supertrend(h, l, c, ST_N, ST_MULT)
    a, b = rth.split("-")
    lo = int(a[:2]) * 60 + int(a[3:5])
    hi = int(b[:2]) * 60 + int(b[3:5])
    mins = (df["dt"].dt.hour.to_numpy() * 60 + df["dt"].dt.minute.to_numpy()).astype(int)
    inwin = (mins >= lo) & (mins < hi)
    return dict(
        o=o,
        h=h,
        l=l,
        c=c,
        atr=atr,
        el=el,
        es=es,
        ok_long=(regime == 1),
        ok_short=(regime == -1),
        st_bull=(c > st),
        inwin=inwin,
        n=len(c),
        ts=df["ts"].to_numpy().astype(np.int64),
        split=int(len(c) * TRAIN_FRAC),
    )


# --------------------------------------------------------------- simulation
def simulate(F, exit_mode, stop_atr, tp_atr, cost_pts):
    """RTH-only entries, flat at the close of the last RTH bar. Entry = close of signal bar.
    Stop/target evaluated intrabar with gap-aware fills; if stop and target are both
    inside one bar the STOP is assumed first. No same-bar re-entry. Opposite signal
    while in a trade = exit at close and reverse."""
    o, h, l, c, atr = F["o"], F["h"], F["l"], F["c"], F["atr"]
    el, es, okl, oks, stb = F["el"], F["es"], F["ok_long"], F["ok_short"], F["st_bull"]
    n = F["n"]
    inwin = F["inwin"]
    entry_ok = inwin
    flat_after = inwin & np.r_[~inwin[1:], True]
    pos, ep, ei, ae, stop, tp, mfe, mae, last_exit = 0, 0.0, -1, 0.0, None, None, 0.0, 0.0, -1
    out = []

    def rec(i, xp, reason):
        out.append((ei, i, pos, ep, xp, ae, mfe, mae, reason))

    def open_pos(i, d):
        nonlocal pos, ep, ei, ae, stop, tp, mfe, mae
        pos, ep, ei, ae, mfe, mae = d, c[i], i, atr[i], 0.0, 0.0
        stop = (ep - d * stop_atr * ae) if stop_atr else None
        tp = (ep + d * tp_atr * ae) if tp_atr else None

    for i in range(n):
        a = atr[i]
        valid = not (np.isnan(a) or a <= 0)
        if pos != 0:
            if pos == 1:
                mfe, mae = max(mfe, h[i] - ep), max(mae, ep - l[i])
            else:
                mfe, mae = max(mfe, ep - l[i]), max(mae, h[i] - ep)
            xp, reason = None, ""
            if stop is not None or tp is not None:
                if pos == 1:
                    if stop is not None and o[i] <= stop:
                        xp, reason = o[i], "stop"
                    elif tp is not None and o[i] >= tp:
                        xp, reason = o[i], "tp"
                    elif stop is not None and l[i] <= stop:
                        xp, reason = stop, "stop"
                    elif tp is not None and h[i] >= tp:
                        xp, reason = tp, "tp"
                else:
                    if stop is not None and o[i] >= stop:
                        xp, reason = o[i], "stop"
                    elif tp is not None and o[i] <= tp:
                        xp, reason = o[i], "tp"
                    elif stop is not None and h[i] >= stop:
                        xp, reason = stop, "stop"
                    elif tp is not None and l[i] <= tp:
                        xp, reason = tp, "tp"
            if (
                xp is None
                and exit_mode == "st_flip"
                and ((pos == 1 and not stb[i]) or (pos == -1 and stb[i]))
            ):
                xp, reason = c[i], "flip"
            if xp is not None:
                rec(i, xp, reason)
                pos = 0
                last_exit = i
            elif (
                valid
                and entry_ok[i]
                and not flat_after[i]
                and ((pos == 1 and es[i] and oks[i]) or (pos == -1 and el[i] and okl[i]))
            ):
                rec(i, c[i], "reverse")
                d = -pos
                pos = 0
                last_exit = i
                open_pos(i, d)
                continue
            elif flat_after[i]:
                rec(i, c[i], "flat")
                pos = 0
                last_exit = i
        if pos == 0 and valid and entry_ok[i] and not flat_after[i] and i != last_exit:
            if el[i] and okl[i]:
                open_pos(i, 1)
            elif es[i] and oks[i]:
                open_pos(i, -1)
    if pos != 0:
        rec(n - 1, c[n - 1], "end")
    rows = []
    for ei_, xi, d, ep_, xp_, ae_, mfe_, mae_, reason in out:
        pts = (xp_ - ep_) * d - cost_pts
        rows.append(
            dict(
                entry_i=ei_,
                exit_i=xi,
                direction=d,
                pnl_pts=pts,
                atr_e=ae_,
                mfe=mfe_,
                mae=mae_,
                pnl_r=pts / ae_,
                reason=reason,
                hold=xi - ei_,
            )
        )
    return pd.DataFrame(rows)


def stats(t):
    if len(t) == 0:
        return dict(n=0, exp=np.nan, pf=np.nan, med=np.nan, win=np.nan)
    r = t["pnl_r"].clip(-15, 15)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return dict(
        n=len(t),
        exp=float(r.mean()),
        pf=float(wins / losses) if losses > 0 else np.nan,
        med=float(r.median()),
        win=float((t["pnl_pts"] > 0).mean()),
    )


def summarize(F, tr, point_value, days_test):
    a, b = tr[tr["entry_i"] < F["split"]], tr[tr["entry_i"] >= F["split"]]
    sa, sb = stats(a), stats(b)
    bar_h = np.median(np.diff(F["ts"])) / 3600.0
    return dict(
        train=sa,
        test=sb,
        test_per_week=len(b) / (days_test / 5.0) if days_test else np.nan,
        test_avg_hold_hours=float(b["hold"].mean() * bar_h) if len(b) else np.nan,
        test_worst_trade_usd_1c=float(b["pnl_pts"].min() * point_value) if len(b) else np.nan,
        test_median_1atr_usd_1c=float(np.nanmedian(b["atr_e"]) * point_value) if len(b) else np.nan,
    )


# ------------------------------------------------------------ Monte Carlo
def build_blocks(df, F, tr, point_value, part, day_roll_hours=7):
    """Trades -> padded [trading_days, K] arrays. Pool = EVERY real trading day of the slice
    (most have zero trades). Trade P&L in REAL dollars for ONE contract = points * point_value."""
    split = F["split"]
    day = (df["dt"] + pd.Timedelta(hours=day_roll_hours)).dt.strftime("%Y-%m-%d").to_numpy()
    if part == "test":
        sel, days = tr[tr["entry_i"] >= split], np.unique(day[split:])
    else:
        sel, days = tr[tr["entry_i"] < split], np.unique(day[:split])
    idx = {d: k for k, d in enumerate(days)}
    per = [[] for _ in days]
    for _, r in sel.sort_values("exit_i").iterrows():
        d = day[int(r["exit_i"])]
        if d in idx:
            per[idx[d]].append(
                (r["pnl_pts"] * point_value, r["mfe"] * point_value, r["mae"] * point_value)
            )
    K = max(1, max(len(x) for x in per))
    nb = len(days)
    pnl, mfe, mae = (np.zeros((nb, K)) for _ in range(3))
    val = np.zeros((nb, K), bool)
    for i, lst in enumerate(per):
        for k, (p, f, m) in enumerate(lst):
            pnl[i, k], mfe[i, k], mae[i, k], val[i, k] = p, f, m, True
    return pnl, mfe, mae, val


def firm_rules(target, kind):
    """Rules for a target tier are SCALED from a $50K/$3000-target tier (assumption, verify per firm)."""
    if (
        kind == "TOPSTEP"
    ):  # soft daily-loss pause, intraday-trailing DD locked at start balance, 55% consistency
        return dict(
            target=target,
            dd=round(target * 2 / 3),
            daily_loss=round(target / 3),
            daily_loss_hard=False,
            consistency=0.55,
            trail_type="intraday_locked",
        )
    return dict(
        target=target,
        dd=round(target * 2 / 3),
        daily_loss=round(target * 0.4),
        daily_loss_hard=True,
        consistency=0.50,
        trail_type="eod",
    )  # LUCID: hard daily loss, EOD trailing DD, 50% consistency


def mc_pass(B, nc, rules, n_sims, rng, max_days=130, conservative=True):
    """Vectorised day-block bootstrap. conservative=True also counts the worst open-trade
    excursion (MFE first, then MAE) against the trailing / daily limits; False = realized P&L only."""
    pnl, mfe, mae, valid = B
    nb, K = pnl.shape
    eq, peak, eodp = np.zeros(n_sims), np.zeros(n_sims), np.zeros(n_sims)
    status = np.zeros(n_sims, np.int8)
    pass_day = np.full(n_sims, np.inf)
    sum_pos, max_pos = np.zeros(n_sims), np.zeros(n_sims)
    dd, dl, hard = rules["dd"], rules["daily_loss"], rules["daily_loss_hard"]
    target, cons, trail = rules["target"], rules["consistency"], rules["trail_type"]
    for day in range(1, max_days + 1):
        if not (status == 0).any():
            break
        idx = rng.integers(0, nb, size=n_sims)
        dp = np.zeros(n_sims)
        eodp_prev = eodp.copy()
        for k in range(K):
            v = valid[idx, k] & (status == 0)
            if not hard:
                v &= ~(dp <= -dl)
            if not v.any():
                break
            p = pnl[idx, k] * nc
            fail = np.zeros(n_sims, bool)
            if conservative:
                fav, adv = mfe[idx, k] * nc, mae[idx, k] * nc
                worst = eq - adv
                if trail == "intraday_locked":
                    peak_c = np.where(v, np.maximum(peak, eq + fav), peak)
                    fail |= v & (worst <= np.minimum(peak_c - dd, 0.0))
                    peak = np.where(v, peak_c, peak)
                else:
                    fail |= v & (worst <= eodp_prev - dd)
                if hard:
                    fail |= v & ((dp - adv) <= -dl)
            eq = np.where(v, eq + p, eq)
            dp = np.where(v, dp + p, dp)
            if trail == "intraday_locked":
                peak = np.where(v, np.maximum(peak, eq), peak)
                fail |= v & (eq <= np.minimum(peak - dd, 0.0))
            if hard:
                fail |= v & (dp <= -dl)
            status = np.where(fail, 2, status)
        act = status == 0
        if trail == "eod":
            status = np.where(act & (eq <= eodp_prev - dd), 2, status)
            act = status == 0
            eodp = np.where(act, np.maximum(eodp, eq), eodp)
        posd = act & (dp > 0)
        sum_pos = np.where(posd, sum_pos + dp, sum_pos)
        max_pos = np.where(posd, np.maximum(max_pos, dp), max_pos)
        ratio = np.where(sum_pos > 0, max_pos / np.maximum(sum_pos, 1e-9), 1.0)
        passed = act & (eq >= target) & (ratio <= cons)
        status = np.where(passed, 1, status)
        pass_day = np.where(passed, day, pass_day)
    return pass_day


GRID_NC = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
HORIZONS = (15, 30, 60, 130)


def prop_eval(df, F, tr, point_value, target=3000):
    """Walk-forward protocol: pick contracts on the TRAIN slice (Topstep, conservative, best <=130-day
    pass rate), then evaluate that contract count on the untouched TEST slice."""
    Btr = build_blocks(df, F, tr, point_value, "train")
    Bte = build_blocks(df, F, tr, point_value, "test")
    rng = np.random.default_rng(11)
    best_nc, best_p = GRID_NC[0], -1.0
    for nc in GRID_NC:
        p = float(
            (
                mc_pass(Btr, nc, firm_rules(target, "TOPSTEP"), 2500, rng, conservative=True) <= 130
            ).mean()
        )
        if p > best_p:
            best_nc, best_p = nc, p
    res = dict(contracts_picked_on_train=best_nc, train_pass130_conservative=best_p)
    for firm in ("TOPSTEP", "LUCID"):
        rules = firm_rules(target, firm)
        rng = np.random.default_rng(5)
        for mode, cons in (("realized_only", False), ("with_open_trade_dd", True)):
            pd_ = mc_pass(Bte, best_nc, rules, 4000, rng, conservative=cons)
            res[f"{firm}_{mode}"] = {
                f"pass_within_{h}d": float((pd_ <= h).mean()) for h in HORIZONS
            }
    return res


# --------------------------------------------------------------------- main
def run(csv, point_value, cost_usd, tz, rth, presets, do_mc=True, target=3000):
    df = load_data(csv, tz)
    F = build_features(df, rth)
    cost_pts = cost_usd / point_value
    test_days = len(np.unique(df["dt"].dt.strftime("%Y-%m-%d").to_numpy()[F["split"] :]))
    out = dict(
        csv=csv,
        bars=F["n"],
        first=str(df["dt"].iloc[0]),
        last=str(df["dt"].iloc[-1]),
        point_value=point_value,
        cost_usd_roundturn=cost_usd,
        cost_points=cost_pts,
        presets={},
    )
    for key in presets:
        p = PRESETS[key]
        tr = simulate(F, p["exit"], p["stop_atr"], p["tp_atr"], cost_pts)
        s = summarize(F, tr, point_value, test_days)
        if do_mc and s["test"]["n"] >= 20:
            s["prop"] = prop_eval(df, F, tr, point_value, target)
        out["presets"][p["name"]] = s
        t, tt = s["train"], s["test"]
        print(
            f"{p['name']:26s} TRAIN n={t['n']:4d} exp={t['exp']:+.3f}R pf={t['pf']:.2f} | "
            f"TEST n={tt['n']:4d} exp={tt['exp']:+.3f}R pf={tt['pf']:.2f} win={tt['win'] * 100:.0f}% "
            f"{s['test_per_week']:.2f}/wk  1 ATR = ${s['test_median_1atr_usd_1c']:.0f}/contract"
        )
        if p["stop_atr"]:
            stop_usd = p["stop_atr"] * s["test_median_1atr_usd_1c"]
            dd = round(target * 2 / 3)
            flag = "  <-- TOO BIG for one contract; use a micro" if stop_usd > 0.35 * dd else ""
            print(
                f"    one full stop, 1 contract = ${stop_usd:.0f} = {stop_usd / dd * 100:.0f}% of the ${dd} max-loss limit{flag}"
            )
        if "prop" in s:
            pr = s["prop"]
            print(f"    contracts (picked on train) = {pr['contracts_picked_on_train']}")
            for firm in ("TOPSTEP", "LUCID"):
                a = pr[f"{firm}_realized_only"]["pass_within_130d"]
                b = pr[f"{firm}_with_open_trade_dd"]["pass_within_130d"]
                c30 = pr[f"{firm}_with_open_trade_dd"]["pass_within_30d"]
                print(
                    f"    {firm} ${target}: pass<=130d realized-only {a * 100:.0f}% | with open-trade DD {b * 100:.0f}% "
                    f"(<=30d {c30 * 100:.0f}%)"
                )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument(
        "--point-value", type=float, required=True, help="USD per 1.0 price point, ONE contract"
    )
    ap.add_argument(
        "--cost-usd",
        type=float,
        default=5.0,
        help="round-turn commission+slippage, USD, one contract",
    )
    ap.add_argument(
        "--tz", default="America/Chicago", help="exchange time zone used for the RTH window"
    )
    ap.add_argument(
        "--rth", default="08:30-15:00", help="regular trading hours HH:MM-HH:MM in --tz"
    )
    ap.add_argument("--presets", nargs="+", default=["A", "B", "C"], choices=list(PRESETS))
    ap.add_argument("--target", type=float, default=3000.0)
    ap.add_argument("--no-mc", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = run(a.csv, a.point_value, a.cost_usd, a.tz, a.rth, a.presets, not a.no_mc, a.target)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2, default=float)
