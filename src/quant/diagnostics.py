"""The bias battery: everything that must be ruled out before a backtest is a finding.

Each function answers one question and returns a dict with a `verdict` of "pass", "warn" or
"fail", plus the numbers behind it. `run_all` bundles them. Nothing here is optional -- the
point of the module is that you cannot report a result without also reporting these.

The biases, and where each is caught:

    look-ahead            tests/test_no_lookahead.py (causality proof, mutation-tested)
    data-snooping / luck  mcpt.py + luck_battery() here
    selection bias        register.TrialRegister + pbo.cscv
    overfitting           pbo.cscv + parameter_plateau()
    sample bias           data_coverage(), regime_balance()
    survivorship / roll   roll_integrity()
    cost blindness        cost_sensitivity()

The one that is NOT here, deliberately: there is no statistic that tells you a strategy will
work. These only tell you when it definitely has not been shown to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _verdict(fail: bool, warn: bool) -> str:
    return "fail" if fail else ("warn" if warn else "pass")


# --------------------------------------------------------------------- luck
def luck_battery(pnl: np.ndarray, *, top_k: int = 5) -> dict:
    """Is the result carried by a handful of trades?

    Trend-following legitimately has positive skew -- neurotrader's Donchian makes its money on
    a fat right tail and a 36% win rate, and that is the design working, not a flaw. So this is
    not a pass/fail on skew. It is a measure of CONCENTRATION: if removing the best few trades
    flips the sign, the backtest is a statement about those few trades, and a 5-year sample
    contains far too few of them to be a statement about the future.
    """
    p = np.asarray(pnl, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return dict(verdict="fail", reason="no trades")
    order = np.argsort(p)[::-1]
    total = p.sum()
    top = p[order[:top_k]].sum()
    without_top = total - top
    wins, losses = p[p > 0], p[p < 0]
    gross = wins.sum()
    pf = gross / -losses.sum() if losses.size and losses.sum() < 0 else np.inf

    # Share of NET total is unstable: when total is near zero it explodes (a "120% of total"
    # that means nothing). Share of GROSS PROFIT is always in [0, 1] and says the thing we
    # actually care about -- how much of the winning came from the top few trades.
    share = float(top / gross) if gross > 0 else np.nan

    fail = total > 0 and without_top <= 0
    warn = bool(np.isfinite(share) and share > 0.5)
    return dict(
        verdict=_verdict(fail, warn),
        n_trades=int(p.size),
        total=float(total),
        top_k=top_k,
        top_k_share_of_gross=float(share) if np.isfinite(share) else None,
        total_without_top_k=float(without_top),
        profit_factor=float(pf),
        win_rate=float((p > 0).mean()),
        note=(
            f"removing the best {top_k} trades turns {total:+.0f} into {without_top:+.0f} -- "
            "the result IS those trades"
            if fail
            else f"best {top_k} trades are {share:.0%} of gross profit"
        ),
    )


def runs_test(pnl: np.ndarray) -> dict:
    """Trade dependence, via the runs test (neurotrader, BM3KZPg6zic).

    A run is a maximal streak of same-signed trades. Positive z = more runs than independence
    predicts = winners tend to follow losers. His Donchian scored z = 2.7 and skipping the trade
    after a winner improved profit factor at every lookback.

    Reported, not acted on. If z is large the filter is worth TESTING -- through the full
    validation ladder like anything else, because it is one more parameter to overfit.
    """
    s = np.sign(np.asarray(pnl, dtype=float))
    s = s[s != 0]
    n = s.size
    if n < 20:
        return dict(verdict="warn", reason=f"only {n} non-zero trades, runs test unreliable")
    n_pos, n_neg = int((s > 0).sum()), int((s < 0).sum())
    if n_pos == 0 or n_neg == 0:
        return dict(verdict="warn", reason="all trades the same sign")
    runs = 1 + int((s[1:] != s[:-1]).sum())
    mu = 2 * n_pos * n_neg / n + 1
    var = (mu - 1) * (mu - 2) / (n - 1)
    z = (runs - mu) / np.sqrt(var) if var > 0 else 0.0
    return dict(
        verdict="pass",
        runs=runs,
        expected_runs=float(mu),
        z=float(z),
        interpretation=(
            "losers tend to be followed by winners -- a skip-after-winner filter is worth testing"
            if z > 2
            else "streaky: winners follow winners"
            if z < -2
            else "no significant trade dependence"
        ),
    )


# --------------------------------------------------------------------- overfitting
def parameter_plateau(surface: dict[tuple, float]) -> dict:
    """Is the best cell on a plateau or a spike?

    "generally when selecting parameters we want to choose values that are stable locally ...
    ideally we see a plateau and choose the center of it" -- and the counter-example from the
    trend-line video, "there's a large spike in performance from 32 to 42 but my guess is that
    is just random luck. Beware of spikes."

    `surface` maps a parameter tuple to a score. Neighbours are cells differing in exactly one
    coordinate by one grid step.
    """
    if len(surface) < 4:
        return dict(verdict="warn", reason="grid too small to judge stability")
    keys = list(surface)
    vals = np.array([surface[k] for k in keys], dtype=float)
    best_i = int(np.nanargmax(vals))
    best_key = keys[best_i]
    axes = [sorted({k[d] for k in keys}) for d in range(len(best_key))]

    nb = []
    for d, axis in enumerate(axes):
        pos = axis.index(best_key[d])
        for step in (-1, 1):
            j = pos + step
            if 0 <= j < len(axis):
                cand = list(best_key)
                cand[d] = axis[j]
                if tuple(cand) in surface:
                    nb.append(surface[tuple(cand)])
    if not nb:
        return dict(verdict="warn", reason="best cell has no neighbours in the grid")

    nb = np.array(nb, dtype=float)
    best, median = float(vals[best_i]), float(np.nanmedian(vals))
    drop = (best - np.nanmean(nb)) / abs(best) if best else np.nan
    # a spike: neighbours fall back toward the middle of the whole surface
    spike = bool(np.nanmean(nb) < median + 0.25 * (best - median))
    return dict(
        verdict=_verdict(False, spike),
        best_cell=str(best_key),
        best_score=best,
        neighbour_mean=float(np.nanmean(nb)),
        surface_median=median,
        relative_drop_to_neighbours=float(drop) if np.isfinite(drop) else None,
        note=(
            "SPIKE: neighbours fall back toward the surface median. Treat as luck until shown "
            "otherwise; prefer the centre of a flat region."
            if spike
            else "plateau: neighbouring parameters perform comparably"
        ),
    )


# --------------------------------------------------------------------- sample bias
def data_coverage(dates: np.ndarray, *, max_gap_days: int = 5) -> dict:
    """Gaps and coverage. A hole in the data is a silent sample bias.

    This project has a real one: the MNQ series contains a 283-day hole, and an earlier
    analysis reported wildly wrong gap counts because the profile files were not date-sorted.
    Sort first, always.
    """
    d = np.sort(np.asarray(dates, dtype="datetime64[D]"))
    if d.size < 2:
        return dict(verdict="fail", reason="not enough data")
    days = np.unique(d)
    deltas = np.diff(days).astype("timedelta64[D]").astype(int)
    gaps = [dict(after=str(days[i]), days=int(g)) for i, g in enumerate(deltas) if g > max_gap_days]
    span = (days[-1] - days[0]).astype(int) + 1
    weekdays = np.is_busday(days.astype("datetime64[D]"))
    coverage = len(days) / max(1, np.busday_count(days[0], days[-1]) + 1)
    biggest = max((g["days"] for g in gaps), default=0)
    return dict(
        verdict=_verdict(biggest > 60, bool(gaps) or coverage < 0.9),
        first=str(days[0]),
        last=str(days[-1]),
        calendar_span_days=int(span),
        trading_days=len(days),
        business_day_coverage=float(coverage),
        non_weekday_sessions=int((~weekdays).sum()),
        n_gaps=len(gaps),
        largest_gap_days=biggest,
        gaps=gaps[:10],
    )


def regime_balance(pnl: np.ndarray, regime: np.ndarray) -> dict:
    """Does the edge exist in more than one regime, or is it one market condition in disguise?

    A strategy profitable only in the regime that happened to dominate the sample has not been
    shown to have an edge; it has been shown to have exposure.
    """
    p, r = np.asarray(pnl, dtype=float), np.asarray(regime)
    out = {}
    for lab in np.unique(r):
        m = r == lab
        seg = p[m]
        if seg.size:
            out[str(lab)] = dict(
                n=int(seg.size),
                share=float(seg.size / p.size),
                total=float(seg.sum()),
                mean=float(seg.mean()),
            )
    positive = [v for v in out.values() if v["total"] > 0]
    dominant = max((v["share"] for v in out.values()), default=1.0)
    return dict(
        verdict=_verdict(len(positive) <= 1 and len(out) > 1, dominant > 0.7),
        by_regime=out,
        n_profitable_regimes=len(positive),
        n_regimes=len(out),
        largest_regime_share=float(dominant),
        note=(
            "profitable in only one regime -- this is exposure, not edge"
            if len(positive) <= 1 and len(out) > 1
            else "profitable across multiple regimes"
        ),
    )


# --------------------------------------------------------------------- futures specifics
def roll_integrity(df: pd.DataFrame, *, jump_atr: float = 6.0) -> dict:
    """Contract-roll and back-adjustment artifacts -- the futures analogue of survivorship bias.

    A stitched continuous series is fine for looking at and wrong for trade-level P&L: the
    splice injects a price jump that no trader could have captured, and a mean-reversion rule
    will happily "trade" it. This looks for overnight jumps too large to be real.
    """
    o, c = df["open"].to_numpy(float), df["close"].to_numpy(float)
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    gap = np.abs(o - np.roll(c, 1))
    gap[0] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(atr > 0, gap / atr, 0.0)
    susp = np.flatnonzero(z > jump_atr)
    dates = (
        df["dt"].dt.strftime("%Y-%m-%d").to_numpy()
        if "dt" in df
        else np.arange(len(df)).astype(str)
    )
    return dict(
        verdict=_verdict(False, susp.size > 0),
        n_suspect_jumps=int(susp.size),
        threshold_atr=jump_atr,
        largest_jump_atr=float(np.nanmax(z)) if z.size else 0.0,
        suspects=[
            dict(i=int(i), when=str(dates[i]), jump_atr=round(float(z[i]), 1)) for i in susp[:10]
        ],
        note=(
            "possible roll or back-adjustment splices. Confirm the backtest trades the ACTUAL "
            "contract and states its roll rule; a splice is not a tradable move."
            if susp.size
            else "no implausible overnight jumps found"
        ),
    )


# --------------------------------------------------------------------- costs
def cost_sensitivity(simulate_fn, costs: np.ndarray) -> dict:
    """At what cost does the edge die? `simulate_fn(cost_pts) -> array of per-trade P&L`.

    The Donchian cliff at low lookbacks is entirely fees. An edge whose breakeven cost sits
    just above the cost you assumed is not an edge, it is an estimate of your broker.
    """
    rows = []
    for c in np.asarray(costs, dtype=float):
        p = np.asarray(simulate_fn(float(c)), dtype=float)
        rows.append(
            dict(
                cost_pts=float(c),
                n=int(p.size),
                total=float(p.sum()),
                mean=float(p.mean()) if p.size else np.nan,
            )
        )
    tot = np.array([r["total"] for r in rows])
    cs = np.array([r["cost_pts"] for r in rows])
    breakeven = None
    sign = np.sign(tot)
    flip = np.flatnonzero(np.diff(sign) < 0)
    if flip.size:
        i = int(flip[0])
        lo, hi = tot[i], tot[i + 1]
        breakeven = (
            float(cs[i] + (cs[i + 1] - cs[i]) * lo / (lo - hi)) if lo != hi else float(cs[i])
        )
    return dict(
        verdict=_verdict(bool(tot[0] <= 0), breakeven is not None and breakeven < 2 * cs[0]),
        curve=rows,
        breakeven_cost_pts=breakeven,
        assumed_cost_pts=float(cs[0]),
        headroom=(float(breakeven / cs[0]) if breakeven and cs[0] > 0 else None),
        note=(
            f"edge dies at {breakeven:.2f} pts, only {breakeven / cs[0]:.1f}x the assumed "
            f"{cs[0]:.2f} -- too little headroom for slippage in fast markets"
            if breakeven and cs[0] > 0 and breakeven < 2 * cs[0]
            else f"breakeven cost {breakeven:.2f} pts"
            if breakeven
            else "profitable at every cost tested"
        ),
    )


# --------------------------------------------------------------------- bundle
def run_all(
    *,
    pnl: np.ndarray,
    dates: np.ndarray | None = None,
    regime: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
    surface: dict | None = None,
    simulate_fn=None,
    costs: np.ndarray | None = None,
) -> dict:
    """Run every applicable check. Missing inputs are skipped, and skipping is reported."""
    out: dict[str, dict] = {"luck": luck_battery(pnl), "trade_dependence": runs_test(pnl)}
    if dates is not None:
        out["coverage"] = data_coverage(dates)
    if regime is not None:
        out["regime"] = regime_balance(pnl, regime)
    if df is not None:
        out["roll"] = roll_integrity(df)
    if surface is not None:
        out["plateau"] = parameter_plateau(surface)
    if simulate_fn is not None and costs is not None:
        out["costs"] = cost_sensitivity(simulate_fn, costs)

    fails = [k for k, v in out.items() if v.get("verdict") == "fail"]
    warns = [k for k, v in out.items() if v.get("verdict") == "warn"]
    out["_summary"] = dict(
        verdict=_verdict(bool(fails), bool(warns)),
        failed=fails,
        warned=warns,
        checks_run=len(out),
        skipped=[
            n
            for n, cond in (
                ("coverage", dates is None),
                ("regime", regime is None),
                ("roll", df is None),
                ("plateau", surface is None),
                ("costs", simulate_fn is None or costs is None),
            )
            if cond
        ],
    )
    return out
