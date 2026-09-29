"""Charts that explain a STRATEGY, as opposed to charts that score one.

`viz.py` answers "is this real?" -- permutation histograms, PBO panels, cost curves. This
module answers the questions you ask first and that a scorecard never shows:

    what does this actually DO?          anatomy()      price, indicators, every entry and exit
    what are the rules, in words?        rule_card()    plain language beside the real code
    could noise have made this curve?    permutation_equity()  neurotrader's red-vs-grey chart
    what do the losing paths look like?  monte_carlo()  the fan, not just the median
    when does it work?                   monthly_heatmap(), rolling_performance()
    how do trades behave?                mae_mfe(), trade_dependence()
    how was it validated?                walkforward_folds()

The anatomy chart is the one that catches errors no statistic will. An entry marker on the
wrong bar, a stop that plots inside the candle it was supposed to be outside, a trade held
through a session close -- those are visible in one glance and invisible in a Sharpe ratio.
"""

from __future__ import annotations

import pathlib
import textwrap

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .viz import GOOD, MUTED, NULL, REAL, WARN, _save, _tag

UP, DOWN = "#26a69a", "#ef5350"


# ------------------------------------------------------------------ anatomy
def anatomy(
    df,
    trades,
    out,
    *,
    start=None,
    n_bars=260,
    indicators=None,
    title="Strategy anatomy",
    atr=None,
    stop_atr=None,
    tp_atr=None,
):
    """Candles, indicators, and every entry/exit marked, over a readable window.

    Look at this before believing any number. Entries on the wrong bar, stops drawn inside the
    candle they should sit outside, a position carried through a session close -- all obvious
    here and all invisible in a summary statistic.

    `trades` needs entry_i, exit_i, direction, pnl_pts (the simulator's output).
    `indicators` is {label: array} drawn over price.
    """
    n = len(df)
    if start is None:  # default to the window with the most activity
        if len(trades):
            mid = int(trades["entry_i"].median())
            start = max(0, min(mid - n_bars // 2, n - n_bars))
        else:
            start = max(0, n // 2)
    end = min(start + n_bars, n)
    d = df.iloc[start:end]
    x = mdates.date2num(pd.to_datetime(d["dt"]).dt.tz_localize(None))
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True, gridspec_kw=dict(height_ratios=[4, 1], hspace=0.05)
    )
    w = (x[1] - x[0]) * 0.7 if len(x) > 1 else 0.001
    for i in range(len(d)):
        col = UP if c[i] >= o[i] else DOWN
        ax.plot([x[i], x[i]], [l[i], h[i]], color=col, lw=0.7, zorder=2)
        ax.add_patch(
            Rectangle(
                (x[i] - w / 2, min(o[i], c[i])),
                w,
                abs(c[i] - o[i]) or 1e-9,
                facecolor=col,
                edgecolor=col,
                lw=0.5,
                zorder=3,
            )
        )
    for lbl, arr in (indicators or {}).items():
        ax.plot(x, np.asarray(arr, float)[start:end], lw=1.3, label=lbl, zorder=4)

    seg = trades[(trades["exit_i"] >= start) & (trades["entry_i"] < end)] if len(trades) else trades
    for _, t in seg.iterrows():
        ei, xi, dirn = int(t["entry_i"]), int(t["exit_i"]), int(t["direction"])
        if not (start <= ei < end):
            continue
        xe = x[ei - start]
        px = float(df["close"].iloc[ei])
        long_ = dirn > 0
        ax.scatter(
            [xe],
            [px],
            marker="^" if long_ else "v",
            s=130,
            color=GOOD if long_ else REAL,
            edgecolor="white",
            lw=0.8,
            zorder=6,
        )
        if atr is not None and stop_atr:
            a = float(np.asarray(atr)[ei])
            ax.plot(
                [xe, x[min(xi, end - 1) - start]],
                [px - dirn * stop_atr * a] * 2,
                color=REAL,
                ls=":",
                lw=0.9,
                zorder=5,
            )
            if tp_atr:
                ax.plot(
                    [xe, x[min(xi, end - 1) - start]],
                    [px + dirn * tp_atr * a] * 2,
                    color=GOOD,
                    ls=":",
                    lw=0.9,
                    zorder=5,
                )
        if start <= xi < end:
            xx = x[xi - start]
            xp = px + t["pnl_pts"] * dirn
            won = t["pnl_pts"] > 0
            ax.scatter(
                [xx],
                [xp],
                marker="X",
                s=110,
                color=GOOD if won else REAL,
                edgecolor="white",
                lw=0.8,
                zorder=6,
            )
            ax.plot([xe, xx], [px, xp], color=GOOD if won else REAL, lw=1.1, alpha=0.55, zorder=4)

    # shade the session so "flat at the close" is checkable by eye
    dt = pd.to_datetime(d["dt"])
    mins = dt.dt.hour * 60 + dt.dt.minute
    rth = ((mins >= 510) & (mins < 900)).to_numpy()
    for i in range(1, len(rth)):
        if rth[i] and not rth[i - 1]:
            j = i
            while j < len(rth) and rth[j]:
                j += 1
            ax.axvspan(x[i], x[min(j, len(x) - 1)], color="#1f6f78", alpha=0.10, zorder=1)

    ax.set_title(
        f"{title}   bars {start}-{end}   {len(seg)} trades shown "
        f"(shaded = RTH; ^ / v entry, X exit; dotted = stop/target)"
    )
    ax.set_ylabel("price")
    ax.grid(alpha=0.15)
    if indicators:
        ax.legend(
            facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8, ncol=4
        )
    axv.bar(x, d["volume"].to_numpy(float), width=w, color=MUTED, alpha=0.55)
    axv.set_ylabel("volume")
    axv.grid(alpha=0.15)
    axv.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    return _save(fig, out)


# ------------------------------------------------------------------ rules
def rule_card(
    out,
    *,
    name,
    entry_long,
    entry_short,
    exit_rule,
    code,
    notes=None,
    costs=None,
    title="Strategy rules",
):
    """The rules in plain language beside the code that implements them.

    Two failure modes this catches: the description quietly drifting from the implementation,
    and a rule nobody can restate without reading 400 lines. If the words and the code disagree,
    one of them is a bug.
    """
    fig = plt.figure(figsize=(15, 8.6))
    fig.suptitle(f"{title} - {name}", fontsize=15, weight="bold", x=0.02, ha="left", y=0.97)
    left = fig.add_axes([0.02, 0.05, 0.42, 0.87])
    left.axis("off")
    right = fig.add_axes([0.47, 0.05, 0.51, 0.87])
    right.axis("off")

    y = 1.0

    def block(ax, head, body, colour, yy, wrap=52):
        ax.text(0, yy, head, fontsize=11, weight="bold", color=colour, transform=ax.transAxes)
        yy -= 0.035
        for line in body if isinstance(body, list) else [body]:
            for seg in textwrap.wrap(line, wrap) or [""]:
                ax.text(0.015, yy, seg, fontsize=9.5, color="#e6edf3", transform=ax.transAxes)
                yy -= 0.028
        return yy - 0.025

    y = block(left, "ENTRY - LONG", entry_long, GOOD, y)
    y = block(left, "ENTRY - SHORT", entry_short, REAL, y)
    y = block(left, "EXIT", exit_rule, WARN, y)
    if costs:
        y = block(left, "COSTS", costs, NULL, y)
    if notes:
        y = block(left, "NOTES", notes, MUTED, y)

    right.text(
        0, 1.0, "IMPLEMENTATION", fontsize=11, weight="bold", color=NULL, transform=right.transAxes
    )
    right.add_patch(
        Rectangle(
            (0, 0),
            1,
            0.955,
            transform=right.transAxes,
            facecolor="#161b22",
            edgecolor="#30363d",
            lw=1,
        )
    )
    yy = 0.93
    for line in textwrap.dedent(code).strip().splitlines():
        right.text(
            0.015,
            yy,
            line.rstrip(),
            fontsize=8.4,
            family="monospace",
            color="#a5d6ff" if line.strip().startswith("#") else "#e6edf3",
            transform=right.transAxes,
        )
        yy -= 0.0225
        if yy < 0.01:
            break
    return _save(fig, out)


# ------------------------------------------------------------------ permutation curves
def permutation_equity(real_pnl, perm_pnls, out, *, title="Real vs permuted equity", p_value=None):
    """neurotrader's signature chart: the real equity curve in red over grey permutations.

    The histogram gives the p-value; this shows the SHAPE. A real curve that merely sits at the
    top of the grey band is a different thing from one that leaves it -- and a real curve buried
    inside the band needs no statistic to interpret.
    """
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for p in perm_pnls:
        ax.plot(np.nancumsum(np.nan_to_num(p)), color=MUTED, lw=0.5, alpha=0.20, zorder=1)
    if len(perm_pnls):
        stack = np.vstack([np.nancumsum(np.nan_to_num(p)) for p in perm_pnls])
        for q, a in ((95, 0.35), (75, 0.25)):
            ax.plot(np.percentile(stack, q, axis=0), color=NULL, lw=1.0, alpha=a, zorder=2)
        ax.plot(
            np.percentile(stack, 50, axis=0),
            color=NULL,
            lw=1.4,
            zorder=2,
            label="permutation median",
        )
    ax.plot(np.nancumsum(np.nan_to_num(real_pnl)), color=REAL, lw=2.0, zorder=3, label="real")
    ax.axhline(0, color=MUTED, lw=0.7, ls=":")
    ax.set_xlabel("bars")
    ax.set_ylabel("cumulative P&L (points)")
    t = f"{title}   {len(perm_pnls)} permutations"
    if p_value is not None:
        t += f"   p = {p_value:.4f}"
        _tag(
            ax,
            "PASS" if p_value < 0.01 else f"FAIL p={p_value:.3f}",
            GOOD if p_value < 0.01 else REAL,
        )
    ax.set_title(t)
    ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    ax.grid(alpha=0.2)
    return _save(fig, out)


# ------------------------------------------------------------------ monte carlo
def monte_carlo(paths, out, *, target=None, ruin=None, title="Monte Carlo", n_show=300):
    """The fan of simulated account paths, not just its median.

    `paths` is (n_sims, n_days) of cumulative account P&L. A median path looks reassuring on
    its own; the point of the simulation is the spread and the left tail, so both are drawn.
    """
    p = np.asarray(paths, float)
    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[3, 1], wspace=0.05)
    )
    idx = np.random.default_rng(0).choice(len(p), size=min(n_show, len(p)), replace=False)
    for i in idx:
        ax.plot(p[i], color=MUTED, lw=0.4, alpha=0.18)
    for q, col, lw in ((5, REAL, 1.3), (50, "#e6edf3", 1.8), (95, GOOD, 1.3)):
        ax.plot(np.percentile(p, q, axis=0), color=col, lw=lw, label=f"{q}th pct")
    if target is not None:
        ax.axhline(target, color=GOOD, ls="--", lw=1.2, label=f"target {target:+,.0f}")
    if ruin is not None:
        ax.axhline(ruin, color=REAL, ls="--", lw=1.2, label=f"max loss {ruin:+,.0f}")
    ax.axhline(0, color=MUTED, lw=0.7, ls=":")
    ax.set_xlabel("trading day")
    ax.set_ylabel("account P&L ($)")
    ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    ax.grid(alpha=0.2)

    final = p[:, -1]
    axh.hist(final, bins=45, orientation="horizontal", color=NULL, alpha=0.85)
    axh.axhline(np.median(final), color="#e6edf3", lw=1.5)
    if target is not None:
        axh.axhline(target, color=GOOD, ls="--", lw=1.2)
        hit = float((p.max(axis=1) >= target).mean())
        axh.set_title(f"reached target\n{hit:.1%}", fontsize=9)
    if ruin is not None:
        axh.axhline(ruin, color=REAL, ls="--", lw=1.2)
    axh.set_yticklabels([])
    axh.set_xlabel("count")
    axh.grid(alpha=0.2)
    ax.set_title(
        f"{title}   {len(p):,} simulations   median final "
        f"${np.median(final):+,.0f}   P(loss) {float((final < 0).mean()):.1%}"
    )
    return _save(fig, out)


# ------------------------------------------------------------------ when does it work
def monthly_heatmap(pnl, dates, out, *, title="P&L by month", point_value=1.0):
    s = pd.Series(np.asarray(pnl, float) * point_value, index=pd.to_datetime(np.asarray(dates)))
    g = s.groupby([s.index.year, s.index.month]).sum().reset_index()
    g.columns = ["year", "month", "pnl"]
    m = g.pivot_table(index="year", columns="month", values="pnl")
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(m) + 2.2))
    v = np.nanmax(np.abs(m.to_numpy())) or 1
    im = ax.imshow(m.to_numpy(), cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(m.shape[1]), [f"{c:02d}" for c in m.columns])
    ax.set_yticks(range(len(m)), [str(i) for i in m.index])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = m.to_numpy()[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:,.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="black" if abs(val) < v * 0.55 else "white",
                )
    fig.colorbar(im, ax=ax, label="P&L")
    ax.set_title(f"{title} - look for a few months carrying the whole result")
    return _save(fig, out)


def rolling_performance(pnl, out, *, window=2000, title="Rolling performance"):
    """Edge decay is invisible in a total. A strategy that worked until 2023 and not since has
    the same profit factor as one that worked throughout."""
    s = pd.Series(np.nan_to_num(np.asarray(pnl, float)))
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw=dict(hspace=0.08)
    )
    roll_sum = s.rolling(window).sum()
    a1.plot(roll_sum, color=REAL, lw=1.2)
    a1.axhline(0, color=MUTED, lw=0.8, ls=":")
    a1.fill_between(roll_sum.index, roll_sum, 0, where=roll_sum > 0, color=GOOD, alpha=0.18)
    a1.fill_between(roll_sum.index, roll_sum, 0, where=roll_sum <= 0, color=REAL, alpha=0.18)
    a1.set_ylabel(f"{window}-bar P&L")
    a1.grid(alpha=0.2)
    win = s.rolling(window).apply(lambda x: (x > 0).sum() / max((x != 0).sum(), 1), raw=True)
    a2.plot(win, color=NULL, lw=1.1)
    a2.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    a2.set_ylabel("rolling win rate")
    a2.set_xlabel("bar")
    a2.grid(alpha=0.2)
    a1.set_title(f"{title} - is the edge steady, or was it one period?")
    return _save(fig, out)


# ------------------------------------------------------------------ trade behaviour
def mae_mfe(trades, out, *, title="MAE / MFE"):
    """How far each trade went against you before it worked, and how much it gave back.

    Stops sitting beyond the MAE cloud are never hit; targets beyond the MFE cloud are never
    reached. This is where exit parameters come from instead of a grid search.
    """
    if not {"mae", "mfe"}.issubset(trades.columns) or trades.empty:
        return None
    won = trades["pnl_pts"] > 0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.scatter(
        trades.loc[won, "mae"],
        trades.loc[won, "pnl_pts"],
        s=12,
        color=GOOD,
        alpha=0.55,
        label="winners",
    )
    a1.scatter(
        trades.loc[~won, "mae"],
        trades.loc[~won, "pnl_pts"],
        s=12,
        color=REAL,
        alpha=0.55,
        label="losers",
    )
    a1.set_xlabel("MAE - worst excursion against (points)")
    a1.set_ylabel("final P&L (points)")
    a1.axhline(0, color=MUTED, lw=0.7, ls=":")
    a1.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    a1.grid(alpha=0.2)
    a1.set_title("a stop beyond this cloud is never hit")
    a2.scatter(trades.loc[won, "mfe"], trades.loc[won, "pnl_pts"], s=12, color=GOOD, alpha=0.55)
    a2.scatter(trades.loc[~won, "mfe"], trades.loc[~won, "pnl_pts"], s=12, color=REAL, alpha=0.55)
    a2.plot([0, trades["mfe"].max()], [0, trades["mfe"].max()], color=MUTED, ls=":", lw=0.9)
    a2.set_xlabel("MFE - best excursion in favour (points)")
    a2.axhline(0, color=MUTED, lw=0.7, ls=":")
    a2.grid(alpha=0.2)
    a2.set_title("distance below the diagonal = profit given back")
    fig.suptitle(title, y=1.01)
    return _save(fig, out)


def trade_dependence(pnl, out, *, runs=None, title="Trade dependence"):
    """Previous trade's result against the next one -- the scatter from neurotrader's runs-test
    video. A tilt means a skip-after-winner (or loser) filter is worth TESTING, not adopting."""
    p = np.asarray(pnl, float)
    if len(p) < 20:
        return None
    prev, nxt = p[:-1], p[1:]
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[2, 1], wspace=0.25)
    )
    a1.scatter(prev, nxt, s=11, color=NULL, alpha=0.5)
    a1.axhline(0, color=MUTED, lw=0.7, ls=":")
    a1.axvline(0, color=MUTED, lw=0.7, ls=":")
    for m, lbl, col in ((prev < 0, "after a loser", GOOD), (prev > 0, "after a winner", REAL)):
        if m.sum():
            a1.plot(
                [prev[m].min(), prev[m].max()],
                [nxt[m].mean()] * 2,
                color=col,
                lw=2,
                label=f"{lbl}: mean {nxt[m].mean():+.2f}",
            )
    a1.set_xlabel("previous trade P&L")
    a1.set_ylabel("next trade P&L")
    a1.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    a1.grid(alpha=0.2)
    after_w = nxt[prev > 0].mean() if (prev > 0).any() else np.nan
    after_l = nxt[prev < 0].mean() if (prev < 0).any() else np.nan
    a2.bar(
        ["after\nwinner", "after\nloser"],
        [after_w, after_l],
        color=[REAL if after_w < 0 else GOOD, GOOD if after_l > 0 else REAL],
        alpha=0.9,
    )
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylabel("mean next-trade P&L")
    a2.grid(alpha=0.2, axis="y")
    t = title
    if runs and "z" in runs:
        t += f"   runs-test z = {runs['z']:+.2f}"
        a2.set_title(runs.get("interpretation", "")[:46], fontsize=8.5)
    fig.suptitle(t, y=1.01)
    return _save(fig, out)


# ------------------------------------------------------------------ walk-forward
def walkforward_folds(folds, split, out, *, dates=None, picks=None, title="Walk-forward"):
    """The fold structure drawn out: which bars trained each block, which bars it then traded.

    Worth one look per dataset. On MNQ it shows immediately that early optimisation windows end
    13 months before the block they trade, because the 396-day hole sits between them.
    """
    fig, ax = plt.subplots(figsize=(13, 0.34 * len(folds) + 2.4))
    for f in folds:
        ax.barh(
            f.i, f.opt.stop - f.opt.start, left=f.opt.start, height=0.62, color=NULL, alpha=0.65
        )
        ax.barh(
            f.i, f.test.stop - f.test.start, left=f.test.start, height=0.62, color=REAL, alpha=0.95
        )
        if picks and f.i < len(picks):
            ax.text(f.test.stop + 200, f.i, str(picks[f.i]), fontsize=7, va="center", color=MUTED)
    h0, _ = split.holdout_bounds
    ax.axvline(split.train_end, color=GOOD, lw=1.3, ls="--", label="train | validate")
    ax.axvline(h0, color=WARN, lw=1.3, ls="--", label="validate | HOLDOUT")
    ax.set_xlabel("bar index" if dates is None else "bar index (dates in caption)")
    ax.set_ylabel("fold")
    ax.invert_yaxis()
    ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    ax.grid(alpha=0.18, axis="x")
    cap = f"{title} - blue = optimisation window, red = out-of-sample block"
    if dates is not None and len(folds):
        d = np.asarray(dates)
        cap += (
            f"   ({str(d[folds[0].test.start])[:10]} -> "
            f"{str(d[min(folds[-1].test.stop - 1, len(d) - 1)])[:10]})"
        )
    ax.set_title(cap)
    return _save(fig, out)


# ------------------------------------------------------------------ the conclusion
def conclusion_card(
    out, *, name, verdict, gates, headline, numbers, what_it_means, title="Backtest conclusion"
):
    """One image that answers "so what?".

    `gates` is [(label, "pass"|"fail"|"warn"|"skipped", detail)]. `numbers` is {label: value}.
    `what_it_means` is the plain-English reading -- the part a table never supplies.
    """
    colour = {"pass": GOOD, "fail": REAL, "warn": WARN, "skipped": MUTED}
    vcol = {"PASSED": GOOD, "REJECTED": REAL, "INCOMPLETE": MUTED}.get(verdict.split()[0], WARN)
    fig = plt.figure(figsize=(13.5, 0.44 * len(gates) + 6.4))
    fig.text(0.02, 0.965, f"{title} - {name}", fontsize=15, weight="bold")
    fig.text(0.98, 0.965, verdict, fontsize=16, weight="bold", color=vcol, ha="right")
    fig.text(0.02, 0.915, headline, fontsize=11, color="#e6edf3", wrap=True)

    ax = fig.add_axes([0.02, 0.30, 0.55, 0.57])
    ax.axis("off")
    ax.text(0, 1.0, "GATES", fontsize=11, weight="bold", color=NULL, transform=ax.transAxes)
    y = 0.94
    for lbl, st, detail in gates:
        ax.text(0.0, y, lbl, fontsize=9.5, transform=ax.transAxes)
        ax.text(
            0.52,
            y,
            st.upper(),
            fontsize=9.5,
            weight="bold",
            color=colour.get(st, MUTED),
            transform=ax.transAxes,
        )
        y -= 0.05
        if detail:
            for seg in textwrap.wrap(detail, 74)[:2]:
                ax.text(0.02, y, seg, fontsize=8.2, color=MUTED, transform=ax.transAxes)
                y -= 0.042

    axn = fig.add_axes([0.60, 0.30, 0.38, 0.57])
    axn.axis("off")
    axn.text(0, 1.0, "KEY NUMBERS", fontsize=11, weight="bold", color=NULL, transform=axn.transAxes)
    y = 0.94
    for k, v in numbers.items():
        axn.text(0.0, y, k, fontsize=9.5, color=MUTED, transform=axn.transAxes)
        axn.text(1.0, y, str(v), fontsize=9.5, weight="bold", ha="right", transform=axn.transAxes)
        y -= 0.055

    axw = fig.add_axes([0.02, 0.02, 0.96, 0.25])
    axw.axis("off")
    axw.text(
        0, 1.0, "WHAT THIS MEANS", fontsize=11, weight="bold", color=vcol, transform=axw.transAxes
    )
    y = 0.86
    for para in what_it_means:
        for seg in textwrap.wrap(para, 140):
            axw.text(0.0, y, seg, fontsize=9.5, color="#e6edf3", transform=axw.transAxes)
            y -= 0.115
        y -= 0.05
    return _save(fig, pathlib.Path(out))
