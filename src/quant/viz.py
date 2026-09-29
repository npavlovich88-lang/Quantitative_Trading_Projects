"""Figures. Every output gets one.

A number in a terminal is forgettable and a table of numbers hides its own shape. The specific
things these are built to make un-missable:

  * where the splits are, so an equity curve can never be read without seeing which part was
    fitted (the single most misleading chart in trading is an equity curve with no split marked)
  * how far the real result sits from the permutation distribution -- the MCPT chart is the
    whole argument in one picture
  * whether a parameter is a plateau or a spike
  * how much headroom the edge has over costs

Style follows neurotrader's: dark ground, red for real, blue/grey for the null. Agg backend, so
this works headless and never blocks on a window.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REAL = "#ff3b30"
NULL = "#3a7bd5"
GOOD = "#2ecc71"
WARN = "#f5a623"
MUTED = "#8a8f98"
SEG = {"train": "#2c3e50", "validate": "#1f6f78", "holdout": "#6b2737"}

plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#0d1117",
        "savefig.facecolor": "#0d1117",
        "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3",
        "axes.edgecolor": "#30363d",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": "#21262d",
        "font.size": 9,
        "axes.titlesize": 11,
        "figure.dpi": 130,
    }
)


def _save(fig, out: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _tag(ax, text, color):
    ax.text(
        0.985,
        0.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=color,
        fontsize=10,
        weight="bold",
        bbox=dict(facecolor="#0d1117", edgecolor=color, boxstyle="round,pad=0.35", alpha=0.9),
    )


# ------------------------------------------------------------------ equity
def equity_with_splits(bar_pnl, split, out, *, dates=None, title="Equity"):
    """Cumulative P&L with the train/validate/holdout regions shaded.

    Never show an equity curve without this. The shading is the difference between "it made
    money" and "it made money on the part we fitted it to".
    """
    p = np.asarray(bar_pnl, dtype=float)
    eq = np.nancumsum(np.nan_to_num(p))
    x = np.arange(len(eq)) if dates is None else np.asarray(dates)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    h0, h1 = split.holdout_bounds
    for label, sl in (
        ("train", split.train),
        ("validate", split.validate),
        ("holdout", slice(h0, h1)),
    ):
        a, b = sl.start, min(sl.stop, len(eq) - 1)
        if b > a:
            ax.axvspan(x[a], x[b], color=SEG[label], alpha=0.35, lw=0)
            ax.text(
                x[(a + b) // 2],
                ax.get_ylim()[1],
                label,
                ha="center",
                va="top",
                color=MUTED,
                fontsize=8,
            )
    ax.plot(x, eq, color=REAL, lw=1.3)
    ax.axhline(0, color=MUTED, lw=0.6, ls=":")
    ax.set_title(f"{title} - shaded by segment; only the holdout is an honest estimate")
    ax.set_ylabel("cumulative P&L (points)")
    ax.grid(alpha=0.25)
    if split.is_holdout_spent():
        _tag(ax, "HOLDOUT SPENT", REAL)
    return _save(fig, out)


def drawdown(bar_pnl, out, *, title="Underwater"):
    p = np.nan_to_num(np.asarray(bar_pnl, dtype=float))
    eq = np.cumsum(p)
    dd = eq - np.maximum.accumulate(eq)
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.fill_between(np.arange(len(dd)), dd, 0, color=REAL, alpha=0.45, lw=0)
    ax.plot(dd, color=REAL, lw=0.8)
    ax.set_title(f"{title} - max {dd.min():,.0f} pts")
    ax.set_ylabel("drawdown (points)")
    ax.grid(alpha=0.25)
    return _save(fig, out)


# ------------------------------------------------------------------ MCPT
def mcpt_histogram(perm_scores, real_score, p_value, out, *, metric="profit factor", title="MCPT"):
    """The chart that carries the argument: where the real result sits in the null."""
    perm = np.asarray([s for s in perm_scores if np.isfinite(s)], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    if perm.size:
        ax.hist(
            perm,
            bins=max(12, int(np.sqrt(perm.size))),
            color=NULL,
            alpha=0.85,
            label=f"permutations (n={perm.size})",
        )
    ax.axvline(real_score, color=REAL, lw=2.2, label=f"real = {real_score:.4f}")
    beaten = int((perm >= real_score).sum()) if perm.size else 0
    ax.set_xlabel(metric)
    ax.set_ylabel("count")
    ok = p_value < 0.01
    ax.set_title(
        f"{title}   p = {p_value:.4f}   ({beaten} of {perm.size} permutations matched or beat it)"
    )
    ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3")
    ax.grid(alpha=0.2)
    _tag(ax, "PASS p<0.01" if ok else f"FAIL p={p_value:.3f}", GOOD if ok else REAL)
    return _save(fig, out)


# ------------------------------------------------------------------ PBO
def pbo_panel(res, out, *, title="Probability of Backtest Overfitting"):
    """Three panels: the logit distribution, IS->OOS degradation, and the dominance check."""
    lam = np.asarray(res["lambdas"], dtype=float)
    is_m = np.asarray(res["is_metric"], dtype=float)
    oos_m = np.asarray(res["oos_metric"], dtype=float)
    pbo = res["pbo"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.hist(lam, bins=30, color=NULL, alpha=0.85)
    ax.axvline(0, color=REAL, lw=2)
    ax.set_title(f"logit distribution\nPBO = {pbo:.3f}  (reject above 0.05)")
    ax.set_xlabel(r"$\lambda_c$   (<= 0 means the IS winner was below the OOS median)")
    ax.grid(alpha=0.2)
    _tag(ax, "PASS" if pbo <= 0.05 else "FAIL", GOOD if pbo <= 0.05 else REAL)

    ax = axes[1]
    ok = np.isfinite(is_m) & np.isfinite(oos_m)
    ax.scatter(is_m[ok], oos_m[ok], s=9, color=NULL, alpha=0.55)
    if ok.sum() > 2:
        xs = np.linspace(np.nanmin(is_m[ok]), np.nanmax(is_m[ok]), 50)
        ax.plot(
            xs, res["degradation_slope"] * xs + res["degradation_intercept"], color=REAL, lw=1.8
        )
    ax.axhline(0, color=MUTED, lw=0.7, ls=":")
    ax.set_title(f"performance degradation\nslope = {res['degradation_slope']:+.3f}")
    ax.set_xlabel("in-sample metric of the chosen config")
    ax.set_ylabel("its out-of-sample metric")
    ax.grid(alpha=0.2)

    ax = axes[2]
    o = np.sort(oos_m[np.isfinite(oos_m)])
    ax.plot(o, np.linspace(0, 1, o.size), color=REAL, lw=1.8, label="OOS of the IS-best")
    ax.axvline(0, color=MUTED, lw=0.7, ls=":")
    ax.set_title(f"P(OOS metric < 0) = {res['prob_oos_loss']:.3f}")
    ax.set_xlabel("out-of-sample metric")
    ax.set_ylabel("cumulative probability")
    ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    ax.grid(alpha=0.2)

    fig.suptitle(
        f"{title}   {res['N_trials']} configurations, S={res['S']}, {res['n_combinations']} splits",
        y=1.02,
    )
    return _save(fig, out)


# ------------------------------------------------------------------ parameters
def parameter_surface(surface, out, *, title="Parameter surface", metric="profit factor"):
    """Heatmap for a 2-D grid, line for 1-D. Plateau vs spike should be obvious at a glance."""
    keys = list(surface)
    dims = len(keys[0]) if isinstance(keys[0], tuple) else 1
    fig, ax = plt.subplots(figsize=(8, 5) if dims == 2 else (9, 4))

    if dims == 2:
        xs = sorted({k[0] for k in keys})
        ys = sorted({k[1] for k in keys})
        z = np.full((len(ys), len(xs)), np.nan)
        for (a, b), v in surface.items():
            z[ys.index(b), xs.index(a)] = v
        im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=(-0.5, len(xs) - 0.5, -0.5, len(ys) - 0.5),
        )
        ax.set_xticks(range(len(xs)), [str(v) for v in xs])
        ax.set_yticks(range(len(ys)), [str(v) for v in ys])
        best = max(surface, key=lambda k: surface[k])
        ax.scatter(
            [xs.index(best[0])],
            [ys.index(best[1])],
            marker="o",
            s=110,
            facecolor="none",
            edgecolor=REAL,
            lw=2,
        )
        fig.colorbar(im, ax=ax, label=metric)
    else:
        ks = sorted(keys)
        ax.plot([str(k) for k in ks], [surface[k] for k in ks], color=REAL, marker="o", ms=3)
        ax.grid(alpha=0.25)
    ax.set_title(f"{title} - look for a plateau; a lone bright cell is luck until proven otherwise")
    return _save(fig, out)


# ------------------------------------------------------------------ costs
def cost_curve(res, out, *, title="Cost sensitivity"):
    cs = [r["cost_pts"] for r in res["curve"]]
    tot = [r["total"] for r in res["curve"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cs, tot, color=REAL, marker="o", ms=4)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.axvline(
        res["assumed_cost_pts"],
        color=GOOD,
        lw=1.4,
        ls="--",
        label=f"assumed {res['assumed_cost_pts']:.2f} pts",
    )
    if res.get("breakeven_cost_pts"):
        ax.axvline(
            res["breakeven_cost_pts"],
            color=WARN,
            lw=1.4,
            ls="--",
            label=f"breakeven {res['breakeven_cost_pts']:.2f} pts",
        )
    ax.set_xlabel("round-turn cost (points)")
    ax.set_ylabel("total P&L (points)")
    ax.set_title(f"{title} - {res['note']}")
    ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, out)


# ------------------------------------------------------------------ trades
def trade_distribution(pnl, out, *, luck=None, title="Trade distribution"):
    p = np.asarray(pnl, dtype=float)
    p = p[np.isfinite(p)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.hist(p, bins=40, color=NULL, alpha=0.85)
    a1.axvline(0, color=MUTED, lw=0.8, ls=":")
    a1.axvline(p.mean(), color=REAL, lw=1.6, label=f"mean {p.mean():+.2f}")
    a1.set_xlabel("P&L per trade (points)")
    a1.set_title(f"{title} - {len(p)} trades, win rate {(p > 0).mean():.0%}")
    a1.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    a1.grid(alpha=0.2)

    # Cumulative P&L in POINTS, best trade first. Normalising by the net total blows up when
    # that total is near zero -- which is precisely the case where concentration matters most.
    order = np.argsort(p)[::-1]
    a2.plot(np.cumsum(p[order]), color=REAL, lw=1.6)
    a2.axhline(p.sum(), color=MUTED, lw=0.8, ls=":", label=f"final {p.sum():+,.0f}")
    a2.axhline(0, color=MUTED, lw=0.6)
    a2.set_xlabel("trades, best first")
    a2.set_ylabel("cumulative P&L (points)")
    a2.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    ttl = "concentration"
    if luck:
        sh = luck.get("top_k_share_of_gross")
        if sh is not None:
            ttl += f" - best {luck['top_k']} = {sh:.0%} of gross profit"
        if luck["verdict"] == "fail":
            _tag(a2, "CARRIED BY A FEW TRADES", REAL)
    a2.set_title(ttl)
    a2.grid(alpha=0.2)
    return _save(fig, out)


def regime_bars(res, out, *, title="P&L by regime"):
    by = res["by_regime"]
    labs = list(by)
    tot = [by[k]["total"] for k in labs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labs, tot, color=[GOOD if t > 0 else REAL for t in tot], alpha=0.9)
    ax.axhline(0, color=MUTED, lw=0.8)
    for i, k in enumerate(labs):
        ax.text(
            i,
            tot[i],
            f"n={by[k]['n']}",
            ha="center",
            va="bottom" if tot[i] > 0 else "top",
            fontsize=8,
            color=MUTED,
        )
    ax.set_ylabel("total P&L (points)")
    ax.set_title(f"{title} - {res['note']}")
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, out)


# ------------------------------------------------------------------ verdict card
def verdict_card(checks, out, *, title="Diagnostics"):
    """One image summarising every check, so a result cannot be shown without its caveats."""
    items = [(k, v.get("verdict", "?")) for k, v in checks.items() if not k.startswith("_")]
    colour = {"pass": GOOD, "warn": WARN, "fail": REAL, "?": MUTED}
    fig, ax = plt.subplots(figsize=(7, 0.52 * len(items) + 1.5))
    ax.axis("off")
    summ = checks.get("_summary", {})
    overall = summ.get("verdict", "?")
    ax.text(0, 1.0, title, fontsize=13, weight="bold", transform=ax.transAxes)
    ax.text(
        1,
        1.0,
        overall.upper(),
        fontsize=13,
        weight="bold",
        ha="right",
        color=colour[overall],
        transform=ax.transAxes,
    )
    for i, (k, v) in enumerate(items):
        y = 0.88 - i * (0.82 / max(len(items), 1))
        ax.text(0.02, y, k.replace("_", " "), fontsize=10, transform=ax.transAxes)
        ax.text(
            0.98,
            y,
            v.upper(),
            fontsize=10,
            weight="bold",
            ha="right",
            color=colour[v],
            transform=ax.transAxes,
        )
    if summ.get("skipped"):
        ax.text(
            0.02,
            0.02,
            "not run: " + ", ".join(summ["skipped"]),
            fontsize=8,
            color=MUTED,
            transform=ax.transAxes,
        )
    return _save(fig, out)
