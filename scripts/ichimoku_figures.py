#!/usr/bin/env python3
"""Figures for the Ichimoku line-information study.

    python scripts/ichimoku_figures.py --dir out/ichimoku_information

Reads statistics.npz written by ichimoku_information.py. The point of these four is to make one
thing impossible to miss: the threshold that matters is not 1.96, it is whatever the best of
2,250 cells reaches on shuffled data, and that number is much higher.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quant.viz import GOOD, MUTED, NULL, REAL, WARN, _save

CLASSIC = ("mid(9)", "mid(26)", "middisp(52,26)", "span(9,26,26)", "chikou(26)")
CLASSIC_NAME = {
    "mid(9)": "Tenkan  mid(9)",
    "mid(26)": "Kijun  mid(26)",
    "middisp(52,26)": "script SenkouA  mid(52)>>26",
    "span(9,26,26)": "script SenkouB  span>>26",
    "chikou(26)": "Chikou  close vs close[26]",
}


def load(d: pathlib.Path):
    z = np.load(d / "statistics.npz", allow_pickle=False)
    return (
        z["real"],
        z["d"],
        z["null"],
        [str(x) for x in z["symbols"]],
        [str(x) for x in z["timeframes"]],
        [int(x) for x in z["horizons"]],
        [str(x) for x in z["labels"]],
    )


def fig_family_null(real, null, out: pathlib.Path) -> None:
    """The whole argument in one picture: where the best real cell sits in the null of bests."""
    n_perm = null.shape[0]
    with np.errstate(invalid="ignore"):
        null_max = np.nanmax(np.abs(null.reshape(n_perm, -1)), axis=1)
    obs = float(np.nanmax(np.abs(real)))
    p = (1 + int((null_max >= obs).sum())) / (1 + n_perm)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), width_ratios=[1.35, 1])
    ax.hist(null_max, bins=45, color=NULL, alpha=0.85, edgecolor="none")
    ax.axvline(obs, color=REAL, lw=2.2)
    ax.axvline(np.percentile(null_max, 95), color=WARN, lw=1.4, ls="--")
    ax.annotate(
        f"best real cell\n|t| = {obs:.3f}",
        xy=(obs, ax.get_ylim()[1] * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        color=REAL,
        fontsize=9,
        va="top",
    )
    ax.annotate(
        f"95th pct of null = {np.percentile(null_max, 95):.2f}",
        xy=(np.percentile(null_max, 95), ax.get_ylim()[1] * 0.55),
        xytext=(-8, 0),
        textcoords="offset points",
        color=WARN,
        fontsize=8,
        ha="right",
    )
    ax.set_xlabel("maximum |t| across all 2,250 cells")
    ax.set_ylabel(f"permutations (n={n_perm})")
    ax.set_title(f"Null distribution of the BEST cell     family p = {p:.4f}")
    ax.grid(alpha=0.25)

    # what the two thresholds look like side by side
    per_cell = 1.96
    fam = float(np.percentile(null_max, 95))
    with np.errstate(invalid="ignore"):
        n_live = int(np.isfinite(real).sum())
        n_naive = int(np.nansum(np.abs(real) > per_cell))
        n_fam = int(np.nansum(np.abs(real) > fam))
    ax2.barh(
        ["|t| > 1.96\n(per-cell 5%)", f"|t| > {fam:.2f}\n(family 5%)"],
        [n_naive, n_fam],
        color=[WARN, GOOD if n_fam else MUTED],
        height=0.55,
    )
    for y, v in enumerate((n_naive, n_fam)):
        ax2.text(v + n_live * 0.01, y, f"{v}", va="center", color="#e6edf3", fontsize=10)
    ax2.axvline(0.05 * n_live, color=MUTED, ls=":", lw=1.2)
    ax2.text(
        0.05 * n_live,
        1.45,
        f"  {0.05 * n_live:.0f} expected by chance",
        color=MUTED,
        fontsize=8,
        va="top",
    )
    ax2.set_xlabel(f"cells passing, of {n_live:,}")
    ax2.set_title("Same data, two thresholds")
    ax2.grid(alpha=0.25, axis="x")
    _save(fig, out)


def fig_surface(real, null, symbols, tfs, labels, out: pathlib.Path) -> None:
    """Every cell, as a picture. If there were structure here it would form a region."""
    n_perm = null.shape[0]
    with np.errstate(invalid="ignore"):
        fam = float(np.percentile(np.nanmax(np.abs(null.reshape(n_perm, -1)), axis=1), 95))
    vmax = max(fam, float(np.nanmax(np.abs(real)))) * 1.02

    fig, axes = plt.subplots(
        len(symbols), len(tfs), figsize=(2.6 * len(tfs), 5.4 * len(symbols)), squeeze=False
    )
    for si, sym in enumerate(symbols):
        for ti, tf in enumerate(tfs):
            ax = axes[si][ti]
            m = real[si, ti]  # (lines, horizons)
            ax.imshow(
                m, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest"
            )
            hit = np.argwhere(np.abs(m) > fam)
            if len(hit):
                ax.scatter(hit[:, 1], hit[:, 0], s=14, facecolors="none", edgecolors="#ffffff")
            ax.set_title(f"{sym} {tf}", fontsize=9)
            ax.set_xticks(range(m.shape[1]))
            ax.set_xticklabels([1, 3, 6, 12, 26, 78], fontsize=6)
            if ti == 0:
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=5)
                ax.set_ylabel("line configuration")
            else:
                ax.set_yticks([])
            if si == len(symbols) - 1:
                ax.set_xlabel("horizon (bars)")
    fig.suptitle(
        f"t-statistic per cell.  Red = continuation, blue = reversion.  "
        f"Circled: |t| > {fam:.2f}, the family 5% threshold.",
        fontsize=10,
        y=0.995,
    )
    _save(fig, out)


def fig_classic(cells: pd.DataFrame, null, out: pathlib.Path) -> None:
    """The five canonical lines, which have no fitted parameters at all."""
    n_perm = null.shape[0]
    with np.errstate(invalid="ignore"):
        fam = float(np.percentile(np.nanmax(np.abs(null.reshape(n_perm, -1)), axis=1), 95))
    sub = cells[cells["line"].isin(CLASSIC)]
    fig, axes = plt.subplots(1, len(CLASSIC), figsize=(3.0 * len(CLASSIC), 4.0), sharey=True)
    for ax, line in zip(axes, CLASSIC, strict=True):
        s = sub[sub["line"] == line]
        x = np.arange(len(s))
        ax.bar(x, s["t"].to_numpy(), color=np.where(s["t"] > 0, REAL, NULL), width=0.85)
        ax.axhline(1.96, color=WARN, lw=1.0, ls="--")
        ax.axhline(-1.96, color=WARN, lw=1.0, ls="--")
        ax.axhline(fam, color=GOOD, lw=1.4)
        ax.axhline(-fam, color=GOOD, lw=1.4)
        ax.set_title(CLASSIC_NAME[line], fontsize=8)
        ax.set_xticks([])
        ax.set_xlabel(f"{len(s)} cells", fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("t-statistic")
    fig.suptitle(
        f"The classical Ichimoku lines, every timeframe and horizon.  "
        f"Dashed amber = 1.96.  Solid green = {fam:.2f}, the threshold once the "
        f"2,250-cell search is paid for.",
        fontsize=9,
    )
    _save(fig, out)


def fig_effect_vs_cost(cells: pd.DataFrame, null, out: pathlib.Path) -> None:
    """Statistical size is not economic size. This is the second gate a cell has to clear."""
    n_perm = null.shape[0]
    with np.errstate(invalid="ignore"):
        fam = float(np.percentile(np.nanmax(np.abs(null.reshape(n_perm, -1)), axis=1), 95))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for sym, mk in (("MES", "o"), ("MNQ", "^")):
        s = cells[cells["symbol"] == sym]
        ax.scatter(
            s["t"].abs(),
            s["d_pts"].abs() / s["cost_pts"],
            s=12,
            marker=mk,
            alpha=0.5,
            edgecolors="none",
            label=sym,
        )
    ax.axhline(1.0, color=WARN, lw=1.3, ls="--")
    ax.axvline(fam, color=GOOD, lw=1.4)
    ax.text(fam, ax.get_ylim()[1] * 0.96, "  family 5% threshold", color=GOOD, fontsize=8, va="top")
    ax.text(
        0.05, 1.08, "one round turn", color=WARN, fontsize=8, transform=ax.get_yaxis_transform()
    )
    ax.set_yscale("log")
    ax.set_xlabel("|t|  (is the effect distinguishable from noise?)")
    ax.set_ylabel("|effect| / round-turn cost  (is it big enough to matter?)")
    ax.set_title("Both gates. A cell has to be right of the green line AND above the amber one.")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/ichimoku_information")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    real, _dd, null, symbols, tfs, _hor, labels = load(d)
    cells = pd.read_csv(d / "cells.csv")
    figs = d / "figures"

    fig_family_null(real, null, figs / "family_null.png")
    fig_surface(real, null, symbols, tfs, labels, figs / "t_surface.png")
    fig_classic(cells, null, figs / "classic_lines.png")
    fig_effect_vs_cost(cells, null, figs / "effect_vs_cost.png")
    print(f"wrote 4 figures to {figs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
