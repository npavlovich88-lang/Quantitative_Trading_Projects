"""The backtest protocol: the only sanctioned way a result leaves this repo.

It runs the ladder in order and STOPS at the first rung that fails. That is the whole design.
A strategy that fails rung 2 does not get a walk-forward, does not get a prop-firm pass rate,
and does not get a number quoted at it -- because every downstream figure computed on an
unvalidated edge is a precise statement about nothing. This project has already produced one
set of those.

    rung 0  PRE-REGISTER   the split and the grid are declared before anything runs
    rung 1  IN-SAMPLE      is it even good on the data we fitted it to?
    rung 2  SELECTION      PBO over EVERY trial in the register (not just the final grid)
    rung 3  LUCK           in-sample MCPT: could noise have done this?
    rung 4  WALK-FORWARD   re-optimise on a trailing window, trade the next block
    rung 5  WF MCPT        permute everything after the first fold, re-run the whole thing
    rung 6  BIAS BATTERY   coverage, regime, roll, costs, concentration, trade dependence

The holdout is not touched by any rung. It is spent once, deliberately, by a human decision,
after everything above has passed -- and `Split` makes that a logged event.

Thresholds, and where they come from:
    in-sample MCPT   p < 0.01                  neurotrader, NLBXgSmRBgU
    walk-forward     p < 0.05 (1y) / 0.01 (2y+)  same
    PBO              <= 0.05                   Bailey et al. 2017, section 3.1, verbatim
They are set here, in code, BEFORE a result is seen. Changing one after looking is how a
threshold becomes a formality.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

from . import diagnostics as diag
from . import pbo as pbo_mod
from . import viz
from .register import TrialRegister
from .splits import Split

P_INSAMPLE = 0.01
P_WALKFORWARD = 0.05
PBO_MAX = 0.05


@dataclasses.dataclass
class Rung:
    name: str
    verdict: str  # "pass" | "fail" | "warn" | "skipped"
    detail: str
    data: dict = dataclasses.field(default_factory=dict)
    figures: list[str] = dataclasses.field(default_factory=list)


class Protocol:
    """Accumulates rungs, stops on the first failure, writes a report and figures."""

    def __init__(
        self, name: str, out_dir: str | pathlib.Path, split: Split, register: TrialRegister
    ):
        self.name = name
        self.out = pathlib.Path(out_dir)
        self.fig_dir = self.out / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.register = register
        self.rungs: list[Rung] = []
        self._halted = False

    # ---------------------------------------------------------------- mechanics
    def add(self, rung: Rung) -> Protocol:
        self.rungs.append(rung)
        if rung.verdict == "fail":
            self._halted = True
        return self

    @property
    def halted(self) -> bool:
        return self._halted

    def skip_rest(self, *names: str) -> None:
        reason = next((r.name for r in self.rungs if r.verdict == "fail"), "an earlier rung")
        for n in names:
            self.rungs.append(Rung(n, "skipped", f"not run: {reason} failed"))

    # ---------------------------------------------------------------- rungs
    def rung_insample(self, metric: float, n_trades: int, *, min_metric: float = 1.0) -> Protocol:
        ok = np.isfinite(metric) and metric > min_metric and n_trades >= 30
        return self.add(
            Rung(
                "1. in-sample quality",
                "pass" if ok else "fail",
                f"PF {metric:.4f} on {n_trades} trades (need > {min_metric} and >= 30 trades). "
                + ("Marginal in sample means nothing left to lose to reality." if not ok else ""),
                dict(metric=metric, n_trades=n_trades),
            )
        )

    def rung_selection(self, segment: str, *, S: int = 10) -> Protocol:
        """PBO over the FULL register -- every configuration ever evaluated on this segment."""
        m, _cfgs = self.register.matrix(segment)
        n_reg = self.register.n_trials(segment)
        if m.size == 0 or m.shape[1] < 2:
            return self.add(
                Rung(
                    "2. selection bias (PBO)",
                    "skipped",
                    f"register holds {n_reg} trials but < 2 with stored P&L",
                )
            )
        res = pbo_mod.cscv(m, S=S)
        fig = viz.pbo_panel(
            res, self.fig_dir / "pbo.png", title=f"PBO - {m.shape[1]} of {n_reg} registered trials"
        )
        ok = res["pbo"] <= PBO_MAX
        # a flat grid inflates PBO; say so rather than failing a plateau by accident
        flat = res["pbo"] > PBO_MAX and abs(res["degradation_slope"]) < 0.2
        return self.add(
            Rung(
                "2. selection bias (PBO)",
                "pass" if ok else ("warn" if flat else "fail"),
                f"PBO {res['pbo']:.3f} (reject above {PBO_MAX}); degradation slope "
                f"{res['degradation_slope']:+.3f}; P(OOS<0) {res['prob_oos_loss']:.3f}"
                + (
                    "  -- but the surface is flat, so PBO is inflated by configurations being hard "
                    "to tell apart rather than by overfitting. Read with the MCPT p-value."
                    if flat
                    else ""
                ),
                res,
                [str(fig)],
            )
        )

    def rung_mcpt(
        self,
        real: float,
        perms,
        p_value: float,
        *,
        label="3. luck (in-sample MCPT)",
        threshold: float = P_INSAMPLE,
        metric="profit factor",
    ) -> Protocol:
        fig = viz.mcpt_histogram(
            perms, real, p_value, self.fig_dir / f"mcpt_{label[0]}.png", metric=metric, title=label
        )
        ok = p_value < threshold
        return self.add(
            Rung(
                label,
                "pass" if ok else "fail",
                f"p = {p_value:.4f} over {len(perms)} permutations (need < {threshold}); "
                f"real {real:.4f} vs permutation median "
                f"{np.nanmedian(perms):.4f}"
                if len(perms)
                else f"p = {p_value:.4f}",
                dict(real=real, p_value=p_value, n_perm=len(perms)),
                [str(fig)],
            )
        )

    def rung_walkforward(self, bar_pnl, metric: float, *, dates=None) -> Protocol:
        figs = [
            str(
                viz.equity_with_splits(
                    bar_pnl,
                    self.split,
                    self.fig_dir / "equity.png",
                    dates=dates,
                    title=f"{self.name} walk-forward",
                )
            ),
            str(viz.drawdown(bar_pnl, self.fig_dir / "drawdown.png")),
        ]
        ok = np.isfinite(metric) and metric > 1.0
        return self.add(
            Rung(
                "4. walk-forward",
                "pass" if ok else "fail",
                f"out-of-sample PF {metric:.4f} (need > 1.0)",
                dict(metric=metric),
                figs,
            )
        )

    def rung_bias(self, **kw) -> Protocol:
        checks = diag.run_all(**kw)
        figs = [
            str(
                viz.verdict_card(
                    checks, self.fig_dir / "diagnostics.png", title=f"{self.name} - bias battery"
                )
            )
        ]
        if "costs" in checks:
            figs.append(str(viz.cost_curve(checks["costs"], self.fig_dir / "costs.png")))
        if "regime" in checks:
            figs.append(str(viz.regime_bars(checks["regime"], self.fig_dir / "regime.png")))
        if kw.get("surface"):
            figs.append(str(viz.parameter_surface(kw["surface"], self.fig_dir / "surface.png")))
        figs.append(
            str(viz.trade_distribution(kw["pnl"], self.fig_dir / "trades.png", luck=checks["luck"]))
        )
        s = checks["_summary"]
        return self.add(
            Rung(
                "6. bias battery",
                s["verdict"],
                f"{len(s['failed'])} failed, {len(s['warned'])} warned"
                + (f"; failed: {', '.join(s['failed'])}" if s["failed"] else "")
                + (f"; warned: {', '.join(s['warned'])}" if s["warned"] else ""),
                checks,
                figs,
            )
        )

    # ---------------------------------------------------------------- output
    def verdict(self) -> str:
        if any(r.verdict == "fail" for r in self.rungs):
            return "REJECTED"
        if any(r.verdict == "skipped" for r in self.rungs):
            return "INCOMPLETE"
        if any(r.verdict == "warn" for r in self.rungs):
            return "PASSED WITH WARNINGS"
        return "PASSED"

    def report(self) -> str:
        v = self.verdict()
        w = 78
        lines = [
            "=" * w,
            f"  {self.name}   ->   {v}",
            "=" * w,
            "",
            self.split.describe(),
            "",
            self.register.summary(),
            "",
            "-" * w,
        ]
        mark = {"pass": "[PASS]", "fail": "[FAIL]", "warn": "[WARN]", "skipped": "[ -- ]"}
        for r in self.rungs:
            lines.append(f"{mark[r.verdict]}  {r.name}")
            for chunk in _wrap(r.detail, w - 10):
                lines.append(f"         {chunk}")
            for f in r.figures:
                lines.append(f"         -> {pathlib.Path(f).name}")
            lines.append("")
        lines.append("-" * w)
        if v == "REJECTED":
            lines += [
                "  This strategy has NOT been shown to have an edge. Do not quote a pass rate,",
                "  an expectancy or a Sharpe from it. The holdout remains unspent, which is the",
                "  one good outcome here.",
            ]
        elif v == "INCOMPLETE":
            lines.append("  Ladder not finished. No conclusion is available yet.")
        else:
            h = self.split.holdout_bounds
            lines += [
                f"  Every rung passed on train/validate. The holdout ({h[1] - h[0]} bars) is still",
                "  frozen. Spending it is a separate, deliberate decision -- and it is ONE look.",
            ]
        lines.append("=" * w)
        return "\n".join(lines)

    def save(self) -> pathlib.Path:
        (self.out / "report.txt").write_text(self.report(), encoding="utf-8")
        payload = dict(
            name=self.name,
            verdict=self.verdict(),
            split=dict(
                n=self.split.n,
                train_end=self.split.train_end,
                validate_end=self.split.validate_end,
                embargo=self.split.embargo,
                holdout_looks=len(self.split.holdout_looks()),
            ),
            thresholds=dict(insample_mcpt=P_INSAMPLE, walkforward_mcpt=P_WALKFORWARD, pbo=PBO_MAX),
            rungs=[dataclasses.asdict(r) for r in self.rungs],
        )
        (self.out / "result.json").write_text(
            json.dumps(payload, indent=1, default=_jsonable), encoding="utf-8"
        )
        return self.out


def _jsonable(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for word in str(text).split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out or [""]
