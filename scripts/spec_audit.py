#!/usr/bin/env python3
"""Audit the momentum-framework spec against what this repo actually has.

    python scripts/spec_audit.py

The spec (spec/momentum_framework.csv) is a taxonomy, not a plan: ~90 components across bars,
L2, composites, trade rules and testing. Left as prose it gets read once and half-built. This
maps every row to one of four states so the gap is a fact rather than an impression:

    BUILT       a tested implementation exists in src/quant
    DATA-READY  the data supports it and it has been verified on a sample; no module yet
    MISSING     nothing built, data available
    CONFLICT    the spec asks for something this project has already measured to be wrong,
                or something its own protocol cannot pay for

CONFLICT is the useful column. A spec written without six rejections in front of it will
recommend things we have already priced.
"""

from __future__ import annotations

import csv
import pathlib
from collections import Counter

SPEC = pathlib.Path(__file__).resolve().parents[1] / "spec" / "momentum_framework.csv"
WIDTH = 96

# component -> (state, note). Keyed on the spec's own `component` column.
STATUS: dict[str, tuple[str, str]] = {
    # ------------------------------------------------------------------ BUILT
    "ATR": ("BUILT", "strategy.true_range + information.atr_shifted, shifted one bar"),
    "Unsigned Efficiency Ratio": ("BUILT", "strategy.efficiency_ratio"),
    "Spread": ("BUILT", "costs.measure_spread; measured MES 0.25 / MNQ 0.50 pts"),
    "Net Return After Costs": ("BUILT", "costs.CostModel.for_prop: commission+fees+spread+slip"),
    "Chronological Split": ("BUILT", "splits.Split, WalkForwardPlan; holdout raises on access"),
    "Multiple Testing Control": ("BUILT", "pre-registration + family-max MCPT + pbo.cscv"),
    "Conditional Expectancy": ("BUILT", "information.evaluate -- this IS the information test"),
    "Forward Return": ("BUILT", "information.evaluate, ATR-normalised, 6 horizons"),
    "Maximum Favorable Excursion": ("BUILT", "kama_ema.simulate_1m records mfe per trade"),
    "Maximum Adverse Excursion": ("BUILT", "kama_ema.simulate_1m records mae per trade"),
    "MFE and MAE": ("BUILT", "viz_strategy.mae_mfe"),
    "Trade Count": ("BUILT", "diagnostics.run_all"),
    "Win Rate": ("BUILT", "diagnostics.luck_battery"),
    "Profit Factor": ("BUILT", "pbo.profit_factor, mcpt.pf"),
    "Drawdown": ("BUILT", "diagnostics.luck_battery"),
    "Average Net Return": ("BUILT", "reported by every ladder run"),
    "Median Net Return": ("BUILT", "diagnostics.luck_battery"),
    "Time of Day": ("BUILT", "permutation.session_groups; the null preserves the RTH profile"),
    "Instrument": ("BUILT", "MES and MNQ tested separately under identical split dates"),
    "EMA Cross Only": ("BUILT", "the baseline six hypotheses were measured against"),
    "Observation Unit": ("BUILT", "information.evaluate is one row per eligible bar"),
    # ------------------------------------------------------------------ DATA-READY
    "OFI": ("DATA-READY", "verified: R^2 0.69 at 1s, 0.93 at 30s vs contemporaneous mid"),
    "Signed Trade Flow": ("DATA-READY", "97.2% of trades print at bid or ask; no Lee-Ready needed"),
    "Trade Imbalance": ("DATA-READY", "same source as signed trade flow"),
    "Level 1 Imbalance": ("DATA-READY", "level-1 rows carry best bid/ask SIZE on every update"),
    "Five-Level Imbalance": ("DATA-READY", "level-2 depth 1..10 present"),
    "Ten-Level Imbalance": ("DATA-READY", "level-2 depth 1..10 present"),
    "Cancellation Imbalance": ("DATA-READY", "level-2 action: 0 insert / 1 update / 2 delete"),
    "Depth Depletion": ("DATA-READY", "same action field; 68k/443k/68k per MES day"),
    "Tick Rate": ("DATA-READY", "3.7M MES / 18.8M MNQ level-1 events per day, microsecond UTC"),
    "Tick Distance Per Second": ("DATA-READY", "microsecond stamps confirmed monotonic"),
    "Trade Signing": ("DATA-READY", "quote rule suffices; 2.8% inside/outside need flagging"),
    # ------------------------------------------------------------------ MISSING
    "Signed Efficiency Ratio": ("MISSING", "one line from the existing unsigned version"),
    "Trend Persistence": ("MISSING", "|sum(sign(r))|/N -- cheapest rotation measure in the spec"),
    "Rotation Count": ("MISSING", "sign changes of consecutive returns"),
    "Opposite Bar Fraction": ("MISSING", "bars closing against the net move"),
    "ATR-Normalized Displacement": ("MISSING", "trivial given ATR"),
    "Signed Return": ("MISSING", "trivial"),
    "Variance Ratio": ("MISSING", "Lo-MacKinlay; the ONLY component that carries its own null"),
    "Range Expansion": ("MISSING", "range / rolling median range"),
    "Close Location Value": ("MISSING", "trivial"),
    "Body-to-Range Ratio": ("MISSING", "trivial"),
    "Relative Volume": ("MISSING", "needs the seasonal baseline first"),
    "Volume Z-Score": ("MISSING", "needs the seasonal baseline first"),
    "Speed Z-Score": ("MISSING", "needs the seasonal baseline first"),
    "Seasonality Normalization": ("MISSING", "phase 2; every z-score above depends on it"),
    "Event-Time Bars": ("MISSING", "the reframe: a dead market stops producing bars"),
    "Kyle Lambda": ("MISSING", "regression of price change on signed volume"),
    "Impulse Size": ("MISSING", "needs swing segmentation"),
    "Pullback Size": ("MISSING", "needs swing segmentation"),
    "Pullback Asymmetry": ("MISSING", "impulse magnitude / pullback magnitude"),
    "Pullback Speed": ("MISSING", "pullback speed / impulse speed"),
    "Pullback Flow Confirmation": ("MISSING", "needs swing segmentation plus signed flow"),
    "Pullback Structure": ("MISSING", "volume profiles exist and have never been opened"),
    "Trade Into Major Reference": ("MISSING", "same -- the only non-MA, non-flow idea in the spec"),
    "High-Impact Scheduled Release": ("MISSING", "no economic calendar in the project"),
    "Unstable Book": ("MISSING", "cancellation + spread instability; data ready"),
    "Price and Flow Divergence": ("MISSING", "data ready once OFI is a module"),
    "Low Tick Rate": ("MISSING", "needs the seasonal baseline"),
    "Low Relative Volume": ("MISSING", "needs the seasonal baseline"),
    "Low Signed Efficiency": ("MISSING", "follows from Signed Efficiency Ratio"),
    "Repeated EMA Crosses": ("MISSING", "crossover frequency per unit displacement"),
    "Execution Model": ("MISSING", "queue position and adverse selection not modelled"),
    # ------------------------------------------------------------------ ARCHITECTURE
    # Framing rows and test-plan steps. They describe how the pieces fit, not a thing
    # to build, so counting them as MISSING would overstate the gap.
    "Direction": ("ARCHITECTURE", "framing or a test-plan step, not an implementable component"),
    "Efficiency": ("ARCHITECTURE", "framing or a test-plan step, not an implementable component"),
    "Speed": ("ARCHITECTURE", "framing or a test-plan step, not an implementable component"),
    "Participation": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Liquidity State": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Regime": ("ARCHITECTURE", "framing or a test-plan step, not an implementable component"),
    "Higher-Timeframe Context": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Regime Filter": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Entry Trigger": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "EMA Role": ("ARCHITECTURE", "framing or a test-plan step, not an implementable component"),
    "Final Trade Gate": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Long Permission": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Short Permission": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Long Entry Trigger": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Short Entry Trigger": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Long Pullback Rule": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Short Pullback Rule": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Signal Candidate": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Efficiency Filter": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Speed Filter": ("ARCHITECTURE", "framing or a test-plan step, not an implementable component"),
    "Flow Filter": ("ARCHITECTURE", "framing or a test-plan step, not an implementable component"),
    "Liquidity Filter": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Bar-Only Prototype": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Event-Time Data": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "OFI and Book Features": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Momentum Definition": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Best Initial Signal": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    "Best Advanced Signal": (
        "ARCHITECTURE",
        "framing or a test-plan step, not an implementable component",
    ),
    # ------------------------------------------------------------------ CONFLICT
    "EMA Cross Limitation": (
        "CONFLICT",
        "The spec is RIGHT and does not know it. We measured the cross across 6,060 "
        "configurations: it carries nothing. But 'EMA STATE as permission' is a different and "
        "still-untested claim, and the spec never separates the two in its testing plan.",
    ),
    "Example Bar Score": (
        "CONFLICT",
        "Weights 0.30/0.25/0.20/0.15/0.10 are invented. Fitting 5 weights is a CONTINUOUS "
        "search; our PBO reached 0.377 on a 45-cell DISCRETE grid. A permutation test cannot "
        "charge for a simplex.",
    ),
    "Bar Momentum Score": ("CONFLICT", "same weighting problem"),
    "L2 Momentum Score": ("CONFLICT", "same, plus an undefined 'adverse liquidity penalty'"),
    "Directional Momentum State Score": (
        "CONFLICT",
        "Combines three weighted sub-scores. Unfittable within this protocol.",
    ),
    "Long Score": ("CONFLICT", "composite of composites"),
    "Short Score": ("CONFLICT", "composite of composites"),
    "Composite Model": (
        "CONFLICT",
        "Phase 6 says 'do not overfit' without saying how it would be detected. That is the "
        "step where all six hypotheses died.",
    ),
    "Long Regime Filter": ("CONFLICT", "'exceeds tested bullish threshold' -- a searched value"),
    "Short Regime Filter": ("CONFLICT", "'exceeds tested bearish threshold' -- a searched value"),
}

PROTOCOL_GAPS = [
    (
        "No permutation null anywhere in ~90 rows",
        "The Testing Plan validates with walk-forward and a holdout. Neither tells you what a "
        "NOISE strategy scores on the same data. Walk-forward on noise still produces a number "
        "and an equity curve -- the MNQ 10m cell that nominally passed at p=0.042 had a "
        "beautiful walk-forward shape and reversed sign out of sample.",
    ),
    (
        "No matched control is required",
        "The spec never says 'compare against a benchmark that should not work'. The EMA control "
        "is exactly what killed HMA/TEMA/VWAP: all four sub-families scored 9%.",
    ),
    (
        "Multiple Testing Control is 1 row of ~90",
        "It is where every rejection actually happened. The spec spends sixty rows on features "
        "and one line on the thing that decides whether any of them are real.",
    ),
    (
        "Thresholds are never counted",
        "Almost every row implies a lookback N and a threshold. Nothing in the spec accounts "
        "for the size of that search, and the composite rows add continuous weights on top.",
    ),
    (
        "Contemporaneous and predictive are not distinguished",
        "OFI is listed as the primary microstructure feature. Measured here it explains 93% of "
        "the SAME-interval mid change and ~0% of the NEXT one. It is a description of the move "
        "in progress, not a forecast, and the spec's wording invites the wrong use.",
    ),
]


def wrap(t: str, indent: str = "    ", width: int = WIDTH - 8) -> list[str]:
    out, line = [], ""
    for w in t.split():
        if len(line) + len(w) + 1 > width:
            out.append(indent + line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(indent + line)
    return out


def main() -> int:
    if not SPEC.exists():
        print(f"spec not found: {SPEC}")
        return 1
    rows = [r for r in csv.DictReader(SPEC.open(encoding="utf-8")) if r.get("component")]

    counts: Counter = Counter()
    by_state: dict[str, list[tuple[str, str]]] = {}
    unknown: list[str] = []
    for r in rows:
        comp = r["component"].strip()
        state, note = STATUS.get(comp, ("MISSING", "unclassified"))
        if comp not in STATUS:
            unknown.append(comp)
        counts[state] += 1
        by_state.setdefault(state, []).append((comp, note))

    print("=" * WIDTH)
    print("  MOMENTUM FRAMEWORK SPEC vs THIS REPO")
    print("=" * WIDTH)
    n_sec = len({r["section"] for r in rows})
    print(f"  {len(rows)} components across {n_sec} sections\n")
    for state in ("BUILT", "DATA-READY", "ARCHITECTURE", "MISSING", "CONFLICT"):
        n = counts.get(state, 0)
        print(f"  {state:<12}{n:>4}   {'#' * int(50 * n / max(len(rows), 1))}")
    print()

    for state in ("CONFLICT", "DATA-READY", "MISSING", "BUILT", "ARCHITECTURE"):
        items = sorted(set(by_state.get(state, [])))
        if not items:
            continue
        print("-" * WIDTH)
        print(f"  {state}  ({len(items)} distinct)")
        print("-" * WIDTH)
        for comp, note in items:
            if len(note) < 70:
                print(f"  {comp:<32}{note}")
            else:
                print(f"  {comp}")
                for line in wrap(note, indent="      "):
                    print(line)
        print()

    print("=" * WIDTH)
    print("  WHERE THE SPEC IS WEAKER THAN THIS PROJECT'S PROTOCOL")
    print("=" * WIDTH)
    for title, why in PROTOCOL_GAPS:
        print(f"\n  {title}")
        for line in wrap(why):
            print(line)
    print()
    if unknown:
        print(
            f"  {len(set(unknown))} components fell through to MISSING unclassified: "
            f"{', '.join(sorted(set(unknown)))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
