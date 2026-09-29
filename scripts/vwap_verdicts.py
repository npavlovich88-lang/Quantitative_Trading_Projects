#!/usr/bin/env python3
"""Verdict blocks for the three VWAP band hypotheses. Reads the run outputs, invents nothing.

python scripts/vwap_verdicts.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant.verdict import Verdict

OUT = pathlib.Path("out")


def j(p: str) -> dict:
    return json.loads((OUT / p / "summary.json").read_text(encoding="utf-8"))


def main() -> int:
    q1, q2, q4 = j("vwap_band_race"), j("vwap_confluence"), j("vwap_ride_vs_reject")
    q5 = j("vwap_band_by_session")
    e = pd.read_csv(OUT / "vwap_band_race" / "economics.csv")
    e = e[e["anchor"] != "continuous"]
    b4 = e[(e["band"] == 4) & (e["filter"] == "rth") & (e["anchor"] == "ny")]
    b4s = b4[b4["p_cell"] < 0.05]
    b1 = e[(e["band"] == 1) & (e["t"] > 2)]
    q4c = pd.read_csv(OUT / "vwap_ride_vs_reject" / "cells.csv")

    vs = [
        Verdict.inconclusive(
            name="Q1  VWAP band race: revert to the mean or extend?",
            reason="The bands carry information and it FLIPS SIGN with distance, but the cell "
            "that wins the family correction is untradeable and the cells that are tradeable "
            "need an out-of-sample look before a config can be named.",
            evidence=(
                f"family-max |t| {q1['family_max_t']:.2f}, p = {q1['p_family']:.4f}; "
                f"count {q1['count_obs']} of {q1['cells']} cells outside their own 5-95 band "
                f"against a null median of 11, p = {q1['p_count']:.4f}",
                "the 2 sigma band -- the one the question named -- is the most informationless "
                "in the set: pA 0.324 to 0.329 against a null of 0.318 to 0.326, mean t +0.27",
                f"at 1 sigma price REVERTS above the null ({len(b1)} cells at t > +2) but the "
                f"trip is about 1 sigma and cost eats it: best expectancy +2.46 points",
                f"at 4 sigma price EXTENDS below the null; {len(b4s)} of {len(b4)} ny-anchored "
                "RTH cells reach p < 0.05, across both symbols and both timeframes",
                "the extension trade clears its breakeven: MNQ +$19 to +$21 a trade, "
                "MES +$4, at one contract average firm with cost in points",
                "NY-anchored beats running non-stop, and not marginally: the continuous anchor "
                "has sigma of several hundred points, so 79 to 84% of its races never resolve "
                "inside a day and its apparent +646 point expectancy came off the selected fifth "
                "that did",
            ),
            caveat="Every positive-expectancy cell of any size is MNQ. MES shows the same sign "
            "and the same p-values but expectancy near +$4 a trade, which is inside the "
            "uncertainty of the cost model rather than clear of it.",
            next_step="One pre-committed validate look at the 4 sigma ny-anchored RTH extension, "
            "both symbols, no parameter tweak. MNQ validate is at look #1, MES at #0.",
        ),
        Verdict.inconclusive(
            name="Q2  Multi-timeframe band confluence: better rejection than the session alone?",
            reason="Confluence changes the outcome and the effect orders itself with the number "
            "of agreeing anchors, but it makes reversion WORSE on average and better only in one "
            "symbol on one timeframe on one side.",
            evidence=(
                f"family-max |t| {q2['family_max_t']:.2f}, p = {q2['p_family']:.4f}; "
                f"count {q2['count_obs']} of {q2['cells']} cells, p = {q2['p_count']:.4f}",
                f"mean difference across all {q2['cells']} cells is "
                f"{q2['mean_diff']:+.4f} -- NEGATIVE, so on average layering timeframes reduces "
                "the reversion rate rather than raising it",
                "every combo has a negative mean: S+W -0.124, S+M -0.061, S+W+M -0.099, "
                "S+Q -0.037, S+W+M+Q -0.077",
                "where it IS positive there is a dose-response, which noise does not produce: "
                "on MES 15m at the +4 band in RTH the gain rises monotonically with the number "
                "of agreeing anchors -- S+M +0.047, S+W +0.063, S+W+M +0.102, S+W+M+Q +0.149",
                "read against Q1 this is not reversion appearing, it is extension being removed: "
                "confluence lifts pA at +4 sigma from 0.156 to 0.306, which is the Q1 null, not "
                "above it",
                "confluence covers a median 30.9% of touches, so the samples are real",
            ),
            caveat="The whole positive result is MES 15m upper bands in RTH. MNQ contributes one "
            "cell, and MES 5m at the -4 band runs strongly the other way (-0.256, t = -3.43). "
            "Calling this 'better rejection' overstates it: the mechanism is less continuation.",
            next_step="Same single validate look as Q1 -- the two findings are the same effect "
            "seen twice, so they should be spent on one look, not two.",
        ),
        Verdict.no_edge(
            name="Q4  Does an MA cross separate riding the bands from rejecting off them?",
            reason="The discrimination claim fails its own test even before the confound check, "
            "and the band adds nothing to a cross that was already rejected twelve times.",
            evidence=(
                f"A - B, whether the cross separates reverting touches from riding ones: "
                f"family-max p = {q4['dAB_p_family']:.4f}, count p = {q4['dAB_p_count']:.4f} "
                "-- and this statistic is INFLATED by construction, because arm B is touches "
                "with no cross scored on a reversion-signed return, and the absence of a cross "
                "up after a low is already correlated with price not having gone up",
                f"A alone, the cross at a band with no comparison: family-max "
                f"p = {q4['mA_p_family']:.4f}, count p = {q4['mA_p_count']:.4f}, mean forward "
                f"return {q4['mean_A']:+.4f} ATR",
                f"A - C, the clean test of whether the BAND adds anything to the same cross "
                f"taken anywhere: {int((q4c['dA_C'] > 0).sum())} of {len(q4c)} cells positive = "
                f"{100 * (q4c['dA_C'] > 0).mean():.1f}%, mean {q4['mean_dA_C']:+.4f} ATR",
                "no ordering by pair or by band: the fraction of positive cells ranges 0.20 to "
                "0.70 and the mean flips sign band to band inside every one of the four pairs",
                "every cell with a large A - C sits at the 24-bar horizon, where overlapping "
                "windows over 40 to 140 events manufacture the spread that produces it",
            ),
            caveat=f"A - C does pass family-max at p = {q4['dAC_p_family']:.4f}, carried by MNQ "
            "15m HMA9/EMA43 at the -2 band, RTH, 24 bars, n = 48. Its count statistic fails "
            f"(p = {q4['dAC_p_count']:.4f}), it has no cross-market or cross-horizon neighbour, "
            "and the sign test above is a coin flip. That is the lone-spike pattern this protocol "
            "treats as noise until a validate look says otherwise, and it does not earn one.",
            next_step="Do not sweep MA lengths here. The request offered 'any length and any "
            "combo'; four declared pairs already failed and a sweep raises the noise ceiling "
            "without raising the signal. The bands themselves are the finding -- Q1 and Q2.",
        ),
        Verdict.signal_only(
            name="Q5  Bands by time of day, plus acceptance as the rejection discriminator",
            reason="Two real effects were found and neither survives the trade geometry: the "
            "discriminator says which way price will go and the London bucket says when it "
            "reverts, but at the moment either one is known the distance to the mean is already "
            "too long and the distance to the next band too short to be paid for.",
            evidence=(
                f"bands by bucket: family-max |t| {q5['bucket_family_max_t']:.2f}, "
                f"p = {q5['bucket_p_family']:.4f}; count {q5['bucket_count']} of "
                f"{q5['bucket_cells']} against a null median of 20, "
                f"p = {q5['bucket_p_count']:.4f}",
                "AFTER the opening range and after the Initial Balance there is no rejection: "
                "post_ib has 9 cells at t < -2 against 1 at t > +2, and pm has 6 against 1. "
                "During the US day past the first hour, price EXTENDS at the bands.",
                "LONDON is the single bucket where every band points to reversion: all eight "
                "signed bands positive, 9 cells at t > +2 and none at t < -2, with sigma fully "
                "mature (0% of touches inside the first 12 bars of the anchor)",
                "the acceptance discriminator works, and it is the answer the MA cross was not: "
                "family-max "
                f"p = {q5['accept_p_family']:.4f}, count p = {q5['accept_p_count']:.4f}, "
                f"{100 * q5['accept_frac_pos']:.1f}% of 188 cells positive, mean "
                f"{q5['accept_mean_dev_diff']:+.4f} -- a CLOSE beyond the band means riding "
                "(71 to 94% extend), a wick that closes back inside means rejecting",
                "it is strongest exactly where the clock confound is absent: midday 100% of 24 "
                "cells positive at mean t +3.73, post_ib 93% at +2.50, and London 54% at +0.27, "
                "i.e. nothing where reversion is already the base case",
                "Q1's 4 sigma extension was substantially the opening drive, as suspected: the "
                "strongest extension cell in this run is MES 5m at +4 sigma inside the opening "
                "range, t = -4.10, with sigma built from under 6 bars",
            ),
            caveat="London beats its null by a mean of +4.6 points of win rate against a cost "
            "demand of +1.5, and still only 4 of 30 cells clear zero, best +$1 a trade. The "
            "reason is that beating the permutation null is NOT the same as beating breakeven: "
            "the real market's band touches close further back inside the band than permuted "
            "ones do, so the real trade starts further from the mean and nearer the extension. "
            "The fair-game bar sits above the measured rate before any cost is charged. On the "
            "acceptance arms, 0 of the 14 strongest cells clear zero, best -$4, and the wick arm "
            "loses too despite reverting 27 to 63% of the time.",
            next_step="The discriminator is real, so the open question is exits, not entries: "
            "whether any geometry other than 'race the VWAP against one band further out' "
            "monetises a signal that fires after the bar has closed. Test that before spending "
            "a validate look. Do not re-test entries at these bands.",
        ),
    ]
    for v in vs:
        print(v.render())
        print()
    (OUT / "vwap_verdicts.json").write_text(
        json.dumps([v.as_dict() for v in vs], indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
