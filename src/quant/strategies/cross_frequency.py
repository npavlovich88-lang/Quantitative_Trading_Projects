"""Recent crossing frequency as the regime filter. The user's own observation, formalised.

THE OBSERVATION

    "we have a cross ... but there is no real momentum in the market to keep this energy going
     ... we end up seeing candlesticks just cross the ema over and over"
    "we aren't crossing the emas over and over and we are still showing willingness to go lower"

Two charts, one bullish cross that went nowhere and one bearish cross that ran. The stated
difference is not how far apart the averages were, and not how the individual candles looked.
It is HOW OFTEN THE PAIR HAS BEEN FLIPPING.

WHY THIS IS A NEW VARIABLE AND NOT A RELABELLED REJECTION

    "emas can be spread far apart even if the market is in a chopy regieme"

That sentence is the whole point, and it is correct. Hypothesis 3 tested SEPARATION, which is a
level at one instant. This is a COUNT OVER HISTORY. A pair can be wide right now and have
flipped six times in the last hour, and separation cannot see that.

Measured on the train segment before building anything, crossing frequency correlates:

    vs separation (hypothesis 3)      -0.05 to -0.07
    vs |displacement| (hypothesis 7)  -0.12 to -0.18
    vs rotation count (hypothesis 7)  -0.01 to -0.04

while separation and |displacement| correlate 0.58 to 0.80 with EACH OTHER -- so those two
rejections were largely one rejection, and this is genuinely orthogonal to both. Its spread is
usable: 2 crosses per 100 bars at the 10th percentile, 5 at the median, 8 at the 90th.

It also is not the rotation count already found to be at chance. That counts sign changes of
bar-to-bar returns: micro, noisy, and a different object from how often a pair of smoothed
averages changes sides.

WHAT IS MEASURED

At each cross, the number of crosses in the PREVIOUS K bars, shifted so the current one is
excluded. Crosses are then split into the low-frequency and high-frequency terciles of that
count, and the forward outcome of the two groups is compared. A positive t means quiet-regime
crosses do better, which is the claim.

Three outcomes, not one: forward return, maximum favourable excursion and maximum adverse
excursion. The question is whether the move GIVES A RUN inside a 5-15 minute window, and that is
MFE, not a mean. All three are oriented so higher is better for the trader.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from ..strategy import ema

EMA_PAIRS = ((8, 21), (10, 29), (20, 50), (50, 200))
COUNT_WINDOWS = (50, 100, 200)
OUTCOMES = ("ret", "mfe", "mae")
CALIB_FRAC = 0.20


@dataclasses.dataclass(frozen=True)
class FreqSpec:
    """One configuration: which EMA pair, over how many bars the crosses are counted."""

    fast: int
    slow: int
    window: int
    outcome: str

    @property
    def label(self) -> str:
        return f"ema{self.fast}-{self.slow}/K{self.window}/{self.outcome}"

    @property
    def family(self) -> str:
        return self.outcome


def freq_family() -> list[FreqSpec]:
    """4 EMA pairs x 3 count windows x 3 outcome labels = 36 configurations."""
    return [FreqSpec(f, s, k, o) for f, s in EMA_PAIRS for k in COUNT_WINDOWS for o in OUTCOMES]


def cross_state(close: np.ndarray, fast: int, slow: int) -> tuple[np.ndarray, np.ndarray]:
    """(cross events, direction at each bar). Direction is +1 while fast is above slow."""
    f, s = ema(close, fast), ema(close, slow)
    above = f > s
    ok = np.isfinite(f) & np.isfinite(s)
    cross = np.r_[False, above[1:] != above[:-1]] & ok & np.r_[False, ok[:-1]]
    return cross, np.where(above, 1.0, -1.0)


def cross_frequency(cross: np.ndarray, window: int) -> np.ndarray:
    """Crosses in the PREVIOUS `window` bars.

    Shifted one bar so the cross being evaluated never counts itself -- without the shift the
    variable would contain its own event and every measured cross would carry a guaranteed +1.
    """
    return pd.Series(cross.astype(float)).rolling(window).sum().shift(1).to_numpy()


def build_masks(
    close: np.ndarray,
    specs: list[FreqSpec],
    calib: slice,
    eligible: np.ndarray,
) -> tuple[dict[tuple[int, int], np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """(direction per EMA pair, one (low-frequency, high-frequency) cross-event pair per spec).

    Terciles come from the calibration slice only. Taken over the full sample they would label a
    cross using the frequency distribution of years that had not happened yet.
    """
    dirs: dict[tuple[int, int], np.ndarray] = {}
    crosses: dict[tuple[int, int], np.ndarray] = {}
    for f, s in {(sp.fast, sp.slow) for sp in specs}:
        crosses[(f, s)], dirs[(f, s)] = cross_state(close, f, s)

    freqs: dict[tuple[int, int, int], np.ndarray] = {}
    for sp in specs:
        key = (sp.fast, sp.slow, sp.window)
        if key not in freqs:
            freqs[key] = cross_frequency(crosses[(sp.fast, sp.slow)], sp.window)

    masks: list[tuple[np.ndarray, np.ndarray]] = []
    for sp in specs:
        cr = crosses[(sp.fast, sp.slow)]
        fq = freqs[(sp.fast, sp.slow, sp.window)]
        pool = fq[calib][cr[calib] & np.isfinite(fq[calib])]
        if len(pool) < 150:
            empty = np.zeros(len(close), bool)
            masks.append((empty, empty))
            continue
        lo_t, hi_t = np.quantile(pool, [1 / 3, 2 / 3])
        base = cr & eligible & np.isfinite(fq)
        # LOW frequency first, so a positive t means "a quieter regime traded better".
        masks.append((base & (fq <= lo_t), base & (fq >= hi_t)))
    return dirs, masks
