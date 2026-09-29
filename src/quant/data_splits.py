"""THE DECLARED SPLITS. Pre-registered 2026-09-22, before any evaluation was run against them.

This file is the contract. Changing a boundary after seeing a result is the thing the whole
protocol exists to prevent, so a change here needs a dated entry in the vault's Decision Log
saying what moved and why. Moving a boundary to improve a number is not a why.

===============================================================================
ONE RULE, BOTH SYMBOLS
===============================================================================

MES and MNQ are the SAME Level 2 MarketTick data. Same source, same format, same quality.
Neither is the better dataset and neither is "primary" -- an earlier version of this file
claimed first one and then the other, and both claims were unsupported. The only difference
between them is the span each one happens to cover:

    MNQ   2021-09-01 -> 2026-09-01   1,203 trading days, with a 396-day hole after 2023-05-31
    MES   2020-01-01 -> 2026-06-29   2,018 trading days, largest gap 14 days

MES shares 1,123 trading days with MNQ and covers 340 of the days inside MNQ's hole, which is
what makes the missing year answerable instead of permanently ambiguous.

So the split is ONE rule with the SAME boundary dates for both symbols. The only per-symbol
difference is where each series starts and stops:

    train      the symbol's first bar  ->  2024-07-01
    validate   2024-07-01              ->  2025-09-01
    holdout    2025-09-01              ->  the symbol's last bar          LOCKED

Identical boundaries are what make a cross-market result mean anything. Split MES on different
dates from MNQ and agreement between them could just be the two windows covering different
markets-in-time, while disagreement would be uninterpretable. Same dates, different instrument
is a confirmation test. Different dates as well is two unrelated numbers.

Consequences of each symbol's own span, stated so they are not mistaken for design choices:

  * MES train is longer (2020-01 -> 2024-06) and covers both the period before MNQ starts and
    the 396-day window MNQ is missing. A property of the data, not a thumb on the scale.
  * MNQ train is truncated by its hole and effectively ends 2023-05-31. The hole then sits
    inside the train/validate boundary, making it a 396-day natural embargo that no trade,
    label or indicator window can straddle. Useful, but an accident.
  * Holdouts end on different days (MNQ 2026-08-31, MES 2026-06-28) because the feeds stop at
    different points. Both are roughly twelve months.

===============================================================================
PRIOR EXPOSURE -- stated plainly, because a quiet holdout is a fake one
===============================================================================

The walk-forward smoke test of 2026-09-21 used train_bars=52000 on MNQ 10m and therefore
traversed every bar from 2023-03-10 onward, including all of the declared MNQ holdout. What was
observed: an out-of-sample profit factor of 1.144, and the optimiser flipping its chosen exit
shape from stop2/tp6 to stop6/tp1-2 partway through. No configuration was selected on it.

So the MNQ holdout is LIGHTLY EXPOSED, not virgin. The ledger records this as look #1, so the
next look is #2 and the split will correctly report itself spent. MES holdouts are untouched.

===============================================================================
MES BARS ARE NOT BUILT YET -- a build task, not a data problem
===============================================================================

Only 129 days of MES exist as OHLCV (2020-01-01..2020-05-31), because the builder takes a
month-scope CLI argument and was run once as a five-month test:

    MES: 140 day files to process (scoped to 202001,202002,202003,202004,202005)

Nothing failed. The serial builder took 3.7 hours for those 129 days (~103 s/day), which makes
the full job look like a 47-hour commitment; build_mnq_bars_parallel.py does MNQ at ~10 s/day.
The 8 logged "errors" are every Saturday in the range -- a reduced 5-column format carrying
271 bytes against a weekday's 90 MB. Dropping them is correct.

The boundaries below do not change when those bars arrive. Only the number of rows behind them
does.
"""

from __future__ import annotations

import os
import pathlib

from .splits import Split, WalkForwardPlan

LEDGER = pathlib.Path(__file__).resolve().parents[2] / "out" / "holdout_ledger.jsonl"

# Market data lives outside the repository -- it is large, licensed, and not redistributable.
# Point QUANT_DATA_ROOT at the directory holding the per-symbol folders; everything below is
# relative to it, so nothing here depends on one machine's drive letters.
DATA_ROOT = pathlib.Path(os.environ.get("QUANT_DATA_ROOT", "./data")).expanduser()

DATA = {
    "MNQ": str(DATA_ROOT / "MNQ_OHLCV" / "MNQ_{tf}_full_session_ohlcv.csv"),
    "MES": str(DATA_ROOT / "MES_OHLCV" / "MES_{tf}_full_session_ohlcv.csv"),
}
RAW = {
    "MNQ": str(DATA_ROOT / "Future_MNQ_T2"),
    "MES": str(DATA_ROOT / "Future_MES_T2"),
}

# ONE rule, both symbols. Dates, not fractions -- a fraction moves silently when the file grows.
TRAIN_END = "2024-07-01"
VALIDATE_END = "2025-09-01"

# Raw tick coverage, measured 2026-09-22. Recorded so a future session can tell the difference
# between "the bars are thin" and "the data is thin". They are not the same problem, and
# conflating them is the mistake this file already made once.
RAW_SPAN = {
    "MNQ": ("2021-09-01", "2026-08-31", 1308),
    "MES": ("2020-01-01", "2026-06-28", 2169),
}
OVERLAP = ("2021-09-01", "2026-06-28")
MNQ_HOLE = ("2023-05-31", "2024-06-30", 396)  # permanent: absent from the raw archives too

# One full RTH session in bars, per timeframe. The embargo must cover the longest holding
# period; the strategy is flat at every RTH close, so one session is the bound.
SESSION_BARS = {
    "1m": 390,
    "3m": 130,
    "5m": 78,
    "10m": 39,
    "15m": 26,
    "20m": 20,
    "30m": 13,
    "45m": 9,
    "60m": 7,
}

BARS_BUILT = {
    "MNQ": True,
    "MES": True,  # built 2026-09-22: 2,177,440 1m bars, 2,018 trading days
}

# Built 2026-09-22 with build_mnq_bars_parallel.py (12 workers, 5,424s for 2,177 day files).
# Pipeline validated first: January 2020 rebuilt in parallel and diffed against the earlier
# serial output -- 29,459 bars, bit-identical.
BUILD_NOTES = {
    "MES": (
        "2,177,440 1m bars, 2020-01-01..2026-06-29, 2,018 trading days. 232 duplicate "
        "timestamps merged. Two gaps >5d (6d and 14d, Sep/Oct 2024). All nine timeframes "
        "pass the integrity battery."
    ),
    "MNQ": (
        "1,296,158 1m bars, 2021-09-01..2026-09-01, 1,203 trading days. 145 duplicate "
        "timestamps merged. Dominated by the permanent 396-day hole. All nine timeframes "
        "pass the integrity battery."
    ),
}

# Days present in MNQ but not MES, measured 2026-09-22: 80 total, fully accounted for --
# 60 fall after MES's feed ends, 17 are 274-1,018 byte Saturday/holiday files with no trades
# (correctly dropped), 3 are genuinely absent from the MES raw archives. No processing defect.
MNQ_ONLY_DAYS = 80

# Contract-roll splices: the continuous series carries 9 bar-to-bar moves above 10 ATR on MES
# 10m. Two are the March 2020 limit-down opens (real), several sit on quarterly expiry dates
# (2024-09-17, 2024-12-16) and are roll artifacts that no trader could have captured. A
# mean-reversion rule will happily "trade" a splice -- see diagnostics.roll_integrity().
ROLL_JUMPS_KNOWN = True


def make_split(symbol: str, timeframe: str, dates, *, embargo: int | None = None) -> Split:
    """The declared Split for a symbol/timeframe.

    Identical boundaries for every symbol; only the data's own start and end differ.
    `dates` is the bar datetime array.
    """
    emb = SESSION_BARS.get(timeframe, 40) if embargo is None else embargo
    return Split.by_date(
        dates,
        train_end=TRAIN_END,
        validate_end=VALIDATE_END,
        embargo=emb,
        name=f"{symbol}_{timeframe}",
        ledger=LEDGER,
    )


# ---------------------------------------------------------------------------- walk-forward
# The rolling scheme that runs INSIDE the validate segment. Declared here with the boundaries,
# because "how often do we re-optimise" is as much a pre-registration as "where does the
# holdout start", and just as easy to tune after the fact if it is not written down.
#
# Twelve months of optimisation, one month of out-of-sample per fold. Over a 14-month validate
# segment that is ~14 folds: enough that no single lucky block carries the result, few enough
# that each optimisation window still has real data behind it.
#
# The bars-per-month figure is MEASURED from the data, never hardcoded. These are FULL SESSION
# files (~23h/day), not RTH-only, so an RTH-derived constant is ~3.5x too small -- which is
# exactly the bug the first version of this file shipped: 43 folds of nine days each, wearing
# the label "monthly".
WF_OPT_MONTHS = 12
WF_STEP_MONTHS = 1


def bars_per_month(dates) -> int:
    """Median bars per calendar month, measured from the actual series."""
    import numpy as np
    import pandas as pd

    s = pd.Series(pd.to_datetime(np.asarray(dates)))
    per = s.groupby([s.dt.year, s.dt.month]).size()
    full = per[per > per.max() * 0.5]  # drop partial first/last months
    return int(full.median())


def make_wf_plan(
    dates,
    *,
    anchored: bool = False,
    opt_months: int = WF_OPT_MONTHS,
    step_months: int = WF_STEP_MONTHS,
) -> WalkForwardPlan:
    """The declared walk-forward scheme, sized from the data it will run on."""
    per_month = bars_per_month(dates)
    return WalkForwardPlan(
        train_lookback=opt_months * per_month,
        step=step_months * per_month,
        anchored=anchored,
    )
