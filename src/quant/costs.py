"""Transaction costs, measured where possible and sourced where not.

The old model was a single number -- 2.5 points round turn on MNQ -- with no derivation behind
it. That is the shape of assumption that survives a whole project unexamined, and it sits
underneath every result: an edge that dies at 3 points and one that dies at 30 look identical
until you vary it.

This replaces it with four components, each of which can be defended separately:

    spread        MEASURED from the tick data. Not assumed, not a round number.
    exchange fee  CME published rate.
    regulatory    NFA published rate.
    commission    YOUR broker or prop firm. Must be set; there is no sensible default.

WHAT THE MEASUREMENT FOUND

Top-of-book quotes are directly present in the MarketTick feed (level=1, type=0 bid / type=1
ask), so the NBBO can be reconstructed without replaying the full book. Measured over weekday
sessions, RTH and all-day, on multiple dates:

    MES   median 0.25 pts = 1 tick.   95-96% of the time at exactly one tick. Stable.
    MNQ   median 0.50 pts = 2 ticks.  Only 8-49% at one tick, varying by day.

That MNQ trades at two ticks rather than one is the substantive finding. It is also why the
per-point cost of MNQ and MES end up closer than the contract sizes suggest: MNQ's spread is
twice as wide in ticks but its point value is 2.5x smaller.

A caution on the first measurement attempt, recorded because it would have been an easy result
to publish: the first sample landed on 2025-02-09, a SUNDAY. Sunday-evening spreads are far
wider, and applying them to an RTH-only strategy would have inflated costs substantially. Always
check the weekday and the session before quoting a microstructure statistic.

HOW MUCH SPREAD DOES A ROUND TURN ACTUALLY PAY

A market order buys at the ask and sells at the bid, so one round turn crosses the spread ONCE,
not twice. `spread_cost_points` is therefore the full spread, not half of it. Resting a limit
order can earn the spread instead of paying it, which this model does not attempt -- assuming
passive fills without modelling queue position is how a backtest invents money.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TICK = 0.25

# Contract specifications -- CME published.
POINT_VALUE = {"MES": 5.0, "MNQ": 2.0}

# --------------------------------------------------------------------------- measured
# Median NBBO spread in POINTS, measured from MarketTick level-1 quotes on weekday sessions
# (2025-02-11 and 2025-06-11, RTH and all-day). See measure_spread() to re-derive.
MEASURED_SPREAD_POINTS = {"MES": 0.25, "MNQ": 0.50}
SPREAD_PROVENANCE = (
    "measured 2026-09-22 from MarketTick level-1 quotes, weekday sessions; "
    "MES 0.25 pts (95-96% at one tick), MNQ 0.50 pts (median two ticks)"
)

# --------------------------------------------------------------------------- published
# CME exchange fee and NFA regulatory fee, per contract per SIDE, micro equity index futures.
# Sourced 2026-09-22 via broker fee schedules quoting the CME schedule. Verify against
# cmegroup.com/company/clearing-fees.html before trusting these to the cent -- fee schedules
# change, and CME announced changes effective 2026-10-01.
EXCHANGE_FEE_PER_SIDE = 0.35
NFA_FEE_PER_SIDE = 0.01
FEE_PROVENANCE = (
    "CME exchange $0.35/side and NFA $0.01/side for micro equity index futures. The exchange "
    "figure was sourced independently from broker schedules AND matches Topstep's published "
    "breakdown ($0.70 round turn) exactly, which is good corroboration. NFA is taken from "
    "Topstep's $0.02 ROUND TURN figure -- note some schedules quote $0.02 per SIDE; Topstep is "
    "authoritative for what is actually charged on a Topstep account. Re-verify at "
    "cmegroup.com/company/clearing-fees.html (CME announced changes effective 2026-10-01)."
)

# --------------------------------------------------------------------------- slippage
# MEASURED by walking the level-2 ask ladder: for an order of N contracts, the size-weighted
# fill price minus the price at the touch. RTH only, 2025-02-11, ~4,000 book snapshots.
#
# This is slippage BEYOND the spread. The spread is already charged separately -- adding a
# flat "slippage" on top of it, as most backtests do, double-counts the first tick.
#
# THE FINDING THAT MATTERS: MES and MNQ are completely different animals here.
#
#   MES  median 82 contracts resting at the touch. Even 50 lots exceed it only 9% of the time,
#        and cost 0.05 ticks. Slippage is genuinely negligible at any size we would trade.
#   MNQ  median SIX contracts at the touch. 10 lots exceed it 92% of the time. Slippage is
#        ~0 to 5 lots, then climbs fast: 0.43 ticks at 10, 1.05 at 20, 2.13 at 50.
#
# So "slippage is not an issue" is true for MES at any size, and true for MNQ only up to about
# five contracts. Above that on MNQ it becomes comparable to the entire fee bill. Our earlier
# prop-firm Monte Carlo swept contract counts up to 32 -- at that size on MNQ the book is being
# walked on essentially every fill, and that cost was nowhere in the model.
SLIPPAGE_TICKS_BY_SIZE = {
    "MNQ": {
        1: 0.00,
        2: 0.00,
        3: 0.01,
        5: 0.07,
        8: 0.28,
        10: 0.43,
        15: 0.74,
        20: 1.05,
        30: 1.49,
        50: 2.13,
    },
    "MES": {
        1: 0.00,
        2: 0.00,
        3: 0.00,
        5: 0.00,
        8: 0.00,
        10: 0.00,
        15: 0.01,
        20: 0.02,
        30: 0.03,
        50: 0.05,
    },
}
SLIPPAGE_PROVENANCE = (
    "measured 2026-09-22 by walking the level-2 ask ladder on RTH book snapshots (2025-02-11); "
    "mean fill price minus touch price, per side, excludes the spread which is charged "
    "separately. Median size at the touch: MES 82 contracts, MNQ 6."
)


def slippage_ticks_for_size(symbol: str, contracts: int) -> float:
    """Interpolate the measured book-walk cost, per side, in ticks.

    Linear between measured points; beyond the largest measured size, extrapolate on the last
    segment's slope -- and treat anything out there as a guess, not a measurement.
    """
    table = SLIPPAGE_TICKS_BY_SIZE[symbol]
    xs = sorted(table)
    ys = [table[x] for x in xs]
    if contracts <= xs[0]:
        return ys[0]
    if contracts >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return ys[-1] + slope * (contracts - xs[-1])
    import numpy as _np

    return float(_np.interp(contracts, xs, ys))


# --------------------------------------------------------------------------- prop firms
# Published rates, fetched 2026-09-22. Values are COMMISSION ONLY, per side -- exchange and NFA
# are added separately by CostModel so the components stay separable.
#
# TOPSTEP publishes a full breakdown and it reconciles exactly:
#     NFA $0.02 + exchange $0.70 + commission $0.50 = $1.22 round turn   (MES and MNQ alike)
#   -> commission $0.25/side.
#
# LUCID publishes "$0.50 per side" for micros on a page headed "Approved Products and
# Commissions", and does NOT state whether that is commission-only or all-in. Two readings:
#     commission-only -> all-in $1.00 + $0.70 + $0.02 = $1.72 round turn
#     all-in          -> $1.00 round turn
# We take the COMMISSION-ONLY reading. It is the natural one given the column is labelled
# "Commission (Per Side)" and it matches how Topstep itemises the same line, and it is the
# conservative one -- overstating cost is the safe direction to be wrong in. Flagged here so a
# future session can correct it from a Lucid statement rather than re-deriving the ambiguity.
PROP_COMMISSION_PER_SIDE = {
    "topstep": 0.25,  # $0.50 round turn, itemised in their published breakdown
    "lucid": 0.50,  # $1.00 round turn, all-in status NOT disclosed -- see note above
}
PROP_PROVENANCE = {
    "topstep": "help.topstep.com TopstepX Commissions and Fees, fetched 2026-09-22: "
    "$1.22 round turn all-in = NFA $0.02 + exchange $0.70 + commission $0.50",
    "lucid": "support.lucidtrading.com Approved Products and Commissions, fetched 2026-09-22: "
    "$0.50 per side for MES/MNQ; all-in status not disclosed, read as commission-only",
}


@dataclass(frozen=True)
class CostModel:
    """Round-turn cost for one contract, decomposed.

    `commission_per_side` has NO default on purpose. Broker rates run roughly $0.25-$1.50 per
    side on micros and prop firms publish their own; picking a number for you would recreate
    exactly the unexamined assumption this module exists to remove.
    """

    symbol: str
    commission_per_side: float
    contracts: int = 1  # drives the measured, size-dependent book-walk slippage
    spread_points: float | None = None  # None -> use the measured value
    slippage_ticks: float | None = None  # None -> measured for `contracts`; a float overrides
    exchange_fee_per_side: float = EXCHANGE_FEE_PER_SIDE
    nfa_fee_per_side: float = NFA_FEE_PER_SIDE

    @classmethod
    def for_prop(cls, symbol: str, firm: str = "average", **kw) -> CostModel:
        """Cost model using a prop firm's published commission.

        `firm="average"` is the mean of Topstep and Lucid, which is what to use when the
        account is undecided. It is NOT a real rate either firm charges -- when the account
        is chosen, name the firm.
        """
        if firm == "average":
            comm = sum(PROP_COMMISSION_PER_SIDE.values()) / len(PROP_COMMISSION_PER_SIDE)
        else:
            comm = PROP_COMMISSION_PER_SIDE[firm]
        return cls(symbol=symbol, commission_per_side=comm, **kw)

    @property
    def point_value(self) -> float:
        return POINT_VALUE[self.symbol]

    @property
    def spread(self) -> float:
        return (
            MEASURED_SPREAD_POINTS[self.symbol]
            if self.spread_points is None
            else self.spread_points
        )

    @property
    def fees_usd(self) -> float:
        """Commission + exchange + regulatory, both sides."""
        return 2 * (self.commission_per_side + self.exchange_fee_per_side + self.nfa_fee_per_side)

    @property
    def spread_usd(self) -> float:
        """One crossing of the spread per round turn -- buy the ask, sell the bid."""
        return self.spread * self.point_value

    @property
    def slippage_per_side_ticks(self) -> float:
        if self.slippage_ticks is not None:
            return self.slippage_ticks
        return slippage_ticks_for_size(self.symbol, self.contracts)

    @property
    def slippage_usd(self) -> float:
        """Per contract, both sides. Charged on top of the spread, never instead of it."""
        return 2 * self.slippage_per_side_ticks * TICK * self.point_value

    @property
    def round_turn_usd(self) -> float:
        return self.fees_usd + self.spread_usd + self.slippage_usd

    @property
    def round_turn_points(self) -> float:
        """What the simulator wants: cost expressed in price points."""
        return self.round_turn_usd / self.point_value

    def breakdown(self) -> str:
        p = self.point_value
        rows = [
            ("commission", 2 * self.commission_per_side),
            ("exchange fee", 2 * self.exchange_fee_per_side),
            ("NFA fee", 2 * self.nfa_fee_per_side),
            (f"spread ({self.spread:.2f} pts)", self.spread_usd),
        ]
        if self.slippage_usd:
            rows.append(
                (
                    f"slippage @{self.contracts} lot ({self.slippage_per_side_ticks:.2f} tk/side)",
                    self.slippage_usd,
                )
            )
        w = max(len(r[0]) for r in rows)
        out = [f"{self.symbol} round-turn cost, 1 contract (point value ${p:.2f})"]
        out += [f"  {n:<{w}}  ${v:>6.2f}   {v / p:>6.3f} pts" for n, v in rows]
        out.append(
            f"  {'TOTAL':<{w}}  ${self.round_turn_usd:>6.2f}   {self.round_turn_points:>6.3f} pts"
        )
        return "\n".join(out)


# --------------------------------------------------------------------------- measurement
QUOTE_COLS = ["raw_timestamp", "level", "type", "price", "volume", "depth", "action"]


def measure_spread(tick_csv: str, *, rth_only: bool = True, nrows: int | None = 4_000_000) -> dict:
    """Reconstruct the NBBO from one raw MarketTick day file and summarise the spread.

    Top-of-book quotes arrive as level=1 records: type 0 = bid, type 1 = ask. Forward-fill each
    side through the event stream and the difference is the quoted spread at every instant.
    """
    df = pd.read_csv(
        tick_csv, sep=";", header=None, names=QUOTE_COLS, nrows=nrows, dtype={0: "string"}
    )
    q = df[(df["level"] == 1) & df["type"].isin([0, 1])].copy()
    if q.empty:
        return dict(n=0, reason="no level-1 quotes in file")
    q["ts"] = pd.to_datetime(q["raw_timestamp"], format="%Y%m%d%H%M%S%f", utc=True)
    q = q.sort_values("ts", kind="stable")
    bid = pd.Series(np.where(q["type"] == 0, q["price"], np.nan)).ffill().to_numpy()
    ask = pd.Series(np.where(q["type"] == 1, q["price"], np.nan)).ffill().to_numpy()
    spread = ask - bid
    ok = np.isfinite(spread) & (spread > 0) & (spread < 50)

    ts = q["ts"].to_numpy()[ok]
    s = spread[ok]
    if rth_only and len(ts):
        ct = pd.DatetimeIndex(ts).tz_convert("America/Chicago")
        mins = ct.hour * 60 + ct.minute
        m = (mins >= 510) & (mins < 900)  # 08:30-15:00 CT
        ts, s = ts[m], s[m]
    if not len(s):
        return dict(n=0, reason="no quotes in the requested session")
    return dict(
        n=len(s),
        day=str(pd.Timestamp(ts[0]).date()),
        weekday=pd.Timestamp(ts[0]).day_name(),
        session="RTH" if rth_only else "all-day",
        median_points=float(np.median(s)),
        mean_points=float(s.mean()),
        median_ticks=float(np.median(s) / TICK),
        pct_one_tick=float(np.mean(s <= TICK * 1.001)),
        p90=float(np.quantile(s, 0.90)),
        p99=float(np.quantile(s, 0.99)),
    )
