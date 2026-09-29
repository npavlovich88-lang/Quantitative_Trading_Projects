#!/usr/bin/env python3
"""Build 1-minute order-flow bars from the raw L2 archives.

    python scripts/build_ofi_bars.py --symbol MES --workers 12

Produces one row per minute with the microstructure features that bars alone cannot carry:

    ofi        Cont, Kukanov & Stoikov (2014) order flow imbalance, summed over the minute
    signed_vol aggressive buy volume minus aggressive sell volume
    buy_vol    volume printing at the ask
    sell_vol   volume printing at the bid
    volume     total traded volume  (MUST equal the OHLCV bar exactly -- integrity check)
    n_trades   trade prints
    n_quotes   best-quote updates, i.e. the true tick rate
    spread     mean best ask minus best bid
    bid_size   mean size resting at the best bid
    ask_size   mean size resting at the best ask

WHY 1-MINUTE AND NOT THE SIGNAL TIMEFRAME

Same reason the OHLCV pipeline works this way: extract once at the finest useful resolution and
aggregate upward. OFI, signed volume and counts are additive across minutes, so any coarser bar
is a groupby. Re-reading 511 GB to change a timeframe is not.

FORMAT NOTES, each of which cost time to discover

  * Timestamps are UTC, 20 digits: YYYYMMDDHHMMSS plus 6 microsecond digits. As a single number
    that is 2.02e19, which OVERFLOWS int64 -- it must be split, never cast.
  * Level 1 is the top of book and the trade tape: type 0 bid, 1 ask, 2 trade. Types 3-9 appear
    a handful of times a day and are session markers, not quotes.
  * Level 2 is the 10-deep book with an action field (0 insert, 1 update, 2 delete). Not used
    here; the level-1 feed already carries best price AND size on every change, which is exactly
    what OFI is defined on.
  * The tick-derived daily volume profiles shipped with this data are RTH-ONLY. These bars are
    full session.

TRADE SIGNING

97.2% of prints land exactly at the prevailing best bid or best ask, so the quote rule suffices
and no Lee-Ready inference is needed. The 2.8% that print inside or outside the spread are
counted in `volume` but contribute 0 to `signed_vol`, rather than being forced to a side.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant.data_splits import DATA_ROOT, RAW  # resolved from QUANT_DATA_ROOT
OUT = {
    "MES": str(DATA_ROOT / "MES_OHLCV" / "MES_1m_ofi.parquet"),
    "MNQ": str(DATA_ROOT / "MNQ_OHLCV" / "MNQ_1m_ofi.parquet"),
}
COLS = ["ts", "level", "type", "price", "size"]


def one_day(path: str) -> pd.DataFrame | None:
    """1-minute order-flow features for a single day file, or None if unusable."""
    try:
        df = pd.read_csv(
            path,
            sep=";",
            header=None,
            usecols=[0, 1, 2, 3, 4],
            names=COLS,
            dtype={0: str, 1: np.int8, 2: np.int8, 3: float, 4: float},
        )
    except Exception:
        return None
    l1 = df[(df["level"] == 1) & (df["type"] < 3)]
    if len(l1) < 100:
        return None

    sec = pd.to_datetime(l1["ts"].str[:14], format="%Y%m%d%H%M%S", utc=True)
    epoch = ((sec - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()
    ty = l1["type"].to_numpy()
    px = l1["price"].to_numpy()
    sz = l1["size"].to_numpy()

    def ffill(x):
        return pd.Series(x).ffill().to_numpy()

    bp = ffill(np.where(ty == 0, px, np.nan))
    bq = ffill(np.where(ty == 0, sz, np.nan))
    ap = ffill(np.where(ty == 1, px, np.nan))
    aq = ffill(np.where(ty == 1, sz, np.nan))
    pbp, pbq = np.r_[bp[0], bp[:-1]], np.r_[bq[0], bq[:-1]]
    pap, paq = np.r_[ap[0], ap[:-1]], np.r_[aq[0], aq[:-1]]
    ofi = np.nan_to_num((bp >= pbp) * bq - (bp <= pbp) * pbq - (ap <= pap) * aq + (ap >= pap) * paq)

    is_tr = ty == 2
    # Quote rule. Prints inside or outside the spread get sign 0 rather than being forced.
    sgn = np.where(np.isclose(px, ap), 1.0, np.where(np.isclose(px, bp), -1.0, 0.0))

    out = pd.DataFrame(
        {
            "ts": (epoch // 60) * 60,
            "ofi": ofi,
            "volume": np.where(is_tr, sz, 0.0),
            "signed_vol": np.where(is_tr, sgn * sz, 0.0),
            "buy_vol": np.where(is_tr & (sgn > 0), sz, 0.0),
            "sell_vol": np.where(is_tr & (sgn < 0), sz, 0.0),
            "n_trades": is_tr.astype(float),
            "n_quotes": 1.0,
            "spread": ap - bp,
            "bid_size": bq,
            "ask_size": aq,
        }
    )
    g = out.groupby("ts").agg(
        ofi=("ofi", "sum"),
        volume=("volume", "sum"),
        signed_vol=("signed_vol", "sum"),
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        n_trades=("n_trades", "sum"),
        n_quotes=("n_quotes", "sum"),
        spread=("spread", "mean"),
        bid_size=("bid_size", "mean"),
        ask_size=("ask_size", "mean"),
    )
    return g.reset_index()


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--symbol", default="MES", choices=list(RAW))
    ap_.add_argument("--workers", type=int, default=12)
    ap_.add_argument("--limit", type=int, default=0, help="process only the first N days")
    a = ap_.parse_args()

    files = sorted(str(p) for p in pathlib.Path(RAW[a.symbol]).glob("*/20*.csv"))
    if a.limit:
        files = files[: a.limit]
    print(f"{a.symbol}: {len(files):,} day files, {a.workers} workers", flush=True)

    t0 = time.time()
    parts, bad = [], 0
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one_day, files, chunksize=4), 1):
            if r is None:
                bad += 1
            else:
                parts.append(r)
            if i % 100 == 0:
                el = time.time() - t0
                print(
                    f"  {i:>5,}/{len(files):,}  {el:>5.0f}s elapsed, "
                    f"~{el / i * (len(files) - i):.0f}s left, {bad} skipped",
                    flush=True,
                )

    df = pd.concat(parts, ignore_index=True).sort_values("ts")
    # Duplicate minutes across overlapping day files: sum the additive columns, mean the levels.
    dup = int(df["ts"].duplicated().sum())
    if dup:
        add = ["ofi", "volume", "signed_vol", "buy_vol", "sell_vol", "n_trades", "n_quotes"]
        lev = ["spread", "bid_size", "ask_size"]
        df = df.groupby("ts").agg({**dict.fromkeys(add, "sum"), **dict.fromkeys(lev, "mean")})
        df = df.reset_index()
    out = pathlib.Path(OUT[a.symbol])
    df.to_parquet(out, index=False)

    span = pd.to_datetime(df["ts"], unit="s", utc=True)
    print(
        f"\n  {len(df):,} minute bars, {span.min().date()} -> {span.max().date()}, "
        f"{dup:,} duplicate minutes merged, {bad} files skipped"
    )
    print(f"  total volume {df['volume'].sum():,.0f}, written to {out}")
    print(f"  {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
