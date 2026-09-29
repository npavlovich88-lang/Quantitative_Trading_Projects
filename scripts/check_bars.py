#!/usr/bin/env python3
"""Run the integrity battery over every timeframe of a symbol, and optionally repair.

    python scripts/check_bars.py MES
    python scripts/check_bars.py MES --repair      # merge duplicate bars, re-derive coarser TFs
    python scripts/check_bars.py MNQ --json out/mnq_integrity.json

Exit code 1 if anything FAILS, so this can gate a build.

"coverage: fail" on MNQ is expected and is not a defect in the bars -- it is the permanent
396-day hole in the underlying archives. Read the coverage block, do not just read the verdict.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant.bar_integrity import (
    check_file,
    derive_timeframe,
    merge_duplicate_bars,
)
from quant.data_splits import DATA, make_split, make_wf_plan

TIMEFRAMES = ["1m", "3m", "5m", "10m", "15m", "20m", "30m", "45m", "60m"]


def v(result: dict, key: str) -> str:
    """Four-character verdict for the table."""
    return result[key]["verdict"][:4] if key in result else "-"


def load(symbol: str, tf: str) -> tuple[pathlib.Path, pd.DataFrame | None]:
    p = pathlib.Path(DATA[symbol].format(tf=tf))
    return p, (pd.read_csv(p) if p.exists() else None)


def repair(symbol: str) -> int:
    """Merge duplicate timestamps in 1m and re-derive every coarser timeframe from it."""
    p1, one = load(symbol, "1m")
    if one is None:
        print(f"  no 1m file at {p1}; nothing to repair")
        return 0
    fixed, n = merge_duplicate_bars(one)
    if n == 0:
        print("  no duplicate timestamps; nothing to repair")
        return 0
    fixed.to_csv(p1, index=False)
    print(f"  1m: merged {n} duplicate timestamp(s), {len(one)} -> {len(fixed)} rows")
    for tf in TIMEFRAMES:
        if tf == "1m":
            continue
        path, _ = load(symbol, tf)
        if path.exists():
            agg = derive_timeframe(fixed, tf)
            agg.to_csv(path, index=False)
            print(f"  {tf:>3}: re-derived, {len(agg)} bars")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", choices=list(DATA))
    ap.add_argument("--repair", action="store_true", help="merge duplicates, re-derive coarser TFs")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.repair:
        print(f"{a.symbol}: repairing")
        repair(a.symbol)
        print()

    one = None
    results: dict[str, dict] = {}
    print(f"{a.symbol}")
    hdr = f"{'tf':>4} {'bars':>9} {'from':>10} {'to':>10} {'sch':>4} {'mon':>4} {'ohlc':>4} "
    hdr += f"{'spc':>4} {'vol':>4} {'jump':>4} {'xTF':>4}  verdict"
    print(hdr)
    print("-" * len(hdr))

    worst = "pass"
    for tf in TIMEFRAMES:
        path, df = load(a.symbol, tf)
        if df is None:
            print(f"{tf:>4}  MISSING  {path}")
            continue
        if tf == "1m":
            one = df
        dates = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_localize(None).to_numpy()
        r = check_file(df, tf, df_1m=one, dates=dates)
        results[tf] = r
        s = r["_summary"]
        # Separate BAR INTEGRITY from COVERAGE. A 396-day hole in the source archives is a
        # property of the data, not a defect in the bars, and letting it drive the headline
        # verdict trains you to ignore a FAIL -- which is how a real defect gets through.
        integrity_failed = [f for f in s["failed"] if f != "coverage"]
        if integrity_failed:
            worst = "fail"
        elif s["warned"] and worst == "pass":
            worst = "warn"
        print(
            f"{tf:>4} {len(df):>9} {str(dates[0])[:10]:>10} {str(dates[-1])[:10]:>10} "
            f"{v(r, 'schema'):>4} {v(r, 'monotonic'):>4} {v(r, 'ohlc_sane'):>4} "
            f"{v(r, 'spacing'):>4} {v(r, 'volume'):>4} {v(r, 'jumps'):>4} "
            f"{v(r, 'cross_tf'):>4}  {'FAIL' if integrity_failed else 'OK'}"
            + (f"  <- {','.join(integrity_failed)}" if integrity_failed else "")
        )

    # coverage and the declared split, reported once on a mid timeframe
    ref = "10m" if "10m" in results else next(iter(results), None)
    if ref:
        cov = results[ref].get("coverage", {})
        print(
            f"\ncoverage ({ref}): {cov.get('trading_days')} trading days, "
            f"{cov.get('business_day_coverage', 0):.2f} of business days, "
            f"{cov.get('n_gaps')} gaps >5d, largest {cov.get('largest_gap_days')}d"
        )
        for g in cov.get("gaps", [])[:5]:
            print(f"    gap after {g['after']}: {g['days']} days")

        _, df = load(a.symbol, ref)
        d = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_localize(None).to_numpy()
        sp = make_split(a.symbol, ref, d)
        print()
        print(sp.describe(d))
        print("  " + make_wf_plan(d).describe(sp, d).replace("\n", "\n  "))

    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.json).write_text(
            json.dumps(results, indent=1, default=str), encoding="utf-8"
        )
        print(f"\nwrote {a.json}")

    print(f"\nOVERALL: {worst.upper()}")
    return 1 if worst == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
