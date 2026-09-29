"""
Compute daily NY RTH-session volume profiles (POC/VAH/VAL + full price/volume
histogram) for every available trading day, from raw MarketTick trade ticks.
Parallel across a worker pool; resumable (skips days already written).

Usage: python build_volume_profiles.py <SYMBOL> [N_WORKERS]
Output: $QUANT_DATA_ROOT/<SYMBOL>_OHLCV/<SYMBOL>_volume_profiles.jsonl (one JSON object per day)
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

from quant.data_splits import DATA_ROOT

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "MES"
N_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SRC_DIR = str(DATA_ROOT / f"Future_{SYMBOL}_T2")
OUT_DIR = str(DATA_ROOT / f"{SYMBOL}_OHLCV")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, f"{SYMBOL}_volume_profiles.jsonl")

TICK = 0.25
RTH_START = pd.Timestamp("08:30").time()
RTH_END = pd.Timestamp("15:00").time()
COLS = ["raw_timestamp", "type", "price", "volume"]


def compute_profile(path: str):
    day_str = os.path.splitext(os.path.basename(path))[0]
    try:
        chunks = []
        for chunk in pd.read_csv(path, sep=";", header=None, usecols=[0, 2, 3, 4], names=COLS,
                                  dtype={"raw_timestamp": "string", "type": "int8", "price": "float64", "volume": "float64"},
                                  chunksize=3_000_000, engine="c"):
            trade = chunk.loc[chunk["type"] == 2, ["raw_timestamp", "price", "volume"]]
            if not trade.empty:
                chunks.append(trade)
    except Exception:
        return day_str, None
    if not chunks:
        return day_str, None
    df = pd.concat(chunks, ignore_index=True)
    df["ts"] = pd.to_datetime(df["raw_timestamp"], format="%Y%m%d%H%M%S%f", utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "price"])
    if df.empty:
        return day_str, None
    ts_ct = df["ts"].dt.tz_convert("America/Chicago")
    rth = df[(ts_ct.dt.time >= RTH_START) & (ts_ct.dt.time < RTH_END)]
    if rth.empty:
        return day_str, None

    bucket = (rth["price"] / TICK).round() * TICK
    vp = rth.groupby(bucket)["volume"].sum().sort_index()
    if vp.empty:
        return day_str, None

    poc = float(vp.idxmax())
    total = float(vp.sum())
    target = total * 0.70
    prices = vp.index.tolist()
    poc_i = prices.index(poc)
    lo, hi = poc_i, poc_i
    acc = vp.iloc[poc_i]
    while acc < target and (lo > 0 or hi < len(prices) - 1):
        vol_below = vp.iloc[lo - 1] if lo > 0 else -1
        vol_above = vp.iloc[hi + 1] if hi < len(prices) - 1 else -1
        if vol_above >= vol_below:
            hi += 1
            acc += vp.iloc[hi]
        else:
            lo -= 1
            acc += vp.iloc[lo]
    val, vah = float(prices[lo]), float(prices[hi])

    result = {
        "day": day_str, "poc": poc, "vah": vah, "val": val, "total_volume": total,
        "profile": [[float(p), float(v)] for p, v in vp.items()],
    }
    return day_str, result


def main():
    day_files = sorted(glob.glob(os.path.join(SRC_DIR, f"Future_{SYMBOL}_T2_*", "*.csv")))
    print(f"{SYMBOL}: {len(day_files)} day files found, {N_WORKERS} workers", flush=True)

    done_days = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_days.add(json.loads(line)["day"])
                except Exception:
                    pass
    print(f"resuming: {len(done_days)} days already computed", flush=True)

    todo = [p for p in day_files if os.path.splitext(os.path.basename(p))[0] not in done_days]
    print(f"remaining: {len(todo)} days to process", flush=True)
    if not todo:
        print("nothing to do.", flush=True)
        return

    t0 = time.time()
    n_done = 0
    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(compute_profile, p): p for p in todo}
            for fut in as_completed(futures):
                day_str, result = fut.result()
                n_done += 1
                if result is not None:
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                if n_done % 25 == 0 or n_done == len(todo):
                    elapsed = time.time() - t0
                    rate = elapsed / n_done
                    remaining = (len(todo) - n_done) * rate
                    print(f"  [{n_done}/{len(todo)}] {day_str} done  "
                          f"({elapsed:.0f}s elapsed, {rate:.1f}s/day avg, ~{remaining/60:.0f}min remaining)", flush=True)

    print(f"\ncomplete: {n_done} days processed in {time.time()-t0:.0f}s -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
