"""Prepare actual NYC TLC samples, with provenance; run once before class.

Default reads existing raw files only. --download explicitly enables downloads.
Sampling happens before duration filtering so the notebook can inspect outliers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import pandas as pd

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
COLUMNS = ["lpep_pickup_datetime", "lpep_dropoff_datetime", "PULocationID", "DOLocationID"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.rows <= 0:
        parser.error("--rows must be positive")
    targets = [args.output / name for name in
               ("train.parquet", "validation.parquet", "inference.csv", "manifest.json")]
    if any(p.exists() for p in targets):
        parser.error("Sample files already exist; choose a new --output directory")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "NYC TLC Green Taxi", "seed": 2026,
                "sampling": "Exact pickup month, then pandas sample(random_state=2026), sorted by pickup time",
                "pandas_version": pd.__version__, "requested_rows_per_month": args.rows,
                "source_page": "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
                "files": {}}
    for month, name in ((1, "train.parquet"), (2, "validation.parquet")):
        filename = f"green_tripdata_2021-{month:02d}.parquet"
        raw = args.raw_dir / filename
        url = f"{BASE}/{filename}"
        if not raw.exists():
            if not args.download:
                parser.error(f"Missing {raw}; supply --download or an existing --raw-dir")
            with urllib.request.urlopen(url, timeout=60) as response, raw.open("xb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
        frame = pd.read_parquet(raw, columns=COLUMNS)
        source_rows = len(frame)
        pickup = pd.to_datetime(frame.lpep_pickup_datetime)
        frame = frame.loc[(pickup.dt.year == 2021) & (pickup.dt.month == month)].copy()
        month_rows = len(frame)
        frame["ride_id"] = [f"green-2021-{month:02d}-{i}" for i in frame.index]
        frame = frame.sample(n=min(args.rows, len(frame)), random_state=2026)
        frame = frame.sort_values("lpep_pickup_datetime").reset_index(drop=True)
        target = args.output / name
        frame.to_parquet(target, index=False)
        manifest["files"][name] = {"source_url": url, "source_sha256": digest(raw),
                                  "source_rows": source_rows, "month_rows": month_rows,
                                  "rows": len(frame), "sha256": digest(target)}
        if month == 2:
            inference = frame[["ride_id", "PULocationID", "DOLocationID"]].head(20)
            target = args.output / "inference.csv"
            inference.to_csv(target, index=False)
            manifest["files"]["inference.csv"] = {"rows": len(inference), "sha256": digest(target),
                                                   "derived_from": name, "selection": "first 20 rows; no target or timestamps"}
    with (args.output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Prepared actual TLC samples in {args.output}")


if __name__ == "__main__":
    main()
