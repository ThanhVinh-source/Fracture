"""
scripts/07_normalize_demo_inputs.py

Normalize Fracture demo inputs to one file per pipeline per day.

Why this exists:
- Earlier demo generation wrote 30 days of runs into one producer_YYYYMMDD file.
- The dashboard can read that, but the behavior is confusing because one input
  date may still contain many pipeline_run_id values.
- This script rewrites demo inputs into the clearer convention:

    inputs/{pipeline_id}/producer_20260519.parquet
    inputs/{pipeline_id}/consumer_20260519.parquet
    inputs/{pipeline_id}/producer_20260520.parquet
    inputs/{pipeline_id}/consumer_20260520.parquet

Each output file contains only the runs for that input date.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["pipeline_run_id", "activity", "timestamp", "team"]
RUN_DATE_RE = re.compile(r"_(\d{8})$")


def read_event_file(path: Path) -> pd.DataFrame:
    """
    Read one CSV or parquet event file.

    Demo inputs are normally parquet, but supporting CSV makes the normalizer
    safe for hand-authored examples too.
    """
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input extension: {path}")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return df[REQUIRED_COLUMNS].copy()


def infer_run_date(row: pd.Series) -> str:
    """
    Infer the intended input date for one event row.

    Prefer the YYYYMMDD suffix in pipeline_run_id because it represents the
    process instance date. Timestamp date is a fallback for logs that do not
    encode dates in run IDs.
    """
    run_id = str(row["pipeline_run_id"])
    match = RUN_DATE_RE.search(run_id)
    if match:
        return match.group(1)

    return pd.to_datetime(row["timestamp"]).strftime("%Y%m%d")


def normalize_side(pipeline_dir: Path, side: str, dry_run: bool = False) -> tuple[int, int]:
    """
    Rewrite producer or consumer files for one pipeline into daily files.

    Returns:
        input_file_count: number of files read.
        output_file_count: number of daily files produced.
    """
    files = sorted(
        list(pipeline_dir.glob(f"{side}_*.parquet"))
        + list(pipeline_dir.glob(f"{side}_*.csv"))
    )

    if not files:
        return 0, 0

    frames = [read_event_file(path) for path in files]
    events = pd.concat(frames, ignore_index=True)

    # If a demo folder contains duplicate copies such as "producer_20260519 2.parquet",
    # keep one copy of each exact event. Otherwise charts may show the same run
    # multiple times even though the pipeline only executed once.
    events = events.drop_duplicates(REQUIRED_COLUMNS).reset_index(drop=True)

    # The temporary __input_date column controls the new file partitioning.
    events["__input_date"] = events.apply(infer_run_date, axis=1)

    if dry_run:
        return len(files), events["__input_date"].nunique()

    # Remove existing files for this side before writing normalized daily files.
    # This prevents a stale aggregate file from remaining beside the split files.
    for path in files:
        path.unlink()

    for input_date, day_events in events.groupby("__input_date", sort=True):
        output = day_events.drop(columns="__input_date")
        output.to_parquet(pipeline_dir / f"{side}_{input_date}.parquet", index=False)

    return len(files), events["__input_date"].nunique()


def normalize_inputs(inputs_dir: Path, pipeline_id: str | None = None, dry_run: bool = False):
    """
    Normalize all pipeline folders, or one selected pipeline folder.

    This function intentionally works at the file layer only. It does not change
    contracts, conformance results, or visualization outputs.
    """
    if pipeline_id:
        pipeline_dirs = [inputs_dir / pipeline_id]
    else:
        pipeline_dirs = sorted(path for path in inputs_dir.iterdir() if path.is_dir())

    total_outputs = 0

    for pipeline_dir in pipeline_dirs:
        if not pipeline_dir.exists():
            print(f"skip {pipeline_dir.name}: input directory missing")
            continue

        producer_in, producer_out = normalize_side(pipeline_dir, "producer", dry_run=dry_run)
        consumer_in, consumer_out = normalize_side(pipeline_dir, "consumer", dry_run=dry_run)
        total_outputs += producer_out + consumer_out

        action = "would normalize" if dry_run else "normalized"
        print(
            f"{action} {pipeline_dir.name}: "
            f"producer {producer_in}->{producer_out} files, "
            f"consumer {consumer_in}->{consumer_out} files"
        )

    print()
    print(f"total output files: {total_outputs}")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize Fracture demo inputs to one file per pipeline per day.",
    )
    parser.add_argument(
        "--inputs-dir",
        default="inputs",
        help="Root input directory. Default: inputs",
    )
    parser.add_argument(
        "--pipeline-id",
        default=None,
        help="Normalize only one pipeline folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned file counts without rewriting files.",
    )
    args = parser.parse_args()

    normalize_inputs(
        inputs_dir=Path(args.inputs_dir),
        pipeline_id=args.pipeline_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
