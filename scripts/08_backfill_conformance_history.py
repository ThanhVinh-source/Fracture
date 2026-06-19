"""
scripts/08_backfill_conformance_history.py

Backfill conformance_log.csv for every available input date.

Why this exists:
- Dashboard Drift Chart reads conformance_log.csv.
- Event-log visuals read producer/consumer files directly.
- If inputs have 30 dates but conformance_log.csv has only one row, event-log
  charts look historical while drift charts still look like a single point.

This script runs the existing Fracture CLI run-all logic once per available
producer input date so every pipeline gets comparable historical rows.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import types
from pathlib import Path

import pandas as pd


# Allow running this script directly from the repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))


def discover_input_dates(inputs_dir: Path) -> list[str]:
    """
    Discover all YYYYMMDD dates present in producer input files.

    Producer files are the anchor because Fracture can still run in producer-only
    mode when a consumer file is intentionally missing.
    """
    dates = set()

    for path in inputs_dir.glob("*/producer_*.parquet"):
        # File names are controlled by Fracture's input contract:
        # producer_YYYYMMDD.parquet.
        date_part = path.stem.replace("producer_", "")
        if len(date_part) == 8 and date_part.isdigit():
            dates.add(date_part)

    return sorted(dates)


def run_all_for_date(run_date: str, contracts_dir: str, inputs_dir: str, quiet: bool) -> int:
    """
    Execute Fracture's existing run-all command for one date.

    Reusing cmd_run_all keeps this script aligned with normal CLI behavior:
    same engine, same conformance calculation, same conformance_log.csv format.
    """
    from fracture.cli import cmd_run_all

    args = types.SimpleNamespace(
        date=run_date,
        team=None,
        contracts_dir=contracts_dir,
        inputs_dir=inputs_dir,
    )

    if quiet:
        # Keep backfill output readable; detailed per-pipeline logs are still
        # available by running `python -m fracture.cli run-all --date YYYYMMDD`.
        with contextlib.redirect_stdout(io.StringIO()):
            return cmd_run_all(args)

    return cmd_run_all(args)


def drop_no_input_rows(log_path: Path) -> int:
    """
    Remove NO_INPUT rows created for pipelines without a file on a fleet date.

    Backfill runs at fleet-date level. Some demo pipelines may intentionally have
    fewer input dates than others, so keeping NO_INPUT rows would make dashboard
    "latest conformance date" point at a day with no measurement.
    """
    if not log_path.exists():
        return 0

    df = pd.read_csv(log_path)

    if "confidence_level" not in df.columns:
        return 0

    no_input_mask = df["confidence_level"].fillna("").eq("NO_INPUT")
    removed = int(no_input_mask.sum())

    if removed:
        df.loc[~no_input_mask].to_csv(log_path, index=False)

    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Backfill conformance_log.csv for every available input date.",
    )
    parser.add_argument(
        "--inputs-dir",
        default="inputs",
        help="Root input directory. Default: inputs",
    )
    parser.add_argument(
        "--contracts-dir",
        default="contracts",
        help="Contracts directory. Default: contracts",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional inclusive YYYYMMDD lower bound.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive YYYYMMDD upper bound.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show normal run-all output for every date.",
    )
    parser.add_argument(
        "--keep-no-input",
        action="store_true",
        help="Keep NO_INPUT rows for dates where a pipeline has no producer file.",
    )
    args = parser.parse_args()

    input_dates = discover_input_dates(Path(args.inputs_dir))

    if args.start_date:
        input_dates = [d for d in input_dates if d >= args.start_date]
    if args.end_date:
        input_dates = [d for d in input_dates if d <= args.end_date]

    if not input_dates:
        print("No producer input dates found.")
        return 1

    print(
        f"Backfilling {len(input_dates)} dates "
        f"({input_dates[0]} to {input_dates[-1]})"
    )

    failures = []
    for index, input_date in enumerate(input_dates, start=1):
        rc = run_all_for_date(
            run_date=input_date,
            contracts_dir=args.contracts_dir,
            inputs_dir=args.inputs_dir,
            quiet=not args.verbose,
        )

        if rc:
            failures.append(input_date)
            print(f"  x {index:02d}/{len(input_dates)} {input_date}")
        else:
            print(f"  v {index:02d}/{len(input_dates)} {input_date}")

    if failures:
        print()
        print("Failed dates:", ", ".join(failures))
        return 1

    if not args.keep_no_input:
        removed = drop_no_input_rows(Path("conformance_log.csv"))
        if removed:
            print()
            print(f"removed {removed} NO_INPUT rows")

    print()
    print("conformance_log.csv backfill complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
