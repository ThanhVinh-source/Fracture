"""
tests/test_visualization_data_layer.py

Tests for the Phase 2 visualization data layer.

These tests verify that dashboard loaders are safe around missing files,
producer-only logs, and output directory creation.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from fracture.visualization import (
    ensure_visualization_dir,
    load_conformance_log,
    load_contracts,
    load_pipeline_events,
)


passed = 0
failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"v {name}")
    except AssertionError as e:
        failed += 1
        print(f"x {name}")
        print(f"  {e}")
    except Exception as e:
        failed += 1
        print(f"x {name}: {type(e).__name__}: {e}")


def test_ensure_visualization_dir_creates_fleet_directory():
    tmp = Path(tempfile.mkdtemp())
    try:
        output_dir = tmp / "outputs" / "visualizations"

        path = ensure_visualization_dir(output_dir=str(output_dir))

        assert path.exists()
        assert path == output_dir
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ensure_visualization_dir_creates_pipeline_directory():
    tmp = Path(tempfile.mkdtemp())
    try:
        output_dir = tmp / "outputs" / "visualizations"

        path = ensure_visualization_dir(
            pipeline_id="payment_batch",
            output_dir=str(output_dir),
        )

        assert path.exists()
        assert path == output_dir / "payment_batch"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_conformance_log_missing_file_returns_empty_dataframe():
    tmp = Path(tempfile.mkdtemp())
    try:
        missing_log = tmp / "conformance_log.csv"

        df, status = load_conformance_log(str(missing_log))

        assert df.empty
        assert "Missing conformance log" in status
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_conformance_log_reads_existing_csv():
    tmp = Path(tempfile.mkdtemp())
    try:
        log_path = tmp / "conformance_log.csv"
        pd.DataFrame([
            {
                "pipeline_id": "payment_batch",
                "run_date": "20260531",
                "final_score": 1.0,
                "timing_zone": "GREEN",
            }
        ]).to_csv(log_path, index=False)

        df, status = load_conformance_log(str(log_path))

        assert status == "ok"
        assert len(df) == 1
        assert df.loc[0, "pipeline_id"] == "payment_batch"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_contracts_missing_directory_returns_empty_dataframe():
    tmp = Path(tempfile.mkdtemp())
    try:
        contracts_dir = tmp / "contracts"

        df, status = load_contracts(str(contracts_dir))

        assert df.empty
        assert "Missing contracts directory" in status
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_pipeline_events_missing_pipeline_returns_empty_state():
    tmp = Path(tempfile.mkdtemp())
    try:
        producer_df, consumer_df, status = load_pipeline_events(
            inputs_dir=str(tmp / "inputs"),
            pipeline_id="missing_pipeline",
            date_str="20260531",
        )

        assert producer_df.empty
        assert consumer_df is None
        assert status.startswith("missing_input")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_pipeline_events_producer_only_mode():
    tmp = Path(tempfile.mkdtemp())
    try:
        inputs_dir = tmp / "inputs"
        pipeline_dir = inputs_dir / "payment_batch"
        pipeline_dir.mkdir(parents=True)

        events = pd.DataFrame([
            {
                "pipeline_run_id": "run_1",
                "activity": "SCHEDULED",
                "timestamp": datetime(2026, 5, 31, 6, 0, tzinfo=timezone.utc),
                "team": "producer",
            },
            {
                "pipeline_run_id": "run_1",
                "activity": "STARTED",
                "timestamp": datetime(2026, 5, 31, 6, 1, tzinfo=timezone.utc),
                "team": "producer",
            },
            {
                "pipeline_run_id": "run_1",
                "activity": "COMPLETED",
                "timestamp": datetime(2026, 5, 31, 6, 45, tzinfo=timezone.utc),
                "team": "producer",
            },
            {
                "pipeline_run_id": "run_1",
                "activity": "DATA_AVAILABLE",
                "timestamp": datetime(2026, 5, 31, 6, 50, tzinfo=timezone.utc),
                "team": "producer",
            },
        ])

        events.to_parquet(pipeline_dir / "producer_20260531.parquet")

        producer_df, consumer_df, status = load_pipeline_events(
            inputs_dir=str(inputs_dir),
            pipeline_id="payment_batch",
            date_str="20260531",
        )

        assert status == "producer_only"
        assert len(producer_df) == 4
        assert consumer_df is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print()
    print("Phase 2 visualization data layer tests")
    print("=" * 50)

    check("ensure_visualization_dir creates fleet directory",
          test_ensure_visualization_dir_creates_fleet_directory)
    check("ensure_visualization_dir creates pipeline directory",
          test_ensure_visualization_dir_creates_pipeline_directory)
    check("load_conformance_log missing file returns empty DataFrame",
          test_load_conformance_log_missing_file_returns_empty_dataframe)
    check("load_conformance_log reads existing CSV",
          test_load_conformance_log_reads_existing_csv)
    check("load_contracts missing directory returns empty DataFrame",
          test_load_contracts_missing_directory_returns_empty_dataframe)
    check("load_pipeline_events missing pipeline returns empty state",
          test_load_pipeline_events_missing_pipeline_returns_empty_state)
    check("load_pipeline_events supports producer-only mode",
          test_load_pipeline_events_producer_only_mode)

    print()
    print(f"Results: {passed + failed} tests  v {passed}  x {failed}")

    if failed:
        raise SystemExit(1)