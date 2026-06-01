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
from argparse import Namespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from fracture.visualization import (
    ensure_visualization_dir,
    load_conformance_log,
    load_contracts,
    load_pipeline_events,
    load_cluster_assignments,
    compute_bilateral_gap_points,
    save_bilateral_gap_timeline,
    prepare_drift_history,
    save_drift_chart,
    extract_changepoint_dates,
)
from fracture.cli import cmd_visualize


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

def test_load_contracts_reads_valid_contract_yaml():
    tmp = Path(tempfile.mkdtemp())
    try:
        contracts_dir = tmp / "contracts"
        contracts_dir.mkdir(parents=True)

        # This YAML mirrors a real registered pipeline contract.
        # The visualization loader should turn it into one dashboard row.
        contract_yaml = """
pipeline_id: payment_batch
owner: owner@example.com
producer_team: payments-platform
consumer_team: settlement-ops
expected_start: "06:00"
expected_end: "08:00"
grace_minutes: 30
p50_minutes: 45
p95_minutes: 60
p99_minutes: 75
criticality: medium
status: active
log_contract:
  transport: parquet
  source_path: inputs/payment_batch/
  required_events:
    - SCHEDULED
    - STARTED
    - VALIDATED
    - COMPLETED
    - DATA_AVAILABLE
  optional_activities:
    - VALIDATED
  terminal_event: COMPLETED
  activity_name_map: {}
  grain: pipeline
  deduplicate_retries: false
"""
        (contracts_dir / "payment_batch.yaml").write_text(contract_yaml)

        df, status = load_contracts(str(contracts_dir))

        assert status == "ok"
        assert len(df) == 1
        assert df.loc[0, "pipeline_id"] == "payment_batch"
        assert df.loc[0, "owner"] == "owner@example.com"
        assert df.loc[0, "status"] == "active"

        # These string fields are prepared for dashboard tables.
        assert "SCHEDULED" in df.loc[0, "required_events"]
        assert "VALIDATED" in df.loc[0, "optional_activities"]

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_load_cluster_assignments_missing_file_returns_empty_dataframe():
    tmp = Path(tempfile.mkdtemp())
    try:
        cluster_path = tmp / "cluster_assignments.csv"

        df, status = load_cluster_assignments(str(cluster_path))

        assert df.empty
        assert "Missing cluster assignments" in status

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        
def test_load_cluster_assignments_reads_existing_csv():
    tmp = Path(tempfile.mkdtemp())
    try:
        cluster_path = tmp / "cluster_assignments.csv"

        # Minimal cluster output expected from clustering.py.
        # Future Fleet Overview charts can use this for colors and grouping.
        pd.DataFrame([
            {
                "pipeline_id": "payment_batch",
                "cluster": "HEALTHY",
                "final_score": 1.02,
                "bilateral_gap_minutes": 4.8,
            },
            {
                "pipeline_id": "risk_report",
                "cluster": "DRIFTING",
                "final_score": 0.81,
                "bilateral_gap_minutes": 28.5,
            },
        ]).to_csv(cluster_path, index=False)

        df, status = load_cluster_assignments(str(cluster_path))

        assert status == "ok"
        assert len(df) == 2
        assert df.loc[0, "pipeline_id"] == "payment_batch"
        assert set(df["cluster"]) == {"HEALTHY", "DRIFTING"}

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def make_bilateral_gap_events():
    base = datetime(2026, 5, 31, 6, 0, tzinfo=timezone.utc)

    producer_df = pd.DataFrame([
        {
            "pipeline_run_id": "run_1",
            "activity": "DATA_AVAILABLE",
            "timestamp": base + pd.Timedelta(minutes=50),
            "team": "producer",
        },
        {
            "pipeline_run_id": "run_2",
            "activity": "DATA_AVAILABLE",
            "timestamp": base + pd.Timedelta(minutes=55),
            "team": "producer",
        },
    ])

    consumer_df = pd.DataFrame([
        {
            "pipeline_run_id": "run_1",
            "activity": "DATA_AVAILABLE",
            "timestamp": base + pd.Timedelta(minutes=78),
            "team": "consumer",
        },
        {
            "pipeline_run_id": "run_2",
            "activity": "DATA_AVAILABLE",
            "timestamp": base + pd.Timedelta(minutes=85),
            "team": "consumer",
        },
    ])

    return producer_df, consumer_df

def test_compute_bilateral_gap_points_returns_matched_gaps():
    producer_df, consumer_df = make_bilateral_gap_events()

    gap_df, status = compute_bilateral_gap_points(producer_df, consumer_df)

    assert status == "ok"
    assert len(gap_df) == 2
    assert gap_df.loc[0, "gap_minutes"] == 28.0
    assert gap_df.loc[1, "gap_minutes"] == 30.0


def test_compute_bilateral_gap_points_handles_producer_only_mode():
    producer_df, _ = make_bilateral_gap_events()

    gap_df, status = compute_bilateral_gap_points(producer_df, None)

    assert gap_df.empty
    assert status == "producer_only"


def test_save_bilateral_gap_timeline_writes_png():
    tmp = Path(tempfile.mkdtemp())
    try:
        producer_df, consumer_df = make_bilateral_gap_events()

        path, status = save_bilateral_gap_timeline(
            producer_df=producer_df,
            consumer_df=consumer_df,
            pipeline_id="payment_batch",
            output_dir=str(tmp / "outputs" / "visualizations"),
        )

        assert status == "ok"
        assert path is not None
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_cmd_visualize_gap_writes_png_from_input_files():
    tmp = Path(tempfile.mkdtemp())
    try:
        inputs_dir = tmp / "inputs"
        output_dir = tmp / "outputs" / "visualizations"
        pipeline_dir = inputs_dir / "payment_batch"
        pipeline_dir.mkdir(parents=True)

        producer_df, consumer_df = make_bilateral_gap_events()

        # The CLI loads the same input layout as a real Fracture run.
        producer_df.to_parquet(pipeline_dir / "producer_20260531.parquet")
        consumer_df.to_parquet(pipeline_dir / "consumer_20260531.parquet")

        args = Namespace(
            key=None,
            pipeline_id="payment_batch",
            date="20260531",
            kind="gap",
            inputs_dir=str(inputs_dir),
            contracts_dir=str(tmp / "contracts"),
            output_dir=str(output_dir),
        )

        exit_code = cmd_visualize(args)

        expected_path = (
            output_dir / "payment_batch" / "bilateral_gap_timeline.png"
        )

        assert exit_code == 0
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cmd_visualize_all_writes_gap_and_drift_png():
    tmp = Path(tempfile.mkdtemp())
    try:
        inputs_dir = tmp / "inputs"
        output_dir = tmp / "outputs" / "visualizations"
        log_path = tmp / "conformance_log.csv"
        pipeline_dir = inputs_dir / "payment_batch"
        pipeline_dir.mkdir(parents=True)

        producer_df, consumer_df = make_bilateral_gap_events()

        # Gap visualization reads raw producer/consumer event logs.
        producer_df.to_parquet(pipeline_dir / "producer_20260531.parquet")
        consumer_df.to_parquet(pipeline_dir / "consumer_20260531.parquet")

        # Drift visualization reads historical scores from conformance_log.csv.
        make_drift_history().to_csv(log_path, index=False)

        args = Namespace(
            key=None,
            pipeline_id="payment_batch",
            date="20260531",
            kind="all",
            inputs_dir=str(inputs_dir),
            contracts_dir=str(tmp / "contracts"),
            output_dir=str(output_dir),
            log_path=str(log_path),
        )

        exit_code = cmd_visualize(args)

        expected_gap_path = (
            output_dir / "payment_batch" / "bilateral_gap_timeline.png"
        )
        expected_drift_path = (
            output_dir / "payment_batch" / "drift_chart.png"
        )

        assert exit_code == 0
        assert expected_gap_path.exists()
        assert expected_gap_path.stat().st_size > 0
        assert expected_drift_path.exists()
        assert expected_drift_path.stat().st_size > 0

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cmd_visualize_rejects_unimplemented_kind():
    args = Namespace(
        key=None,
        pipeline_id="payment_batch",
        date="20260531",
        kind="petri",
        inputs_dir="inputs",
        contracts_dir="contracts",
        output_dir="outputs/visualizations",
    )

    # Petri/dfg/heatmap are planned, but Phase 3 currently implements gap and drift.
    assert cmd_visualize(args) == 1

def make_drift_history():
    # Minimal conformance_log-like data for one declining pipeline.
    return pd.DataFrame([
        {"pipeline_id": "payment_batch", "run_date": "20260501", "final_score": 0.98},
        {"pipeline_id": "payment_batch", "run_date": "20260502", "final_score": 0.94},
        {"pipeline_id": "payment_batch", "run_date": "20260503", "final_score": 0.90},
        {"pipeline_id": "payment_batch", "run_date": "20260504", "final_score": 0.86},
        {"pipeline_id": "other_pipeline", "run_date": "20260504", "final_score": 1.00},
    ])

def make_changepoint_drift_history():
    # Minimal conformance_log-like data with one detected changepoint.
    # This simulates the analytical layer persisting ruptures output into CSV.
    return pd.DataFrame([
        {
            "pipeline_id": "payment_batch",
            "run_date": "20260501",
            "final_score": 0.98,
            "changepoint_detected": False,
            "changepoint_date": "",
        },
        {
            "pipeline_id": "payment_batch",
            "run_date": "20260502",
            "final_score": 0.96,
            "changepoint_detected": False,
            "changepoint_date": "",
        },
        {
            "pipeline_id": "payment_batch",
            "run_date": "20260503",
            "final_score": 0.88,
            "changepoint_detected": True,
            "changepoint_date": "20260503",
        },
        {
            "pipeline_id": "payment_batch",
            "run_date": "20260504",
            "final_score": 0.82,
            "changepoint_detected": True,
            "changepoint_date": "20260503",
        },
    ])

def test_prepare_drift_history_filters_pipeline_and_scores():
    history, status = prepare_drift_history(
        conformance_df=make_drift_history(),
        pipeline_id="payment_batch",
    )

    assert status == "ok"
    assert len(history) == 4
    assert history["pipeline_id"].nunique() == 1
    assert history["final_score"].iloc[-1] == 0.86


def test_save_drift_chart_writes_png():
    tmp = Path(tempfile.mkdtemp())
    try:
        path, status = save_drift_chart(
            conformance_df=make_drift_history(),
            pipeline_id="payment_batch",
            output_dir=str(tmp / "outputs" / "visualizations"),
        )

        assert status == "ok"
        assert path is not None
        assert path.exists()
        assert path.name == "drift_chart.png"
        assert path.stat().st_size > 0

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cmd_visualize_drift_writes_png_from_conformance_log():
    tmp = Path(tempfile.mkdtemp())
    try:
        output_dir = tmp / "outputs" / "visualizations"
        log_path = tmp / "conformance_log.csv"

        # Drift visual reads conformance_log.csv directly.
        make_drift_history().to_csv(log_path, index=False)

        args = Namespace(
            key=None,
            pipeline_id="payment_batch",
            date=None,
            kind="drift",
            inputs_dir=str(tmp / "inputs"),
            contracts_dir=str(tmp / "contracts"),
            output_dir=str(output_dir),
            log_path=str(log_path),
        )

        exit_code = cmd_visualize(args)

        expected_path = output_dir / "payment_batch" / "drift_chart.png"

        assert exit_code == 0
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_extract_changepoint_dates_deduplicates_detected_dates():
    history, status = prepare_drift_history(
        conformance_df=make_changepoint_drift_history(),
        pipeline_id="payment_batch",
    )

    changepoint_dates = extract_changepoint_dates(history)

    assert status == "ok"
    assert len(changepoint_dates) == 1
    assert str(changepoint_dates[0].date()) == "2026-05-03"


def test_save_drift_chart_with_changepoint_marker_writes_png():
    tmp = Path(tempfile.mkdtemp())
    try:
        path, status = save_drift_chart(
            conformance_df=make_changepoint_drift_history(),
            pipeline_id="payment_batch",
            output_dir=str(tmp / "outputs" / "visualizations"),
        )

        # The test verifies that changepoint-enabled history renders safely.
        # Pixel-level marker validation can be handled later in visual QA.
        assert status == "ok"
        assert path is not None
        assert path.exists()
        assert path.name == "drift_chart.png"
        assert path.stat().st_size > 0

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
    check("load_contracts reads valid contract YAML",
          test_load_contracts_reads_valid_contract_yaml)
    check("load_cluster_assignments missing file returns empty DataFrame",
          test_load_cluster_assignments_missing_file_returns_empty_dataframe)
    check("load_cluster_assignments reads existing CSV",
          test_load_cluster_assignments_reads_existing_csv)
    check("compute_bilateral_gap_points returns matched gaps",
          test_compute_bilateral_gap_points_returns_matched_gaps)
    check("compute_bilateral_gap_points handles producer-only mode",
          test_compute_bilateral_gap_points_handles_producer_only_mode)
    check("save_bilateral_gap_timeline writes PNG",
          test_save_bilateral_gap_timeline_writes_png)
    check("cmd_visualize gap writes PNG from input files",
          test_cmd_visualize_gap_writes_png_from_input_files)
    check("cmd_visualize all writes gap and drift PNGs",
          test_cmd_visualize_all_writes_gap_and_drift_png)
    check("cmd_visualize rejects unimplemented kind",
          test_cmd_visualize_rejects_unimplemented_kind)
    check("prepare_drift_history filters pipeline score history",
          test_prepare_drift_history_filters_pipeline_and_scores)
    check("save_drift_chart writes PNG",
          test_save_drift_chart_writes_png)
    check("cmd_visualize drift writes PNG from conformance log",
          test_cmd_visualize_drift_writes_png_from_conformance_log)
    check("extract_changepoint_dates deduplicates detected dates",
          test_extract_changepoint_dates_deduplicates_detected_dates)
    check("save_drift_chart with changepoint marker writes PNG",
          test_save_drift_chart_with_changepoint_marker_writes_png)

    print()
    print(f"Results: {passed + failed} tests  v {passed}  x {failed}")

    if failed:
        raise SystemExit(1)