"""
tests/test_comparison.py

Tests for Comparative Process Mining.

These tests verify that Fracture can compare process perspectives:
- producer vs consumer handoff gap
- contract model vs actual dominant variant
- current period vs previous period from conformance history
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from argparse import Namespace
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from fracture.cli import cmd_compare
from fracture.comparison import (
    compare_contract_actual,
    compare_periods,
    compare_producer_consumer,
    format_comparison_result,
)
from fracture.schema import PipelineContract


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


def make_contract():
    """
    Build a valid active contract for comparison tests.

    The same object shape is used by unit tests and temporary YAML files.
    """
    return PipelineContract(
        pipeline_id="payment_batch",
        owner="data-team@example.com",
        producer_team="producer",
        consumer_team="consumer",
        criticality="medium",
        status="active",
        expected_start="06:00",
        expected_end="07:00",
        grace_minutes=10,
        p50_minutes=20,
        p95_minutes=40,
        p99_minutes=45,
        log_contract={
            # source_path is required by schema even when tests pass
            # DataFrames directly instead of reading files.
            "source_path": "inputs/payment_batch/",
            "required_events": [
                "SCHEDULED",
                "STARTED",
                "COMPLETED",
                "DATA_AVAILABLE",
            ],
            "terminal_event": "DATA_AVAILABLE",
            "upstream_producer_event": "DATA_AVAILABLE",
            "upstream_consumer_event": "DATA_AVAILABLE",
            "grain": "pipeline",
        },
    )


def make_handoff_events(offsets_by_run, team):
    """
    Build minimal handoff event logs for producer-consumer comparison.

    Each run has one DATA_AVAILABLE marker; comparison only needs the handoff
    timestamp, not the full lifecycle.
    """
    base = datetime(2026, 5, 31, 6, 0, tzinfo=timezone.utc)
    rows = []

    for run_id, minute_offset in offsets_by_run.items():
        rows.append({
            "pipeline_run_id": run_id,
            "activity": "DATA_AVAILABLE",
            "timestamp": base + timedelta(minutes=minute_offset),
            "team": team,
        })

    return pd.DataFrame(rows)


def make_variant_events(variants_by_run, team="producer"):
    """
    Convert activity lists into full Fracture traces.

    This is used by contract-vs-actual comparison because that mode needs a
    complete ordered sequence, not only a handoff timestamp.
    """
    base = datetime(2026, 5, 31, 6, 0, tzinfo=timezone.utc)
    rows = []

    for run_index, (run_id, activities) in enumerate(variants_by_run.items()):
        for activity_index, activity in enumerate(activities):
            rows.append({
                "pipeline_run_id": run_id,
                "activity": activity,
                "timestamp": base + timedelta(days=run_index, minutes=activity_index),
                "team": team,
            })

    return pd.DataFrame(rows)


def test_compare_producer_consumer_detects_gap():
    contract = make_contract()
    producer = make_handoff_events({"run_1": 30, "run_2": 40}, team="producer")
    consumer = make_handoff_events({"run_1": 42, "run_2": 62}, team="consumer")

    result = compare_producer_consumer(producer, consumer, contract)

    assert result.status == "ok"
    assert result.metrics["matched_runs"] == 2
    assert result.metrics["mean_gap_minutes"] == 17.0
    assert result.metrics["p95_gap_minutes"] > 20
    assert any("severe" in finding for finding in result.findings)


def test_compare_producer_consumer_flags_negative_gap():
    contract = make_contract()
    producer = make_handoff_events({"run_1": 30}, team="producer")
    consumer = make_handoff_events({"run_1": 25}, team="consumer")

    result = compare_producer_consumer(producer, consumer, contract)

    assert result.status == "ok"
    assert result.metrics["negative_gap_count"] == 1
    assert any("clock sync" in finding for finding in result.findings)


def test_compare_contract_actual_matches_contract_path():
    contract = make_contract()
    events = make_variant_events({
        "run_1": ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"],
        "run_2": ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"],
    })

    result = compare_contract_actual(events, contract)

    assert result.status == "ok"
    assert result.metrics["dominant_matches_contract"] is True
    assert result.findings == ["Dominant actual variant matches the contract path."]


def test_compare_contract_actual_reports_missing_activity():
    contract = make_contract()
    events = make_variant_events({
        "run_1": ["SCHEDULED", "STARTED", "DATA_AVAILABLE"],
        "run_2": ["SCHEDULED", "STARTED", "DATA_AVAILABLE"],
    })

    result = compare_contract_actual(events, contract)

    assert result.status == "ok"
    assert result.metrics["dominant_matches_contract"] is False
    assert any("missing required activities: COMPLETED" in finding for finding in result.findings)


def test_compare_periods_detects_score_drop_and_gap_widening():
    rows = []
    start = datetime(2026, 6, 1)

    for i in range(14):
        previous_period = i < 7
        rows.append({
            "pipeline_id": "payment_batch",
            "run_date": (start + timedelta(days=i)).strftime("%Y%m%d"),
            "final_score": 1.0 if previous_period else 0.9,
            "bilateral_gap_minutes": 5.0 if previous_period else 12.0,
        })

    result = compare_periods(pd.DataFrame(rows), "payment_batch", period_size=7)

    assert result.status == "ok"
    assert result.metrics["score_delta"] == -0.1
    assert result.metrics["gap_delta_minutes"] == 7.0
    assert any("score dropped" in finding for finding in result.findings)
    assert any("gap widened" in finding for finding in result.findings)


def test_format_comparison_result_is_human_readable():
    contract = make_contract()
    producer = make_handoff_events({"run_1": 30}, team="producer")
    consumer = make_handoff_events({"run_1": 42}, team="consumer")
    result = compare_producer_consumer(producer, consumer, contract)

    text = format_comparison_result(result)

    assert "Comparative Process Mining: payment_batch" in text
    assert "Mode: producer-consumer" in text
    assert "mean_gap_minutes" in text


def test_cmd_compare_producer_consumer_reads_input_csv():
    tmp = Path(tempfile.mkdtemp())
    try:
        contracts_dir = tmp / "contracts"
        inputs_dir = tmp / "inputs" / "payment_batch"
        contracts_dir.mkdir(parents=True)
        inputs_dir.mkdir(parents=True)

        (contracts_dir / "payment_batch.yaml").write_text(
            yaml.safe_dump(make_contract().model_dump(mode="json")),
            encoding="utf-8",
        )

        producer = make_handoff_events({"run_1": 30, "run_2": 40}, team="producer")
        consumer = make_handoff_events({"run_1": 42, "run_2": 62}, team="consumer")
        producer.to_csv(inputs_dir / "producer_20260531.csv", index=False)
        consumer.to_csv(inputs_dir / "consumer_20260531.csv", index=False)

        args = Namespace(
            key=None,
            pipeline_id="payment_batch",
            date="20260531",
            mode="producer-consumer",
            side="producer",
            period_size=7,
            log_path=str(tmp / "conformance_log.csv"),
            inputs_dir=str(tmp / "inputs"),
            contracts_dir=str(contracts_dir),
        )

        output = StringIO()
        with redirect_stdout(output):
            exit_code = cmd_compare(args)

        assert exit_code == 0
        assert "Comparative Process Mining: payment_batch" in output.getvalue()
        assert "mean_gap_minutes: 17.0" in output.getvalue()

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print()
    print("Phase 5 comparative process mining tests")
    print("=" * 50)

    check("compare producer-consumer detects gap",
          test_compare_producer_consumer_detects_gap)
    check("compare producer-consumer flags negative gap",
          test_compare_producer_consumer_flags_negative_gap)
    check("compare contract-actual matches contract path",
          test_compare_contract_actual_matches_contract_path)
    check("compare contract-actual reports missing activity",
          test_compare_contract_actual_reports_missing_activity)
    check("compare periods detects score drop and gap widening",
          test_compare_periods_detects_score_drop_and_gap_widening)
    check("format comparison result is human readable",
          test_format_comparison_result_is_human_readable)
    check("cmd_compare producer-consumer reads input CSV",
          test_cmd_compare_producer_consumer_reads_input_csv)

    print()
    print(f"Results: {passed + failed} tests  v {passed}  x {failed}")

    if failed:
        sys.exit(1)
