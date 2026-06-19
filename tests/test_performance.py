"""
tests/test_performance.py

Tests for Performance Analysis.

These tests verify that Fracture can compute arc durations, run durations, and
the slowest p95 bottleneck from actual event logs.
"""

from __future__ import annotations

import sys
import shutil
import tempfile
import yaml
from argparse import Namespace
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from fracture.cli import cmd_performance
from fracture.performance import build_performance_summary
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


def make_contract(deduplicate_retries=False):
    """Build a valid active contract for performance tests."""
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
            # source_path is required by schema even when tests pass DataFrames.
            "source_path": "inputs/payment_batch/",
            "required_events": [
                "SCHEDULED",
                "STARTED",
                "COMPLETED",
                "DATA_AVAILABLE",
            ],
            "terminal_event": "DATA_AVAILABLE",
            "deduplicate_retries": deduplicate_retries,
            "grain": "pipeline",
        },
    )


def make_timed_events():
    """
    Build two traces with deliberately different arc durations.

    STARTED -> COMPLETED is the bottleneck because it has the highest p95.
    """
    base = datetime(2026, 5, 31, 6, 0, tzinfo=timezone.utc)
    rows = []

    traces = {
        "run_1": [
            ("SCHEDULED", 0),
            ("STARTED", 5),
            ("COMPLETED", 25),
            ("DATA_AVAILABLE", 27),
        ],
        "run_2": [
            ("SCHEDULED", 0),
            ("STARTED", 7),
            ("COMPLETED", 32),
            ("DATA_AVAILABLE", 35),
        ],
    }

    for run_index, (run_id, events) in enumerate(traces.items()):
        for activity, minute_offset in events:
            rows.append({
                "pipeline_run_id": run_id,
                "activity": activity,
                "timestamp": base + timedelta(days=run_index, minutes=minute_offset),
                "team": "producer",
            })

    return pd.DataFrame(rows)


def test_build_performance_summary_computes_arc_durations():
    contract = make_contract()

    summary = build_performance_summary(make_timed_events(), contract)
    arcs = {
        (arc["source"], arc["target"]): arc
        for arc in summary.arcs
    }

    assert summary.status == "ok"
    assert summary.n_traces == 2
    assert arcs[("SCHEDULED", "STARTED")]["count"] == 2
    assert arcs[("SCHEDULED", "STARTED")]["mean_minutes"] == 6.0
    assert arcs[("STARTED", "COMPLETED")]["mean_minutes"] == 22.5
    assert arcs[("COMPLETED", "DATA_AVAILABLE")]["mean_minutes"] == 2.5


def test_build_performance_summary_identifies_bottleneck_arc():
    contract = make_contract()

    summary = build_performance_summary(make_timed_events(), contract)

    assert summary.bottleneck_arc["source"] == "STARTED"
    assert summary.bottleneck_arc["target"] == "COMPLETED"
    assert summary.bottleneck_arc["p95_minutes"] > 24


def test_build_performance_summary_computes_run_duration():
    contract = make_contract()

    summary = build_performance_summary(make_timed_events(), contract)

    assert summary.run_duration_minutes["count"] == 2
    assert summary.run_duration_minutes["mean_minutes"] == 31.0
    assert summary.run_duration_minutes["p50_minutes"] == 31.0


def test_build_performance_summary_no_input():
    contract = make_contract()

    summary = build_performance_summary(pd.DataFrame(), contract)

    assert summary.status == "no_input"
    assert summary.n_traces == 0
    assert summary.arcs == []
    assert summary.bottleneck_arc is None


def test_cmd_performance_reads_input_csv():
    """
    Verify the public CLI command is wired to the performance module.

    This guards the demo command:
    python -m fracture.cli performance --pipeline-id X --date YYYYMMDD
    """
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

        events = make_timed_events()
        events.to_csv(inputs_dir / "producer_20260531.csv", index=False)

        args = Namespace(
            key=None,
            pipeline_id="payment_batch",
            date="20260531",
            inputs_dir=str(tmp / "inputs"),
            contracts_dir=str(contracts_dir),
        )

        output = StringIO()
        with redirect_stdout(output):
            exit_code = cmd_performance(args)

        text = output.getvalue()
        assert exit_code == 0
        assert "Performance Analysis: payment_batch" in text
        assert "Bottleneck arc:" in text
        assert "STARTED -> COMPLETED" in text

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print()
    print("Phase 5 performance analysis tests")
    print("=" * 50)

    check("performance summary computes arc durations",
          test_build_performance_summary_computes_arc_durations)
    check("performance summary identifies bottleneck arc",
          test_build_performance_summary_identifies_bottleneck_arc)
    check("performance summary computes run duration",
          test_build_performance_summary_computes_run_duration)
    check("performance summary no input",
          test_build_performance_summary_no_input)
    check("cmd_performance reads input CSV",
          test_cmd_performance_reads_input_csv)

    print()
    print(f"Results: {passed + failed} tests  v {passed}  x {failed}")

    if failed:
        sys.exit(1)
