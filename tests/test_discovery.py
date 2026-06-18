"""
tests/test_discovery.py

Phase 5 Step 1 tests for Process Discovery.

These tests verify that Fracture can read actual event traces, discover
variants, identify the dominant actual path, and compare it with the contract.
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

from fracture.cli import cmd_discover
from fracture.discovery import discover_variants, format_discovery_summary
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


def make_contract(
    required_events=None,
    optional_activities=None,
    activity_name_map=None,
    deduplicate_retries=False,
):
    """
    Build a valid active contract for discovery tests.

    The contract mirrors real Fracture YAML but stays small enough for focused
    unit tests.
    """
    required_events = required_events or [
        "SCHEDULED",
        "STARTED",
        "COMPLETED",
        "DATA_AVAILABLE",
    ]

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
            # source_path is required by schema even when the unit test passes
            # DataFrames directly instead of reading files.
            "source_path": "inputs/payment_batch/",
            "required_events": required_events,
            "optional_activities": optional_activities or [],
            "terminal_event": "DATA_AVAILABLE",
            "activity_name_map": activity_name_map or {},
            "deduplicate_retries": deduplicate_retries,
            "grain": "pipeline",
        },
    )


def make_events(variants_by_run, team="producer"):
    """
    Convert simple activity lists into a Fracture event DataFrame.

    Each key is one pipeline_run_id. Each activity list becomes one trace.
    """
    rows = []
    base = datetime(2026, 5, 31, 6, 0, tzinfo=timezone.utc)

    for run_index, (run_id, activities) in enumerate(variants_by_run.items()):
        for activity_index, activity in enumerate(activities):
            rows.append({
                "pipeline_run_id": run_id,
                "activity": activity,
                "timestamp": base + timedelta(days=run_index, minutes=activity_index),
                "team": team,
            })

    return pd.DataFrame(rows)


def make_contract_yaml():
    """YAML-shaped contract used by the CLI integration test."""
    return make_contract().model_dump(mode="json")


def test_discover_healthy_dominant_variant_matches_contract():
    contract = make_contract()
    events = make_events({
        "run_1": ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"],
        "run_2": ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"],
    })

    summary = discover_variants(events, contract)

    assert summary.status == "ok"
    assert summary.n_traces == 2
    assert summary.n_variants == 1
    assert summary.dominant_matches_contract is True
    assert summary.deviations == []


def test_discover_activity_name_map_normalizes_variant():
    contract = make_contract(
        activity_name_map={
            "job_scheduled": "SCHEDULED",
            "job_started": "STARTED",
            "job_completed": "COMPLETED",
            "file_ready": "DATA_AVAILABLE",
        }
    )
    events = make_events({
        "run_1": ["job_scheduled", "job_started", "job_completed", "file_ready"],
    })

    summary = discover_variants(events, contract)

    assert summary.dominant_variant == [
        "SCHEDULED",
        "STARTED",
        "COMPLETED",
        "DATA_AVAILABLE",
    ]
    assert summary.dominant_matches_contract is True


def test_discover_repeated_activity_reports_deviation():
    contract = make_contract(deduplicate_retries=False)
    events = make_events({
        "run_1": [
            "SCHEDULED",
            "STARTED",
            "STARTED",
            "COMPLETED",
            "DATA_AVAILABLE",
        ],
    })

    summary = discover_variants(events, contract)

    assert summary.dominant_matches_contract is False
    assert any("repeated activities observed: STARTED" in d for d in summary.deviations)


def test_discover_deduplicate_retries_recovers_contract_path():
    contract = make_contract(deduplicate_retries=True)
    events = make_events({
        "run_1": [
            "SCHEDULED",
            "STARTED",
            "STARTED",
            "COMPLETED",
            "DATA_AVAILABLE",
        ],
    })

    summary = discover_variants(events, contract)

    assert summary.dominant_variant == [
        "SCHEDULED",
        "STARTED",
        "COMPLETED",
        "DATA_AVAILABLE",
    ]
    assert summary.dominant_matches_contract is True


def test_discover_optional_activity_can_be_skipped():
    contract = make_contract(
        required_events=[
            "SCHEDULED",
            "VALIDATED",
            "STARTED",
            "COMPLETED",
            "DATA_AVAILABLE",
        ],
        optional_activities=["VALIDATED"],
    )
    events = make_events({
        "run_1": ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"],
    })

    summary = discover_variants(events, contract)

    # VALIDATED is optional, so the comparison path removes it for this variant.
    assert summary.comparison_path == [
        "SCHEDULED",
        "STARTED",
        "COMPLETED",
        "DATA_AVAILABLE",
    ]
    assert summary.dominant_matches_contract is True


def test_discover_missing_activity_reports_deviation():
    contract = make_contract()
    events = make_events({
        "run_1": ["SCHEDULED", "STARTED", "DATA_AVAILABLE"],
    })

    summary = discover_variants(events, contract)

    assert summary.dominant_matches_contract is False
    assert any("missing required activities: COMPLETED" in d for d in summary.deviations)


def test_discover_no_input_returns_no_input():
    contract = make_contract()

    summary = discover_variants(pd.DataFrame(), contract)

    assert summary.status == "no_input"
    assert summary.n_traces == 0
    assert summary.deviations == ["no input events available for discovery"]


def test_format_discovery_summary_is_human_readable():
    contract = make_contract()
    events = make_events({
        "run_1": ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"],
    })
    summary = discover_variants(events, contract)

    text = format_discovery_summary(summary)

    assert "Process Discovery: payment_batch" in text
    assert "Dominant variant:" in text
    assert "Dominant path matches contract: yes" in text


def test_cmd_discover_reads_input_csv():
    tmp = Path(tempfile.mkdtemp())
    try:
        contracts_dir = tmp / "contracts"
        inputs_dir = tmp / "inputs" / "payment_batch"
        contracts_dir.mkdir(parents=True)
        inputs_dir.mkdir(parents=True)

        (contracts_dir / "payment_batch.yaml").write_text(
            yaml.safe_dump(make_contract_yaml()),
            encoding="utf-8",
        )

        events = make_events({
            "run_1": ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"],
            "run_2": ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"],
        })
        events.to_csv(inputs_dir / "producer_20260531.csv", index=False)

        args = Namespace(
            key=None,
            pipeline_id="payment_batch",
            date="20260531",
            side="producer",
            inputs_dir=str(tmp / "inputs"),
            contracts_dir=str(contracts_dir),
        )

        output = StringIO()
        with redirect_stdout(output):
            exit_code = cmd_discover(args)

        assert exit_code == 0
        assert "Process Discovery: payment_batch" in output.getvalue()
        assert "Dominant path matches contract: yes" in output.getvalue()

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print()
    print("Phase 5 process discovery tests")
    print("=" * 50)

    check("discover healthy dominant variant matches contract",
          test_discover_healthy_dominant_variant_matches_contract)
    check("discover activity_name_map normalizes variant",
          test_discover_activity_name_map_normalizes_variant)
    check("discover repeated activity reports deviation",
          test_discover_repeated_activity_reports_deviation)
    check("discover deduplicate_retries recovers contract path",
          test_discover_deduplicate_retries_recovers_contract_path)
    check("discover optional activity can be skipped",
          test_discover_optional_activity_can_be_skipped)
    check("discover missing activity reports deviation",
          test_discover_missing_activity_reports_deviation)
    check("discover no input returns no_input",
          test_discover_no_input_returns_no_input)
    check("format discovery summary is human readable",
          test_format_discovery_summary_is_human_readable)
    check("cmd_discover reads input CSV",
          test_cmd_discover_reads_input_csv)

    print()
    print(f"Results: {passed + failed} tests  v {passed}  x {failed}")

    if failed:
        sys.exit(1)
