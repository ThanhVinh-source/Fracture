"""
Phase 1 core readiness tests.

These tests protect the contract-to-ingestion-to-conformance path before
Fracture adds dashboard and visualization layers.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from fracture.schema import LogContract, PipelineContract, Criticality
from fracture.ingest import normalize_events


def make_contract(**log_updates):
    log_data = {
        "source_path": "inputs/test_pipeline/",
        "required_events": [
            "SCHEDULED",
            "STARTED",
            "VALIDATED",
            "COMPLETED",
            "DATA_AVAILABLE",
        ],
        "terminal_event": "COMPLETED",
        "activity_name_map": {},
        "deduplicate_retries": False,
    }
    log_data.update(log_updates)

    return PipelineContract(
        pipeline_id="test_pipeline",
        owner="owner@example.com",
        producer_team="producer",
        consumer_team="consumer",
        expected_start="06:00",
        expected_end="08:00",
        criticality=Criticality.MEDIUM,
        grace_minutes=30,
        p50_minutes=45,
        p95_minutes=60,
        p99_minutes=75,
        log_contract=LogContract(**log_data),
    )


def test_optional_activities_load_from_contract():
    contract = make_contract(optional_activities=["VALIDATED"])

    assert contract.log_contract.optional_activities == ["VALIDATED"]


def test_optional_activity_must_exist_in_required_events():
    with pytest.raises(ValueError, match="optional_activities"):
        make_contract(optional_activities=["NOT_IN_PROCESS"])


def test_terminal_optional_activity_warns():
    with pytest.warns(UserWarning, match="terminal_event"):
        make_contract(optional_activities=["COMPLETED"])

def make_events():
    base = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    return pd.DataFrame([
        {
            "pipeline_run_id": "run_1",
            "activity": "job_scheduled",
            "timestamp": base,
            "team": "producer",
        },
        {
            "pipeline_run_id": "run_1",
            "activity": "job_started",
            "timestamp": base + timedelta(minutes=1),
            "team": "producer",
        },
        {
            "pipeline_run_id": "run_1",
            "activity": "job_done",
            "timestamp": base + timedelta(minutes=40),
            "team": "producer",
        },
    ])


def test_activity_name_map_normalizes_raw_activity_names():
    contract = make_contract(
        required_events=["SCHEDULED", "STARTED", "COMPLETED"],
        terminal_event="COMPLETED",
        activity_name_map={
            "job_scheduled": "SCHEDULED",
            "job_started": "STARTED",
            "job_done": "COMPLETED",
        },
    )

    normalized = normalize_events(make_events(), contract)

    assert normalized["activity"].tolist() == [
        "SCHEDULED",
        "STARTED",
        "COMPLETED",
    ]


def test_deduplicate_retries_keeps_latest_duplicate_activity():
    base = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    events = pd.DataFrame([
        {"pipeline_run_id": "run_1", "activity": "SCHEDULED", "timestamp": base, "team": "producer"},
        {"pipeline_run_id": "run_1", "activity": "STARTED", "timestamp": base + timedelta(minutes=1), "team": "producer"},
        {"pipeline_run_id": "run_1", "activity": "STARTED", "timestamp": base + timedelta(minutes=2), "team": "producer"},
        {"pipeline_run_id": "run_1", "activity": "COMPLETED", "timestamp": base + timedelta(minutes=40), "team": "producer"},
    ])

    contract = make_contract(
        required_events=["SCHEDULED", "STARTED", "COMPLETED"],
        terminal_event="COMPLETED",
        deduplicate_retries=True,
    )

    normalized = normalize_events(events, contract)

    assert normalized["activity"].tolist() == [
        "SCHEDULED",
        "STARTED",
        "COMPLETED",
    ]
    assert normalized.loc[normalized["activity"] == "STARTED", "timestamp"].iloc[0] == (
        base + timedelta(minutes=2)
    )

if __name__ == "__main__":
    test_optional_activities_load_from_contract()
    print("✓ optional_activities load from contract")

    test_optional_activity_must_exist_in_required_events()
    print("✓ optional activity must exist in required_events")

    test_terminal_optional_activity_warns()
    print("✓ terminal optional activity warns")
    
    test_activity_name_map_normalizes_raw_activity_names()
    print("✓ activity_name_map normalizes raw activity names")

    test_deduplicate_retries_keeps_latest_duplicate_activity()
    print("✓ deduplicate_retries keeps latest duplicate activity")

    print("\nPhase 1 core readiness tests passed.")