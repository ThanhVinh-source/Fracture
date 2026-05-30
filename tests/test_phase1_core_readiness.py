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

if __name__ == "__main__":
    test_optional_activities_load_from_contract()
    print("✓ optional_activities load from contract")

    test_optional_activity_must_exist_in_required_events()
    print("✓ optional activity must exist in required_events")

    test_terminal_optional_activity_warns()
    print("✓ terminal optional activity warns")

    print("\nPhase 1 core readiness tests passed.")