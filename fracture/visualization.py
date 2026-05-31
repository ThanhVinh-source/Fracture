"""
fracture/visualization.py

Safe data-loading helpers for Fracture visualizations.

This module does not build the Streamlit dashboard yet. It prepares the
data layer that future charts and dashboard pages will use.

Design goal:
- Missing files should not crash visualization code.
- Producer-only pipelines should still be readable.
- Dashboard code should receive clear status messages instead of exceptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from fracture.ingest import load_pipeline_events as ingest_load_pipeline_events
from fracture.schema import load_contract

def ensure_visualization_dir (
        pipeline_id: Optional[str] = None,
        output_dir: str = "outputs/visualizations",
) -> Path:
    """
    Create and return the standard visualization output directory.

    Fleet-level charts use outputs/visualizations/.
    Pipeline-level charts use outputs/visualizations/{pipeline_id}/.
    """
    base = Path(output_dir)
    
    # Pipeline-specific visuals should live in a stable per-pipeline folder.
    path = base / pipeline_id if pipeline_id else base
    path.mkdir(parents=True, exist_ok=True)

    return path

def load_conformance_log(
        log_path: str = "conformance_log.csv",
) -> tuple[pd.DataFrame, str]:
    """
    Load conformance_log.csv for dashboard and chart use.

    Returns an empty DataFrame when the file does not exist, so the dashboard
    can show a friendly setup message instead of crashing.
    """
    path = Path(log_path)

    if not path.exists():
        # Empty state: Fracture has not run yet or no results were logged
        return pd.DataFrame(), f"Missing conformance log: {path}"
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        # Corrupt CSV shound not crash a dashboard page.
        return pd.DataFrame(), f"Could not read conformance log: {e}"
    
    if "run_date" in df.columns:
        # Keep original value if parsing fails; charts can still show raw rows.
        df["run_date"] = pd.to_datetime(df["run_date"], errors="ignore")
    
    return df, "ok"

def load_contracts(
    contracts_dir: str = "contracts",
) -> tuple[pd.DataFrame, str]:
    """
    Load all YAML contracts into a dashboard-friendly DataFrame.

    The dashboard uses this for Fleet Overview and pipeline selectors.
    Missing or invalid contracts are skipped with a readable status message.
    """
    root = Path(contracts_dir)

    if not root.exists():
        # Empty state: project may not have registered pipelines yet.
        return pd.DataFrame(), f"Missing contracts directory: {root}"

    rows = []
    errors = []

    for path in sorted(root.glob("*.yaml")):
        try:
            contract = load_contract(path)
            rows.append({
                "pipeline_id": contract.pipeline_id,
                "owner": contract.owner,
                "producer_team": contract.producer_team,
                "consumer_team": contract.consumer_team,
                "status": contract.status.value,
                "criticality": contract.criticality.value,
                "expected_start": contract.expected_start,
                "expected_end": contract.expected_end,
                "grain": contract.log_contract.grain,
                "parent_grain": contract.log_contract.parent_grain or "",
                "required_events": " → ".join(contract.log_contract.required_events),
                "optional_activities": " → ".join(contract.log_contract.optional_activities),
            })
        except Exception as e:
            # One bad contract should not prevent the whole fleet dashboard loading.
            errors.append(f"{path.name}: {e}")

    status = "ok" if not errors else "loaded_with_errors: " + "; ".join(errors)
    return pd.DataFrame(rows), status


def load_pipeline_events(
    inputs_dir: str = "inputs",
    pipeline_id: Optional[str] = None,
    date_str: Optional[str] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
    """
    Load producer and optional consumer events for one pipeline.

    This wraps the existing ingestion loader but converts file errors into
    dashboard-safe empty states.
    """
    if not pipeline_id:
        # Dashboard page has not selected a pipeline yet.
        return pd.DataFrame(), None, "No pipeline selected"

    try:
        producer_df, consumer_df = ingest_load_pipeline_events(
            pipeline_id=pipeline_id,
            inputs_dir=inputs_dir,
            date_str=date_str,
        )
        if consumer_df is None:
            return producer_df, None, "producer_only"

        return producer_df, consumer_df, "ok"

    except FileNotFoundError as e:
        # Missing input files are common before a demo/user has run extraction.
        return pd.DataFrame(), None, f"missing_input: {e}"

    except ValueError as e:
        # Bad event format should be displayed as a data-quality/setup issue.
        return pd.DataFrame(), None, f"invalid_input: {e}"