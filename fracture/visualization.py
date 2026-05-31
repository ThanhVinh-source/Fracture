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
import matplotlib.pyplot as plt

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
        # Parse run_date for charts when possible.
        # If parsing fails, keep the original values so raw tables still work.
        try:
            df["run_date"] = pd.to_datetime(df["run_date"])
        except Exception:
            pass
    
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
    
def load_cluster_assignments(
    cluster_path: str = "cluster_assignments.csv",
) -> tuple[pd.DataFrame, str]:
    """
    Load optional cluster assignments for Fleet Overview visualizations.

    Clustering is not required for Fracture to run. If the file is missing,
    dashboards should still load and simply hide cluster-specific charts.
    """
    path = Path(cluster_path)

    if not path.exists():
        # Empty state: clustering has not been generated yet.
        # This should not block the Fleet Overview dashboard.
        return pd.DataFrame(), f"Missing cluster assignments: {path}"

    try:
        df = pd.read_csv(path)
    except Exception as e:
        # A corrupt optional cluster file should not crash visualization pages.
        return pd.DataFrame(), f"Could not read cluster assignments: {e}"

    return df, "ok"

def compute_bilateral_gap_points(
    producer_df: pd.DataFrame,
    consumer_df: Optional[pd.DataFrame],
    producer_event: str = "DATA_AVAILABLE",
    consumer_event: str = "DATA_AVAILABLE",
) -> tuple[pd.DataFrame, str]:
    """
    Build matched producer-consumer handoff points for the gap timeline.

    Each output row represents one pipeline_run_id where both sides emitted
    the handoff event. Producer-only mode returns an empty DataFrame with a
    clear status so visual code can show a friendly message.
    """
    if producer_df is None or producer_df.empty:
        # Without producer handoff events there is no anchor for the timeline.
        return pd.DataFrame(), "missing_producer_events"

    if consumer_df is None or consumer_df.empty:
        # Producer-only mode is valid, but the bilateral gap is unknown.
        return pd.DataFrame(), "producer_only"

    producer_points = producer_df[producer_df["activity"] == producer_event].copy()
    consumer_points = consumer_df[consumer_df["activity"] == consumer_event].copy()

    if producer_points.empty:
        # The expected handoff marker is missing from producer logs.
        return pd.DataFrame(), f"missing_producer_event:{producer_event}"

    if consumer_points.empty:
        # The expected consumer acknowledgement marker is missing.
        return pd.DataFrame(), f"missing_consumer_event:{consumer_event}"

    producer_points = producer_points[["pipeline_run_id", "timestamp"]].rename(
        columns={"timestamp": "producer_timestamp"}
    )
    consumer_points = consumer_points[["pipeline_run_id", "timestamp"]].rename(
        columns={"timestamp": "consumer_timestamp"}
    )

    # Inner join keeps only runs where both producer and consumer emitted the
    # handoff marker. Unmatched runs are useful later, but the first timeline
    # should show confirmed bilateral gaps only.
    matched = producer_points.merge(
        consumer_points,
        on="pipeline_run_id",
        how="inner",
    )

    if matched.empty:
        return pd.DataFrame(), "no_matched_handoff_events"

    matched["producer_timestamp"] = pd.to_datetime(
        matched["producer_timestamp"], utc=True
    )
    matched["consumer_timestamp"] = pd.to_datetime(
        matched["consumer_timestamp"], utc=True
    )

    matched["gap_minutes"] = (
        matched["consumer_timestamp"] - matched["producer_timestamp"]
    ).dt.total_seconds() / 60.0

    matched = matched.sort_values("producer_timestamp").reset_index(drop=True)

    return matched, "ok"

def save_bilateral_gap_timeline(
    producer_df: pd.DataFrame,
    consumer_df: Optional[pd.DataFrame],
    pipeline_id: str,
    output_dir: str = "outputs/visualizations",
    producer_event: str = "DATA_AVAILABLE",
    consumer_event: str = "DATA_AVAILABLE",
) -> tuple[Optional[Path], str]:
    """
    Save a static bilateral gap timeline PNG for one pipeline.

    The chart shows producer handoff timestamps, consumer acknowledgement
    timestamps, and the waiting gap between them.
    """
    gap_df, status = compute_bilateral_gap_points(
        producer_df=producer_df,
        consumer_df=consumer_df,
        producer_event=producer_event,
        consumer_event=consumer_event,
    )

    if status != "ok":
        # Do not create misleading empty images. Return the status so CLI or
        # dashboard code can explain why the timeline is unavailable.
        return None, status

    output_path = ensure_visualization_dir(
        pipeline_id=pipeline_id,
        output_dir=output_dir,
    ) / "bilateral_gap_timeline.png"

    fig, ax = plt.subplots(figsize=(10, 5))

    y_positions = range(len(gap_df))

    # Horizontal lines make the waiting time visually obvious.
    ax.hlines(
        y=y_positions,
        xmin=gap_df["producer_timestamp"],
        xmax=gap_df["consumer_timestamp"],
        color="#d97706",
        linewidth=2,
        label="handoff gap",
    )

    # Producer marker: when the upstream side says data is available.
    ax.scatter(
        gap_df["producer_timestamp"],
        y_positions,
        color="#2563eb",
        s=40,
        label="producer DATA_AVAILABLE",
        zorder=3,
    )

    # Consumer marker: when the downstream side actually receives it.
    ax.scatter(
        gap_df["consumer_timestamp"],
        y_positions,
        color="#dc2626",
        s=40,
        label="consumer DATA_AVAILABLE",
        zorder=3,
    )

    # Annotate only a manageable number of points to avoid unreadable images.
    max_annotations = min(len(gap_df), 12)
    for idx in range(max_annotations):
        row = gap_df.iloc[idx]
        ax.text(
            row["consumer_timestamp"],
            idx,
            f" {row['gap_minutes']:.1f}m",
            va="center",
            fontsize=8,
            color="#7f1d1d",
        )

    ax.set_title(f"Bilateral Gap Timeline — {pipeline_id}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Pipeline run")
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(gap_df["pipeline_run_id"].astype(str).tolist())
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path, "ok"