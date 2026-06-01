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
import numpy as np

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

def _parse_run_date_series(values: pd.Series) -> pd.Series:
    """
    Parse Fracture run_date values safely for charts.

    Fracture usually stores run_date as YYYYMMDD, for example 20260531.
    Reading CSV can turn that into an integer, so we parse through string form
    to avoid pandas treating it as a Unix timestamp.
    """
    # Convert every value to clean text before date parsing.
    as_text = values.astype(str).str.strip()

    # Compact Fracture dates use YYYYMMDD format.
    compact_mask = as_text.str.fullmatch(r"\d{8}", na=False)

    # Start with empty parsed dates so invalid rows become NaT, not crashes.
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    # Parse standard Fracture dates explicitly.
    if compact_mask.any():
        parsed.loc[compact_mask] = pd.to_datetime(
            as_text.loc[compact_mask],
            format="%Y%m%d",
            errors="coerce",
        )

    # Parse any other readable date format, such as 2026-05-31.
    if (~compact_mask).any():
        parsed.loc[~compact_mask] = pd.to_datetime(
            as_text.loc[~compact_mask],
            errors="coerce",
        )

    return parsed

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
        # Parse run_date in a Fracture-safe way for drift charts.
        df["run_date"] = _parse_run_date_series(df["run_date"])
    
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

def prepare_drift_history(
    conformance_df: pd.DataFrame,
    pipeline_id: str,
) -> tuple[pd.DataFrame, str]:
    """
    Prepare score history for one pipeline's drift chart.

    The drift chart uses conformance_log.csv, not raw input event files.
    """
    required_columns = {"pipeline_id", "run_date", "final_score"}
    missing_columns = required_columns - set(conformance_df.columns)

    if missing_columns:
        # Dashboard/CLI should explain bad logs instead of crashing.
        return pd.DataFrame(), f"missing_columns:{sorted(missing_columns)}"

    # Keep only the selected pipeline.
    history = conformance_df[conformance_df["pipeline_id"] == pipeline_id].copy()

    if history.empty:
        # The pipeline has no logged conformance results yet.
        return pd.DataFrame(), "no_history_for_pipeline"

    # Convert score values to numeric; failed runs may contain empty cells.
    history["final_score"] = pd.to_numeric(
        history["final_score"],
        errors="coerce",
    )

    # Ensure run_date is parsed even if caller passed a raw DataFrame.
    history["run_date"] = _parse_run_date_series(history["run_date"])

    # Drop rows that cannot be plotted.
    history = history.dropna(subset=["run_date", "final_score"])

    if history.empty:
        # There were rows, but none had usable date + score values.
        return pd.DataFrame(), "no_plottable_score_history"

    # Sort by time so line charts and trend lines are meaningful.
    history = history.sort_values("run_date").reset_index(drop=True)

    return history, "ok"


def save_drift_chart(
    conformance_df: pd.DataFrame,
    pipeline_id: str,
    output_dir: str = "outputs/visualizations",
    healthy_threshold: float = 0.85,
    critical_threshold: float = 0.70,
) -> tuple[Optional[Path], str]:
    """
    Save a static drift chart PNG for one pipeline.

    The chart shows final_score over time and highlights GREEN/AMBER/RED zones.
    """
    history, status = prepare_drift_history(
        conformance_df=conformance_df,
        pipeline_id=pipeline_id,
    )

    if status != "ok":
        # Do not create a misleading empty chart.
        return None, status

    output_path = ensure_visualization_dir(
        pipeline_id=pipeline_id,
        output_dir=output_dir,
    ) / "drift_chart.png"

    scores = history["final_score"].astype(float)
    y_max = max(1.05, float(scores.max()) + 0.05)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Background zones make score health readable without reading numbers.
    ax.axhspan(0.00, critical_threshold, color="#fee2e2", alpha=0.8, label="RED")
    ax.axhspan(critical_threshold, healthy_threshold, color="#fef3c7", alpha=0.8, label="AMBER")
    ax.axhspan(healthy_threshold, y_max, color="#dcfce7", alpha=0.8, label="GREEN")

    # Main score trajectory.
    ax.plot(
        history["run_date"],
        scores,
        marker="o",
        linewidth=2,
        color="#2563eb",
        label="final_score",
    )

    summary_lines = [f"Runs: {len(history)}"]

    if len(history) >= 2:
        # Fit a simple linear trend over run order.
        x = np.arange(len(history))
        y = scores.to_numpy()
        slope_per_run, intercept = np.polyfit(x, y, 1)
        trend_y = intercept + slope_per_run * x

        ax.plot(
            history["run_date"],
            trend_y,
            linestyle="--",
            linewidth=2,
            color="#111827",
            label="trend",
        )

        if slope_per_run < -0.001:
            summary_lines.append(f"Trend: declining ({slope_per_run:.4f}/run)")
        elif slope_per_run > 0.001:
            summary_lines.append(f"Trend: improving ({slope_per_run:.4f}/run)")
        else:
            summary_lines.append("Trend: stable")

        # Project when score may cross the healthy threshold.
        if len(history) >= 3 and slope_per_run < 0:
            latest_score = float(scores.iloc[-1])
            latest_date = history["run_date"].iloc[-1]

            if latest_score <= healthy_threshold:
                summary_lines.append("Projection: already below healthy threshold")
            else:
                day_steps = history["run_date"].diff().dt.days.dropna()
                median_days = max(float(day_steps.median()), 1.0) if not day_steps.empty else 1.0

                runs_until_threshold = (healthy_threshold - latest_score) / slope_per_run
                days_until_threshold = runs_until_threshold * median_days

                if 0 <= days_until_threshold <= 90:
                    projected_date = latest_date + pd.Timedelta(days=days_until_threshold)
                    summary_lines.append(
                        f"Projected AMBER: {projected_date.date()}"
                    )
                else:
                    summary_lines.append("Projection: no near-term AMBER risk")
        else:
            summary_lines.append("Projection: insufficient declining history")
    else:
        # One point is still useful as a baseline, but not enough for a trend.
        summary_lines.append("Trend: insufficient history")
        summary_lines.append("Projection: unavailable")

    ax.text(
        0.02,
        0.03,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )

    ax.set_title(f"Drift Chart — {pipeline_id}")
    ax.set_xlabel("Run date")
    ax.set_ylabel("Final score")
    ax.set_ylim(0, y_max)
    if len(history) == 1:
        # With one point, matplotlib expands the date axis too widely.
        # Keep a small window around the only run date so the chart looks intentional.
        only_date = history["run_date"].iloc[0]
        ax.set_xlim(
            only_date - pd.Timedelta(days=7),
            only_date + pd.Timedelta(days=7),
        )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path, "ok"