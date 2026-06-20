"""
dashboard.py

Streamlit dashboard for Fracture.

This file is the UI layer. It should stay lightweight:
- load data safely
- arrange dashboard pages
- call reusable helpers from fracture.visualization
- avoid duplicating process-mining logic already implemented in Fracture
"""

from __future__ import annotations

from datetime import date, datetime
import json
import os
from pathlib import Path
import re
from typing import Optional

import pandas as pd
import streamlit as st

from fracture.visualization import (
    load_conformance_log,
    load_contracts,
    load_pipeline_events,
    save_bilateral_gap_analysis,
    save_bilateral_gap_timeline,
    save_drift_chart,
    save_fleet_heatmap,
    save_contract_petri_net,
    save_discovered_dfg,
    save_execution_time_drift,
    save_performance_dfg,
)
from fracture.schema import load_contract

VISUALIZATION_ROOT = Path("outputs/visualizations")
DEMO_DATA_ROOT = Path("demo_data")


def resolve_data_root(
    explicit_root: Optional[str] = None,
    base_dir: Path = Path("."),
) -> Path:
    """
    Choose the data folder used by the dashboard.

    Local development keeps Fracture runtime files at the project root
    (`conformance_log.csv`, `contracts/`, `inputs/`). Streamlit Cloud does not
    have those generated files unless they are committed, so the dashboard falls
    back to `demo_data/` when the root runtime log is missing.
    """
    if explicit_root:
        # Environment override is useful for testing and for hosted deployments
        # that want to pin the dashboard to a specific data bundle.
        return Path(explicit_root).expanduser()

    runtime_root = base_dir
    demo_root = base_dir / DEMO_DATA_ROOT

    if (runtime_root / "conformance_log.csv").exists():
        return runtime_root

    if (demo_root / "conformance_log.csv").exists():
        return demo_root

    # Return the normal local root so the empty-state messages still point to
    # the standard local workflow when no data has been generated yet.
    return runtime_root


DATA_ROOT = resolve_data_root(os.getenv("FRACTURE_DATA_ROOT"))
CONFORMANCE_LOG_PATH = DATA_ROOT / "conformance_log.csv"
CONTRACTS_ROOT = DATA_ROOT / "contracts"
INPUTS_ROOT = DATA_ROOT / "inputs"


# Page configuration controls the browser tab title and layout width.
st.set_page_config(
    page_title="Fracture Dashboard",
    page_icon="📊",
    layout="wide",
)


def format_score(value) -> str:
    """
    Format score values for dashboard display.

    conformance_log.csv may contain empty cells for failed or skipped runs.
    This helper avoids crashing when a score is missing.
    """
    if pd.isna(value):
        return "n/a"

    try:
        return f"{float(value):.2f}"
    except Exception:
        return "n/a"


def render_empty_state(message: str):
    """
    Show a friendly setup message instead of a broken dashboard.

    Missing files are normal before the user runs Fracture for the first time.
    """
    st.info(message)
    st.code(
        "python -m fracture.cli run-all\n"
        "python -m fracture.cli visualize --pipeline-id trade_positions_sftp --kind all",
        language="bash",
    )


def render_metric_guide():
    """
    Explain dashboard symbols and labels in one collapsible place.

    The dashboard has several health concepts. Keeping the guide close to the
    data helps readers avoid mixing up final_score bands and timing_zone labels.
    """
    with st.expander("How to read this dashboard", expanded=False):
        st.markdown(
            """
            **Final score** is the overall conformance score:
            `sequence_fitness * 0.35 + timing_score * 0.50 + completeness_score * 0.15`.

            **Score color bands** in the drift chart:
            GREEN means `final_score >= 0.85`, AMBER means `0.70 <= final_score < 0.85`,
            RED means `final_score < 0.70`.

            **Timing zone** is separate from final score:
            GREEN means the run finished before p95, AMBER means between p95 and p99,
            RED means inside grace after p99, BREACH means past p99 plus grace.

            **p50 / p95 / p99** are historical runtime percentiles from the contract.
            p50 is the median runtime, p95 means 95% of past runs completed before that
            duration, and p99 means 99% completed before that duration. Fracture treats
            p95 as the early warning boundary and p99 plus grace as the final breach
            boundary.

            **Confidence** tells how trustworthy the measurement is based on log quality
            and history. LOW or UNRELIABLE means fix the logging before overreacting.

            **Pattern** explains behavior over time: STABLE is consistent, DRIFTING is
            trending worse, and INTERMITTENT clusters around specific days or conditions.

            **Bilateral gap** is the handoff delay:
            consumer `DATA_AVAILABLE` minus producer `DATA_AVAILABLE`.
            """
        )


def explain_timing_score_mismatch(latest: pd.Series):
    """
    Warn when final_score is healthy but timing_zone is not GREEN.

    This is a common interpretation trap: the drift chart colors final_score,
    while timing_zone measures only the timing component of the run.
    """
    final_score = pd.to_numeric(latest.get("final_score"), errors="coerce")
    timing_zone = latest.get("timing_zone", "")

    if pd.isna(final_score):
        return

    if final_score >= 0.85 and timing_zone not in ("", "GREEN"):
        st.warning(
            f"Overall final_score is GREEN ({final_score:.2f}), but timing_zone is "
            f"{timing_zone}. The pipeline is broadly conformant, but the latest run "
            "is approaching its timing SLA boundary."
        )


def numeric_value(row: pd.Series, column: str):
    """
    Read a numeric value from a conformance row.

    CSV values can arrive as strings, empty cells, or real numbers. This helper
    normalizes them before they are used in metric cards and warnings.
    """
    if column not in row.index:
        return None

    value = pd.to_numeric(row.get(column), errors="coerce")
    if pd.isna(value):
        return None

    return float(value)


def get_latest_pipeline_row(
    conformance_df: pd.DataFrame,
    pipeline_id: str,
) -> Optional[pd.Series]:
    """
    Return the latest measured row for one pipeline.

    Several dashboard pages need the same "current state" row. Keeping the
    lookup in one helper prevents Fleet, Pipeline Detail, and Visualizations
    from accidentally using different sorting rules.
    """
    history = get_pipeline_history(conformance_df, pipeline_id)

    if history.empty:
        return None

    return history.iloc[-1]


def build_pipeline_takeaway(latest: Optional[pd.Series]) -> tuple[str, str]:
    """
    Convert the latest conformance row into one demo-friendly takeaway.

    The dashboard already shows many metrics. This helper gives non-technical
    readers a single sentence that explains what they should notice first.
    """
    if latest is None:
        return "info", "No conformance result is available for this pipeline yet."

    final_score = numeric_value(latest, "final_score")
    gap = numeric_value(latest, "bilateral_gap_minutes")
    timing_zone = str(latest.get("timing_zone", "") or "")
    confidence = str(latest.get("confidence_level", "") or "")

    if confidence in {"LOW", "UNRELIABLE"}:
        return (
            "warning",
            "Measurement confidence is low. Fix or validate event extraction before acting on the score.",
        )

    if gap is not None and gap > 20:
        return (
            "error",
            f"The main issue is handoff delay: the consumer waits {gap:.1f} minutes after producer availability.",
        )

    if timing_zone in {"RED", "BREACH"}:
        return (
            "error",
            f"The latest run is in timing zone {timing_zone}. Investigate runtime before the next SLA window.",
        )

    if timing_zone == "AMBER":
        return (
            "warning",
            "The pipeline is close to its timing boundary. Monitor runtime and handoff before it worsens.",
        )

    if final_score is not None and final_score >= 0.85:
        return (
            "success",
            "The latest measured run is broadly conformant. Continue monitoring for drift and hidden handoff gaps.",
        )

    return (
        "info",
        "The latest result needs review. Use the score breakdown and visual tabs to identify the driver.",
    )


def render_takeaway(kind: str, message: str):
    """
    Render one short interpretation card.

    Streamlit status components make the dashboard easier to scan during a
    live demo: red means act now, yellow means monitor, blue means context,
    and green means no immediate issue.
    """
    if kind == "error":
        st.error(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "success":
        st.success(message)
    else:
        st.info(message)


def render_gap_callout(latest: pd.Series):
    """
    Highlight bilateral handoff risk when the producer-consumer gap is large.

    The 20-minute threshold matches the high-gap count used on Fleet Overview.
    """
    gap = numeric_value(latest, "bilateral_gap_minutes")

    if gap is None:
        st.info("Bilateral gap is unavailable for this run.")
    elif gap > 20:
        st.error(
            f"Bilateral gap is high: {gap:.1f} minutes. "
            "The consumer receives data much later than the producer marks it available."
        )
    elif gap > 10:
        st.warning(
            f"Bilateral gap is elevated: {gap:.1f} minutes. "
            "Monitor this handoff before it becomes a downstream delay."
        )
    else:
        st.success(f"Bilateral gap is low: {gap:.1f} minutes.")


def render_pipeline_snapshot(
    selected_pipeline: str,
    conformance_df: pd.DataFrame,
    available_input_dates: Optional[list[str]] = None,
):
    """
    Show the selected pipeline's current story before detailed visuals.

    This section is intentionally compact: it tells a reviewer what the latest
    score, zone, confidence, and handoff gap are before they inspect charts.
    """
    latest = get_latest_pipeline_row(conformance_df, selected_pipeline)
    available_input_dates = available_input_dates or []
    history = get_pipeline_history(conformance_df, selected_pipeline)

    st.subheader(selected_pipeline)

    kind, message = build_pipeline_takeaway(latest)
    render_takeaway(kind, message)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("History rows", len(history))
    col2.metric("Input dates", len(available_input_dates))

    if latest is None:
        col3.metric("Latest score", "n/a")
        col4.metric("Timing zone", "n/a")
        col5.metric("Gap", "n/a")
        return

    col3.metric("Latest score", format_score(latest.get("final_score")))
    col4.metric("Timing zone", latest.get("timing_zone", "n/a"))

    gap = numeric_value(latest, "bilateral_gap_minutes")
    col5.metric("Gap", "n/a" if gap is None else f"{gap:.1f} min")

    if len(history) <= 1:
        st.info(
            "Only one conformance row is available. Drift and prediction views "
            "can show the latest state, but they cannot prove a trend yet."
        )
    elif len(history) < 5:
        st.warning(
            f"{len(history)} conformance rows are available. Multi-day charts are visible, "
            "but prediction confidence is limited until at least 5 points exist."
        )


def render_variant_explainer(latest: pd.Series):
    """
    Show the variant explanation as readable text instead of a cramped table row.

    This field can be long because it explains skipped, repeated, or missing
    process variants. A separate info block keeps the score table compact.
    """
    explainer = str(latest.get("variant_explainer", "")).strip()

    if explainer and explainer.lower() != "nan":
        st.info(explainer)
    else:
        st.caption("No variant explanation was recorded for this run.")


def get_pipeline_history(conformance_df: pd.DataFrame, pipeline_id: str) -> pd.DataFrame:
    """
    Return all conformance rows for one pipeline.

    Trend visuals such as the Drift Chart should use every historical row for
    the selected pipeline, not only the input date selected for event-log charts.
    """
    if conformance_df.empty or "pipeline_id" not in conformance_df.columns:
        return pd.DataFrame()

    history = conformance_df[conformance_df["pipeline_id"] == pipeline_id].copy()

    if "run_date" in history.columns:
        # Sort makes metric summaries match the same chronological order as charts.
        history = history.sort_values("run_date")

    return history


def get_history_counts(conformance_df: pd.DataFrame) -> dict[str, int]:
    """
    Count how many measured days each pipeline has.

    The dashboard uses this to prioritize pipelines with useful trend history
    instead of defaulting to a one-day pipeline that cannot show drift.
    """
    if conformance_df.empty or "pipeline_id" not in conformance_df.columns:
        return {}

    counts = conformance_df.groupby("pipeline_id").size()
    return {str(pipeline_id): int(count) for pipeline_id, count in counts.items()}


def get_available_input_dates(pipeline_id: str) -> list[str]:
    """
    Find event-log dates available under inputs/{pipeline_id}/.

    Gap, DFG, and Performance visuals need producer/consumer event files such as
    producer_20260618.parquet. This helper lets the dashboard default to a date
    that actually exists instead of today's date when today's files are absent.
    """
    pipeline_input_dir = INPUTS_ROOT / pipeline_id

    if not pipeline_input_dir.exists():
        return []

    dates = set()
    for path in pipeline_input_dir.iterdir():
        # Only producer files are required as the anchor for an input date.
        # Consumer files are checked later by load_pipeline_events().
        match = re.match(r"producer_(\d{8})\.(csv|parquet)$", path.name)
        if match:
            dates.add(match.group(1))

    return sorted(dates)


def load_pipeline_event_scope(
    pipeline_id: str,
    date_str: str,
    available_dates: list[str],
    use_all_event_dates: bool,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
    """
    Load event logs for either one date or the full available date range.

    Drift charts already use all rows from conformance_log.csv. Event-log charts
    such as Bilateral Gap, DFG, and Performance need the same behavior when the
    user wants to compare many days together.
    """
    if use_all_event_dates and available_dates:
        selected_dates = available_dates
    else:
        selected_dates = [date_str]

    producer_frames = []
    consumer_frames = []
    skipped = []

    for input_date in selected_dates:
        producer_df, consumer_df, status = load_pipeline_events(
            inputs_dir=str(INPUTS_ROOT),
            pipeline_id=pipeline_id,
            date_str=input_date,
        )

        if producer_df.empty:
            # Keep loading other dates; one missing day should not block a
            # multi-day chart if enough other days are available.
            skipped.append(f"{input_date}: {status}")
            continue

        producer_frames.append(producer_df)

        if consumer_df is None:
            skipped.append(f"{input_date}: consumer log unavailable")
        else:
            consumer_frames.append(consumer_df)

    if not producer_frames:
        return pd.DataFrame(), None, "missing_input: no producer events in selected date scope"

    producer_all = pd.concat(producer_frames, ignore_index=True)
    consumer_all = (
        pd.concat(consumer_frames, ignore_index=True)
        if consumer_frames
        else None
    )

    scope_label = (
        f"all available input dates ({len(selected_dates)} dates)"
        if use_all_event_dates and available_dates
        else f"input date {date_str}"
    )

    if skipped:
        return producer_all, consumer_all, f"partial {scope_label}: " + "; ".join(skipped)

    return producer_all, consumer_all, f"ok: {scope_label}"


def render_contract_summary_cards(
    selected_pipeline: str,
    contracts_df: pd.DataFrame,
):
    """
    Render the most useful contract metadata as compact cards.

    The raw contract table is still available in an expander, but the main page
    should immediately show ownership, SLA window, criticality, and grain.
    """
    if contracts_df.empty or "pipeline_id" not in contracts_df.columns:
        st.info("Contracts directory not available or no valid contracts found.")
        return

    contract_row = contracts_df[contracts_df["pipeline_id"] == selected_pipeline]

    if contract_row.empty:
        st.info("No contract metadata loaded for this pipeline.")
        return

    contract = contract_row.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Owner", contract.get("owner", "n/a"))
    col2.metric("Criticality", contract.get("criticality", "n/a"))
    col3.metric("Status", contract.get("status", "n/a"))
    col4.metric("Grain", contract.get("grain", "n/a"))

    col5, col6, col7 = st.columns(3)
    col5.metric("Expected start", contract.get("expected_start", "n/a"))
    col6.metric("Expected end", contract.get("expected_end", "n/a"))
    col7.metric("Consumer team", contract.get("consumer_team", "n/a"))

    required_events = contract.get("required_events", "")
    if required_events:
        st.caption(f"Expected process: {required_events}")

    with st.expander("Raw contract metadata", expanded=False):
        st.dataframe(
            contract_row,
            width="stretch",
            hide_index=True,
        )


def sorted_filter_options(df: pd.DataFrame, column: str) -> list[str]:
    """
    Return stable sidebar filter values for a categorical column.

    Missing columns are allowed because older conformance logs may not contain
    every dashboard field yet.
    """
    if column not in df.columns:
        return []

    values = df[column].dropna().astype(str)
    values = values[values.str.strip() != ""]

    return sorted(values.unique())


def apply_fleet_filters(latest: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Fleet Overview filters selected in the sidebar.

    Filters operate on the latest row per pipeline because Fleet Overview is a
    current-health view, not a full historical analysis page.
    """
    filtered = latest.copy()

    timing_options = sorted_filter_options(latest, "timing_zone")
    pattern_options = sorted_filter_options(latest, "pattern")
    confidence_options = sorted_filter_options(latest, "confidence_level")

    selected_timing = st.sidebar.multiselect(
        "Timing zone",
        timing_options,
        default=timing_options,
    )

    selected_pattern = st.sidebar.multiselect(
        "Pattern",
        pattern_options,
        default=pattern_options,
    )

    selected_confidence = st.sidebar.multiselect(
        "Confidence",
        confidence_options,
        default=confidence_options,
    )

    high_gap_only = st.sidebar.checkbox(
        "High-gap only",
        value=False,
        help="Show only pipelines where bilateral_gap_minutes is above 20.",
    )

    if selected_timing and "timing_zone" in filtered.columns:
        filtered = filtered[filtered["timing_zone"].astype(str).isin(selected_timing)]

    if selected_pattern and "pattern" in filtered.columns:
        filtered = filtered[filtered["pattern"].astype(str).isin(selected_pattern)]

    if selected_confidence and "confidence_level" in filtered.columns:
        filtered = filtered[
            filtered["confidence_level"].astype(str).isin(selected_confidence)
        ]

    if high_gap_only and "bilateral_gap_minutes" in filtered.columns:
        gaps = pd.to_numeric(filtered["bilateral_gap_minutes"], errors="coerce")
        filtered = filtered[gaps > 20]

    return filtered


def render_fleet_overview(conformance_df: pd.DataFrame):
    """
    Render the fleet-level overview page.

    This page summarizes all pipelines from conformance_log.csv.
    """
    st.title("Fleet Overview")
    render_metric_guide()

    if conformance_df.empty:
        render_empty_state("No conformance history found yet.")
        return

    df = conformance_df.copy()

    # Scores are read from CSV, so convert them from text to numeric safely.
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")

    # Keep the latest row per pipeline for current fleet status.
    latest = (
        df.sort_values("run_date")
        .groupby("pipeline_id", as_index=False)
        .tail(1)
    )

    filtered_latest = apply_fleet_filters(latest)

    total_pipelines = filtered_latest["pipeline_id"].nunique()
    total_available = latest["pipeline_id"].nunique()
    average_score = filtered_latest["final_score"].mean()
    high_gap_count = 0

    if "bilateral_gap_minutes" in filtered_latest.columns:
        gaps = pd.to_numeric(
            filtered_latest["bilateral_gap_minutes"],
            errors="coerce",
        )
        high_gap_count = int((gaps > 20).sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Pipelines", f"{total_pipelines}/{total_available}")
    col2.metric("Average score", format_score(average_score))
    col3.metric("High-gap pipelines", high_gap_count)

    st.subheader("Latest Pipeline Health")

    visible_columns = [
        "pipeline_id",
        "run_date",
        "final_score",
        "timing_zone",
        "confidence_level",
        "pattern",
        "bilateral_gap_minutes",
    ]

    visible_columns = [col for col in visible_columns if col in latest.columns]

    if filtered_latest.empty:
        st.info("No pipelines match the selected filters.")
    else:
        st.dataframe(
            filtered_latest[visible_columns].sort_values("pipeline_id"),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "Note: timing_zone is the latest timing classification only. "
        "final_score is the weighted overall conformance score."
    )


def render_pipeline_detail(conformance_df: pd.DataFrame, contracts_df: pd.DataFrame):
    """
    Render detail view for one selected pipeline.

    This page explains the latest conformance result and contract metadata.
    """
    st.title("Pipeline Detail")
    render_metric_guide()

    if conformance_df.empty:
        render_empty_state("No conformance history found yet.")
        return

    df = conformance_df.copy()
    pipelines = sorted(df["pipeline_id"].dropna().unique())

    if not pipelines:
        render_empty_state("No pipeline_id values found in conformance_log.csv.")
        return

    selected_pipeline = st.sidebar.selectbox(
        "Pipeline",
        pipelines,
    )

    history = get_pipeline_history(df, selected_pipeline)
    latest = get_latest_pipeline_row(df, selected_pipeline)

    if history.empty or latest is None:
        st.warning("No history for this pipeline.")
        return

    st.subheader(selected_pipeline)
    kind, message = build_pipeline_takeaway(latest)
    render_takeaway(kind, message)
    explain_timing_score_mismatch(latest)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Final score", format_score(latest.get("final_score")))
    col2.metric("Timing zone", latest.get("timing_zone", "n/a"))
    col3.metric("Confidence", latest.get("confidence_level", "n/a"))
    col4.metric("Pattern", latest.get("pattern", "n/a"))

    render_gap_callout(latest)

    st.subheader("Score Breakdown")

    score_col1, score_col2, score_col3 = st.columns(3)
    score_col1.metric(
        "Sequence fitness",
        format_score(latest.get("sequence_fitness")),
    )
    score_col2.metric(
        "Timing score",
        format_score(latest.get("timing_score")),
    )
    score_col3.metric(
        "Completeness score",
        format_score(latest.get("completeness_score")),
    )

    st.caption(
        "Score formula: final_score = sequence_fitness * 0.35 "
        "+ timing_score * 0.50 + completeness_score * 0.15"
    )

    st.subheader("Variant Explanation")
    render_variant_explainer(latest)

    st.subheader("Contract Summary")
    render_contract_summary_cards(selected_pipeline, contracts_df)


def render_visualization_image(path: Path, title: str, missing_hint: str):
    """
    Render one exported PNG if it exists.

    Phase 4 reuses static artifacts from Phase 3 instead of duplicating chart
    logic in Streamlit. If a file is missing, the dashboard shows the command
    needed to create it.
    """
    st.subheader(title)

    if path.exists():
        # Streamlit can render local PNG files directly from the project folder.
        st.image(str(path), width="stretch")
        st.caption(str(path))
        return

    st.info(missing_hint)


def render_chart_note(title: str, body: str):
    """
    Add a short explanation below a visual.

    Static PNGs are useful in reports, but dashboard viewers still need a plain
    sentence that says what the chart proves and how to read it.
    """
    st.caption(f"{title}: {body}")


def load_json_artifact(path: Path) -> tuple[dict, str]:
    """
    Load a structured process-mining JSON artifact safely.

    Prediction and recommendation commands write JSON beside PNG artifacts.
    The dashboard should read those files directly, but missing/invalid JSON
    should become a visible dashboard status instead of a Streamlit crash.
    """
    if not path.exists():
        return {}, f"missing artifact: {path}"

    try:
        # Keep encoding explicit so the dashboard behaves the same on macOS,
        # Linux, and CI machines.
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {}, f"invalid artifact: {e}"

    if not isinstance(payload, dict):
        # All Fracture process-mining artifacts are JSON objects at the top
        # level. Lists/scalars usually mean the wrong file was selected.
        return {}, "invalid artifact: expected a JSON object"

    return payload, "ok"


def format_artifact_value(value, ndigits: int = 2) -> str:
    """
    Format JSON metric values for compact dashboard cards and tables.

    JSON artifacts can contain None/NaN when prediction is unavailable, so this
    helper mirrors format_score but works for generic metrics too.
    """
    if value is None:
        return "n/a"

    try:
        if pd.isna(value):
            return "n/a"
    except Exception:
        pass

    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def highest_recommendation_severity(recommendations: list[dict]) -> str:
    """
    Return the strongest action severity in a recommendation artifact.

    This gives the dashboard a single headline severity while still showing all
    detailed recommendation rows below.
    """
    severity_rank = {
        "INFO": 0,
        "WATCH": 1,
        "ACTION": 2,
        "URGENT": 3,
        "BLOCKED": 4,
    }

    if not recommendations:
        return "n/a"

    return max(
        (str(item.get("severity", "INFO")).upper() for item in recommendations),
        key=lambda severity: severity_rank.get(severity, -1),
    )


def render_artifact_command(command: str):
    """
    Show the CLI command that creates a missing dashboard artifact.

    The dashboard is allowed to be opened before every artifact exists, so a
    short command keeps the recovery path obvious during demos.
    """
    st.code(command, language="bash")


def save_json_artifact(payload: dict, pipeline_id: str, filename: str) -> Path:
    """
    Persist a dashboard-generated JSON artifact beside PNG visualizations.

    The CLI also writes these files. This small dashboard writer exists so the
    Streamlit page can regenerate missing prediction/action artifacts without
    importing CLI command handlers.
    """
    pipeline_dir = VISUALIZATION_ROOT / pipeline_id
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    # Copy before adding metadata so caller-owned dictionaries are not mutated.
    output_payload = dict(payload)
    output_payload["generated_at"] = datetime.now().isoformat()

    output_path = pipeline_dir / filename
    output_path.write_text(
        json.dumps(output_payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    return output_path


def render_prediction_artifact(payload: dict, status: str, pipeline_id: str):
    """
    Render predictive process-mining output from prediction.json.

    Prediction answers whether the pipeline is drifting toward a future score
    breach or widening producer-consumer gap.
    """
    st.subheader("Prediction")

    if not payload:
        st.info("No prediction artifact exported yet.")
        st.caption(status)
        render_artifact_command(
            f"python -m fracture.cli predict --pipeline-id {pipeline_id}"
        )
        return

    artifact_status = str(payload.get("status", "unknown"))
    metrics = payload.get("metrics", {}) or {}
    predictions = payload.get("predictions", []) or []
    findings = payload.get("findings", []) or []

    if artifact_status == "ok":
        st.success("Prediction artifact loaded.")
    else:
        # Non-ok artifacts are still useful: they explain why prediction was
        # skipped, for example low confidence or insufficient history.
        st.warning(f"Prediction status: {artifact_status}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", artifact_status)
    col2.metric(
        "Score points",
        f"{metrics.get('available_score_points', 0)}/"
        f"{metrics.get('required_score_points', 'n/a')}",
    )
    col3.metric(
        "Gap points",
        f"{metrics.get('available_gap_points', 0)}/"
        f"{metrics.get('required_gap_points', 'n/a')}",
    )
    col4.metric(
        "Latest gap",
        f"{format_artifact_value(metrics.get('latest_gap_minutes'))} min",
    )

    trend_col1, trend_col2, trend_col3 = st.columns(3)
    trend_col1.metric(
        "Score slope/day",
        format_artifact_value(metrics.get("score_slope_per_day"), ndigits=4),
    )
    trend_col2.metric(
        "Gap slope/day",
        format_artifact_value(metrics.get("gap_slope_minutes_per_day"), ndigits=4),
    )
    trend_col3.metric(
        "Latest confidence",
        metrics.get("latest_confidence_level", "n/a"),
    )

    if predictions:
        # Keep the table compact: it should show decision-relevant fields, not
        # every internal detail from the JSON payload.
        prediction_rows = []
        for item in predictions:
            prediction_rows.append({
                "type": item.get("prediction_type", ""),
                "projected_date": item.get("projected_date") or "n/a",
                "days": format_artifact_value(
                    item.get("days_until_projection"),
                    ndigits=0,
                ),
                "confidence": item.get("confidence", ""),
                "urgency": item.get("urgency", ""),
                "threshold": format_artifact_value(item.get("threshold")),
                "slope": format_artifact_value(item.get("slope_per_day"), ndigits=4),
                "explanation": item.get("explanation", ""),
            })

        st.table(pd.DataFrame(prediction_rows))
    else:
        st.info("No active projection was returned for this pipeline.")

    if findings:
        st.markdown("**Findings**")
        for finding in findings:
            st.write(f"- {finding}")


def render_recommendation_artifact(payload: dict, status: str, pipeline_id: str):
    """
    Render action-oriented process-mining output from recommendations.json.

    Recommendations translate diagnostics into owner, severity, likely cause,
    concrete action, and the next CLI command to run.
    """
    st.subheader("Actions")

    if not payload:
        st.info("No recommendation artifact exported yet.")
        st.caption(status)
        render_artifact_command(
            f"python -m fracture.cli recommend --pipeline-id {pipeline_id}"
        )
        return

    artifact_status = str(payload.get("status", "unknown"))
    metrics = payload.get("metrics", {}) or {}
    recommendations = payload.get("recommendations", []) or []
    highest_severity = highest_recommendation_severity(recommendations)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", artifact_status)
    col2.metric("Recommendations", len(recommendations))
    col3.metric("Highest severity", highest_severity)
    col4.metric("Latest zone", metrics.get("timing_zone", "n/a"))

    if not recommendations:
        st.success("No action recommendation is currently needed.")
        return

    for item in recommendations:
        severity = str(item.get("severity", "INFO")).upper()
        message = f"[{severity}] {item.get('finding', '')}"

        # Severity uses visible Streamlit states so the most urgent action is
        # easy to spot before reading the detailed table.
        if severity in {"URGENT", "BLOCKED"}:
            st.error(message)
        elif severity == "ACTION":
            st.warning(message)
        elif severity == "WATCH":
            st.info(message)
        else:
            st.success(message)

    recommendation_rows = []
    for item in recommendations:
        recommendation_rows.append({
            "severity": item.get("severity", ""),
            "owner": item.get("owner", ""),
            "finding": item.get("finding", ""),
            "probable_cause": item.get("probable_cause", ""),
            "recommended_action": item.get("recommended_action", ""),
            "next_command": item.get("next_command", ""),
            "confidence": item.get("confidence", ""),
        })

    st.table(pd.DataFrame(recommendation_rows))


def load_pipeline_contract(pipeline_id: str):
    """
    Load one contract object for visualization generation.

    The dashboard needs the real PipelineContract object for Petri net export
    and for pipeline-specific producer/consumer handoff event names.
    """
    contract_path = CONTRACTS_ROOT / f"{pipeline_id}.yaml"

    if not contract_path.exists():
        # Missing contracts should not crash the dashboard; only Petri export is skipped.
        return None, f"missing contract: {contract_path}"

    try:
        return load_contract(str(contract_path)), "ok"
    except Exception as e:
        # Invalid contracts are surfaced as a status message instead of a hard crash.
        return None, f"invalid contract: {e}"


def build_event_visual_scope(
    pipeline_id: str,
    date_str: str,
    available_dates: Optional[list[str]],
    use_all_event_dates: bool,
) -> tuple:
    """
    Build a stable key for the event-log visual input scope.

    Gap, DFG, and performance charts are written to fixed file names. Without a
    scope key, changing the sidebar date would leave the old PNG in place
    because the dashboard only checks whether the output file exists.
    """
    available_dates = available_dates or []

    if use_all_event_dates:
        # In all-dates mode, any change in discovered input files should trigger
        # a redraw because the chart represents the full event history.
        date_scope = tuple(available_dates)
        mode = "all"
    else:
        # In single-date mode, pressing Enter in the Input date field should
        # redraw event-log charts for the selected date.
        date_scope = (str(date_str).strip(),)
        mode = "single"

    return (pipeline_id, mode, date_scope)


def describe_event_scope_availability(
    date_str: str,
    available_dates: Optional[list[str]],
    use_all_event_dates: bool,
) -> tuple[bool, str]:
    """
    Tell whether event-log visuals can be generated for the selected scope.

    Bilateral Gap, DFG, and Performance charts read producer/consumer input
    files. If a user types a date that does not exist, the dashboard must show
    "unavailable" instead of reusing a stale PNG from a previous valid date.
    """
    available_dates = available_dates or []
    cleaned_date = str(date_str).strip()

    if use_all_event_dates:
        if available_dates:
            return True, "Event-log visuals are using the full available input range."

        return False, "Unavailable: no input dates exist for this pipeline."

    if not available_dates:
        return False, "Unavailable: no producer input files exist for this pipeline."

    if cleaned_date not in available_dates:
        return (
            False,
            f"Unavailable: input date {cleaned_date} has no event log. "
            f"Available range is {available_dates[0]} to {available_dates[-1]}.",
        )

    return True, f"Event-log visuals are using input date {cleaned_date}."


def ensure_visualizations_for_pipeline(
    pipeline_id: str,
    conformance_df: pd.DataFrame,
    date_str: str,
    available_dates: Optional[list[str]] = None,
    use_all_event_dates: bool = False,
    event_scope_available: bool = True,
    event_scope_status: str = "",
    event_scope_changed: bool = False,
    force: bool = False,
) -> list[str]:
    """
    Generate missing static visualization files for the selected pipeline.

    This turns the dashboard from a passive image viewer into a thin UI over
    the existing Phase 3 export functions. It still does not duplicate chart
    logic; it only calls the reusable visualization helpers.
    """
    statuses = []
    pipeline_dir = VISUALIZATION_ROOT / pipeline_id
    available_dates = available_dates or []
    force_event_visuals = event_scope_available and (force or event_scope_changed)

    contract, contract_status = load_pipeline_contract(pipeline_id)

    # Contract controls custom handoff event names. Defaults match Fracture vocabulary.
    producer_event = "DATA_AVAILABLE"
    consumer_event = "DATA_AVAILABLE"
    if contract is not None:
        producer_event = contract.log_contract.upstream_producer_event
        consumer_event = contract.log_contract.upstream_consumer_event

    gap_path = pipeline_dir / "bilateral_gap_timeline.png"
    gap_analysis_path = pipeline_dir / "bilateral_gap_analysis.png"
    if not event_scope_available:
        statuses.append(f"event-log visuals unavailable: {event_scope_status}")
    elif force_event_visuals or not gap_path.exists() or not gap_analysis_path.exists():
        producer_df, consumer_df, input_status = load_pipeline_event_scope(
            pipeline_id=pipeline_id,
            date_str=date_str,
            available_dates=available_dates,
            use_all_event_dates=use_all_event_dates,
        )

        if producer_df.empty:
            statuses.append(f"gap skipped: {input_status}")
        elif consumer_df is None:
            statuses.append("gap skipped: consumer log unavailable")
        else:
            path, status = save_bilateral_gap_timeline(
                producer_df=producer_df,
                consumer_df=consumer_df,
                pipeline_id=pipeline_id,
                output_dir=str(VISUALIZATION_ROOT),
                producer_event=producer_event,
                consumer_event=consumer_event,
            )
            statuses.append(f"gap: {status}" if path is None else f"gap created: {path}")

            path, status = save_bilateral_gap_analysis(
                producer_df=producer_df,
                consumer_df=consumer_df,
                pipeline_id=pipeline_id,
                output_dir=str(VISUALIZATION_ROOT),
                producer_event=producer_event,
                consumer_event=consumer_event,
            )
            statuses.append(
                f"gap analysis: {status}"
                if path is None
                else f"gap analysis created: {path}"
            )

    drift_path = pipeline_dir / "drift_chart.png"
    if force or not drift_path.exists():
        if conformance_df.empty:
            statuses.append("drift skipped: conformance_log.csv unavailable")
        else:
            path, status = save_drift_chart(
                conformance_df=conformance_df,
                pipeline_id=pipeline_id,
                output_dir=str(VISUALIZATION_ROOT),
            )
            statuses.append(f"drift: {status}" if path is None else f"drift created: {path}")

    petri_path = pipeline_dir / "contract_petri_net.png"
    if force or not petri_path.exists():
        if contract is None:
            statuses.append(f"petri skipped: {contract_status}")
        else:
            path, status = save_contract_petri_net(
                contract=contract,
                output_dir=str(VISUALIZATION_ROOT),
            )
            statuses.append(f"petri: {status}" if path is None else f"petri created: {path}")

    dfg_path = pipeline_dir / "discovered_dfg_producer.png"
    if event_scope_available and (force_event_visuals or not dfg_path.exists()):
        if contract is None:
            statuses.append(f"dfg skipped: {contract_status}")
        else:
            producer_df, consumer_df, input_status = load_pipeline_event_scope(
                pipeline_id=pipeline_id,
                date_str=date_str,
                available_dates=available_dates,
                use_all_event_dates=use_all_event_dates,
            )

            if producer_df.empty:
                statuses.append(f"dfg producer skipped: {input_status}")
            else:
                path, status = save_discovered_dfg(
                    events_df=producer_df,
                    contract=contract,
                    log_side="producer",
                    output_dir=str(VISUALIZATION_ROOT),
                )
                statuses.append(
                    f"dfg producer: {status}" if path is None else f"dfg producer created: {path}"
                )

            if consumer_df is None:
                statuses.append("dfg consumer skipped: consumer log unavailable")
            else:
                path, status = save_discovered_dfg(
                    events_df=consumer_df,
                    contract=contract,
                    log_side="consumer",
                    output_dir=str(VISUALIZATION_ROOT),
                )
                statuses.append(
                    f"dfg consumer: {status}" if path is None else f"dfg consumer created: {path}"
                )

    performance_path = pipeline_dir / "performance_dfg_producer.png"
    execution_drift_path = pipeline_dir / "execution_time_drift_producer.png"
    if event_scope_available and (
        force_event_visuals
        or not performance_path.exists()
        or not execution_drift_path.exists()
    ):
        if contract is None:
            statuses.append(f"performance skipped: {contract_status}")
        else:
            producer_df, consumer_df, input_status = load_pipeline_event_scope(
                pipeline_id=pipeline_id,
                date_str=date_str,
                available_dates=available_dates,
                use_all_event_dates=use_all_event_dates,
            )

            if producer_df.empty:
                statuses.append(f"performance producer skipped: {input_status}")
            else:
                path, status = save_performance_dfg(
                    events_df=producer_df,
                    contract=contract,
                    log_side="producer",
                    output_dir=str(VISUALIZATION_ROOT),
                )
                statuses.append(
                    f"performance producer: {status}"
                    if path is None
                    else f"performance producer created: {path}"
                )

                path, status = save_execution_time_drift(
                    events_df=producer_df,
                    contract=contract,
                    log_side="producer",
                    output_dir=str(VISUALIZATION_ROOT),
                )
                statuses.append(
                    f"execution drift producer: {status}"
                    if path is None
                    else f"execution drift producer created: {path}"
                )

            if consumer_df is None:
                statuses.append("performance consumer skipped: consumer log unavailable")
            else:
                path, status = save_performance_dfg(
                    events_df=consumer_df,
                    contract=contract,
                    log_side="consumer",
                    output_dir=str(VISUALIZATION_ROOT),
                )
                statuses.append(
                    f"performance consumer: {status}"
                    if path is None
                    else f"performance consumer created: {path}"
                )

                path, status = save_execution_time_drift(
                    events_df=consumer_df,
                    contract=contract,
                    log_side="consumer",
                    output_dir=str(VISUALIZATION_ROOT),
                )
                statuses.append(
                    f"execution drift consumer: {status}"
                    if path is None
                    else f"execution drift consumer created: {path}"
                )

    prediction_path = pipeline_dir / "prediction.json"
    recommendation_path = pipeline_dir / "recommendations.json"
    if force or not prediction_path.exists() or not recommendation_path.exists():
        if conformance_df.empty:
            statuses.append("prediction/actions skipped: conformance_log.csv unavailable")
        else:
            try:
                from fracture.prediction import predict_pipeline
                from fracture.recommendation import recommend_pipeline

                # Prediction uses the scored conformance history, not raw event
                # logs. Even a skipped prediction is saved so the dashboard can
                # explain "insufficient history" or "low confidence".
                prediction_result = predict_pipeline(
                    conformance_df=conformance_df,
                    pipeline_id=pipeline_id,
                )
                prediction_output = save_json_artifact(
                    prediction_result.as_dict(),
                    pipeline_id=pipeline_id,
                    filename="prediction.json",
                )
                statuses.append(f"prediction created: {prediction_output}")

                # Prefer contract owner for the action route. If the contract
                # is missing, the recommendation module falls back to log owner
                # columns or a generic pipeline owner.
                recommendation_owner = contract.owner if contract is not None else None
                recommendation_result = recommend_pipeline(
                    conformance_df=conformance_df,
                    pipeline_id=pipeline_id,
                    prediction_result=prediction_result,
                    owner=recommendation_owner,
                )
                recommendation_output = save_json_artifact(
                    recommendation_result.as_dict(),
                    pipeline_id=pipeline_id,
                    filename="recommendations.json",
                )
                statuses.append(f"recommendations created: {recommendation_output}")
            except Exception as e:
                # The rest of the dashboard should remain usable even if the
                # analytical JSON artifacts cannot be produced.
                statuses.append(f"prediction/actions skipped: {e}")

    heatmap_path = VISUALIZATION_ROOT / "fleet_heatmap.png"
    if force or not heatmap_path.exists():
        if conformance_df.empty:
            statuses.append("heatmap skipped: conformance_log.csv unavailable")
        else:
            path, status = save_fleet_heatmap(
                conformance_df=conformance_df,
                output_dir=str(VISUALIZATION_ROOT),
            )
            statuses.append(f"heatmap: {status}" if path is None else f"heatmap created: {path}")

    return statuses


def get_pipeline_options(conformance_df: pd.DataFrame) -> list[str]:
    """
    Build a pipeline selector from conformance history or visualization folders.

    conformance_log.csv is the preferred source because it reflects measured
    pipelines. Folder fallback lets the page still work after static export.
    """
    if not conformance_df.empty and "pipeline_id" in conformance_df.columns:
        counts = get_history_counts(conformance_df)

        # Put pipelines with more historical rows first so trend charts are useful
        # immediately when the dashboard opens.
        pipelines = sorted(
            conformance_df["pipeline_id"].dropna().unique(),
            key=lambda pipeline_id: (-counts.get(str(pipeline_id), 0), str(pipeline_id)),
        )
        if pipelines:
            return pipelines

    if VISUALIZATION_ROOT.exists():
        # Per-pipeline visualizations live under outputs/visualizations/{pipeline_id}/.
        return sorted(
            path.name
            for path in VISUALIZATION_ROOT.iterdir()
            if path.is_dir()
        )

    return []


def render_visualizations(conformance_df: pd.DataFrame):
    """
    Render Phase 3 static visualizations inside the dashboard.

    This page is intentionally simple: it proves that CLI/static exports can
    become dashboard assets without changing the process-mining engine.
    """
    st.title("Visualizations")
    render_metric_guide()

    pipelines = get_pipeline_options(conformance_df)

    if not pipelines:
        render_empty_state("No visualization folders or conformance history found yet.")
        return

    selected_pipeline = st.sidebar.selectbox(
        "Visualization pipeline",
        pipelines,
    )

    available_dates = get_available_input_dates(selected_pipeline)
    default_input_date = (
        available_dates[-1]
        if available_dates
        else date.today().strftime("%Y%m%d")
    )

    date_str = st.sidebar.text_input(
        "Input date",
        value=default_input_date,
        help="Used for producer/consumer input files such as producer_YYYYMMDD.parquet.",
        key=f"visualization_input_date_{selected_pipeline}",
    )

    if available_dates:
        # Keep the date context visible because some visuals use event files,
        # while drift uses the full conformance history.
        st.sidebar.caption(
            f"Available input range: {available_dates[0]} to {available_dates[-1]}"
        )
    else:
        st.sidebar.caption("No input files found for this pipeline yet.")

    has_multiple_input_dates = len(available_dates) > 1

    use_all_event_dates = st.sidebar.checkbox(
        "Use all input dates",
        value=has_multiple_input_dates,
        help=(
            "When enabled, Bilateral Gap, DFG, and Performance visuals use every "
            "available producer_YYYYMMDD file for the selected pipeline. When disabled, "
            "they use only the Input date above."
        ),
        # If there is only one input date, toggling this cannot change the chart.
        # Disable it so the UI does not imply a second date range exists.
        disabled=not has_multiple_input_dates,
    )

    event_scope_available, event_scope_status = describe_event_scope_availability(
        date_str=date_str,
        available_dates=available_dates,
        use_all_event_dates=use_all_event_dates,
    )

    if event_scope_available:
        st.sidebar.caption(event_scope_status)
    else:
        # A stale PNG may exist from a previous valid date. Make the invalid
        # date state explicit so users do not mistake old output for this date.
        st.sidebar.error(event_scope_status)

    event_visual_scope = build_event_visual_scope(
        pipeline_id=selected_pipeline,
        date_str=date_str,
        available_dates=available_dates,
        use_all_event_dates=use_all_event_dates,
    )
    event_visual_scope_key = f"event_visual_scope::{selected_pipeline}"
    event_scope_changed = (
        st.session_state.get(event_visual_scope_key) != event_visual_scope
    )
    st.session_state[event_visual_scope_key] = event_visual_scope

    force_regenerate = st.sidebar.button("Regenerate visuals")

    with st.spinner("Preparing visualizations..."):
        generation_statuses = ensure_visualizations_for_pipeline(
            pipeline_id=selected_pipeline,
            conformance_df=conformance_df,
            date_str=date_str,
            available_dates=available_dates,
            use_all_event_dates=use_all_event_dates,
            event_scope_available=event_scope_available,
            event_scope_status=event_scope_status,
            event_scope_changed=event_scope_changed,
            force=force_regenerate,
        )

    pipeline_dir = VISUALIZATION_ROOT / selected_pipeline

    if generation_statuses:
        with st.expander("Generation status", expanded=False):
            for status in generation_statuses:
                st.write(status)

    render_pipeline_snapshot(
        selected_pipeline=selected_pipeline,
        conformance_df=conformance_df,
        available_input_dates=available_dates,
    )

    gap_tab, drift_tab, petri_tab, dfg_tab, performance_tab, actions_tab, heatmap_tab = st.tabs([
        "Bilateral Gap",
        "Drift",
        "Petri Net",
        "Discovered DFG",
        "Performance DFG",
        "Prediction & Actions",
        "Fleet Heatmap",
    ])

    with gap_tab:
        if event_scope_available:
            render_visualization_image(
                pipeline_dir / "bilateral_gap_timeline.png",
                "Bilateral Gap Timeline",
                "No bilateral gap timeline exported yet.",
            )
            render_chart_note(
                "What this proves",
                "orange handoff lines show how long the consumer waits after the producer publishes DATA_AVAILABLE.",
            )
            render_visualization_image(
                pipeline_dir / "bilateral_gap_analysis.png",
                "Bilateral Gap Analysis",
                "No bilateral gap analysis exported yet.",
            )
            render_chart_note(
                "How to read it",
                "left bars show gap by run; the right trend panel shows whether the handoff delay is widening.",
            )
        else:
            st.subheader("Bilateral Gap Timeline")
            st.info(event_scope_status)

    with drift_tab:
        st.caption(
            "Drift chart colors represent final_score thresholds. "
            "A pipeline can have an AMBER timing_zone while its overall final_score remains GREEN."
        )
        render_visualization_image(
            pipeline_dir / "drift_chart.png",
            "Drift Chart",
            "No drift chart exported yet.",
        )
        render_chart_note(
            "What this proves",
            "the score history shows whether conformance is stable, improving, or drifting toward risk.",
        )

    with petri_tab:
        render_visualization_image(
            pipeline_dir / "contract_petri_net.png",
            "Contract Petri Net",
            "No contract Petri net exported yet.",
        )
        render_chart_note(
            "What this proves",
            "the YAML contract has been converted into a formal process model for token replay.",
        )

    with dfg_tab:
        if event_scope_available:
            render_visualization_image(
                pipeline_dir / "discovered_dfg_producer.png",
                "Discovered Producer DFG",
                "No producer DFG exported yet.",
            )
            render_chart_note(
                "Producer DFG",
                "the graph is discovered from actual producer events, not manually drawn from the contract.",
            )
            render_visualization_image(
                pipeline_dir / "discovered_dfg_consumer.png",
                "Discovered Consumer DFG",
                "No consumer DFG exported yet.",
            )
            render_chart_note(
                "Consumer DFG",
                "differences from the producer graph reveal downstream ordering or logging behavior.",
            )
        else:
            st.subheader("Discovered DFG")
            st.info(event_scope_status)

    with performance_tab:
        if event_scope_available:
            st.caption(
                "Performance DFG labels each actual arc with count, mean duration, "
                "and p95 duration. The highlighted arc is the slowest p95 bottleneck."
            )
            render_visualization_image(
                pipeline_dir / "performance_dfg_producer.png",
                "Producer Performance DFG",
                "No producer Performance DFG exported yet.",
            )
            render_chart_note(
                "Producer performance",
                "arc labels show where producer-side execution time accumulates between activities.",
            )
            render_visualization_image(
                pipeline_dir / "performance_dfg_consumer.png",
                "Consumer Performance DFG",
                "No consumer Performance DFG exported yet.",
            )
            render_chart_note(
                "Consumer performance",
                "consumer-side arcs include downstream waiting and pickup delay after producer availability.",
            )
            render_visualization_image(
                pipeline_dir / "execution_time_drift_producer.png",
                "Producer Execution Time Drift",
                "No producer execution time drift exported yet.",
            )
            render_chart_note(
                "Producer drift",
                "the trend line shows whether producer execution duration is moving toward SLA pressure.",
            )
            render_visualization_image(
                pipeline_dir / "execution_time_drift_consumer.png",
                "Consumer Execution Time Drift",
                "No consumer execution time drift exported yet.",
            )
            render_chart_note(
                "Consumer drift",
                "the trend line shows whether downstream pickup and processing are becoming slower.",
            )
        else:
            st.subheader("Performance DFG")
            st.info(event_scope_status)

    with actions_tab:
        st.caption(
            "Prediction reads conformance history. Actions translate the latest "
            "diagnostics into severity, owner, suggested fix, and next command."
        )

        prediction_payload, prediction_status = load_json_artifact(
            pipeline_dir / "prediction.json"
        )
        recommendation_payload, recommendation_status = load_json_artifact(
            pipeline_dir / "recommendations.json"
        )

        render_prediction_artifact(
            payload=prediction_payload,
            status=prediction_status,
            pipeline_id=selected_pipeline,
        )

        st.divider()

        render_recommendation_artifact(
            payload=recommendation_payload,
            status=recommendation_status,
            pipeline_id=selected_pipeline,
        )

    with heatmap_tab:
        render_visualization_image(
            VISUALIZATION_ROOT / "fleet_heatmap.png",
            "Fleet Weekday Heatmap",
            "No fleet heatmap exported yet.",
        )
        render_chart_note(
            "What this proves",
            "weekday bands reveal intermittent or calendar-specific fleet behavior across pipelines.",
        )


def main():
    """
    Main Streamlit entry point.

    The dashboard loads data once and routes pages from the sidebar.
    """
    st.sidebar.title("Fracture")

    page = st.sidebar.radio(
        "Page",
        [
            "Fleet Overview",
            "Pipeline Detail",
            "Visualizations",
        ],
    )

    conformance_df, conformance_status = load_conformance_log(str(CONFORMANCE_LOG_PATH))
    contracts_df, contracts_status = load_contracts(str(CONTRACTS_ROOT))

    st.sidebar.caption(f"Conformance: {conformance_status}")
    st.sidebar.caption(f"Contracts: {contracts_status}")
    st.sidebar.caption(f"Data root: {DATA_ROOT}")

    if page == "Fleet Overview":
        render_fleet_overview(conformance_df)

    elif page == "Pipeline Detail":
        render_pipeline_detail(conformance_df, contracts_df)

    elif page == "Visualizations":
        render_visualizations(conformance_df)


if __name__ == "__main__":
    main()
