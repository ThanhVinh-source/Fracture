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

from datetime import date
from pathlib import Path

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
CONTRACTS_ROOT = Path("contracts")
INPUTS_ROOT = Path("inputs")


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
            use_container_width=True,
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
            use_container_width=True,
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

    history = df[df["pipeline_id"] == selected_pipeline].copy()

    if history.empty:
        st.warning("No history for this pipeline.")
        return

    latest = history.sort_values("run_date").iloc[-1]

    st.subheader(selected_pipeline)
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
        st.image(str(path), use_container_width=True)
        st.caption(str(path))
        return

    st.info(missing_hint)


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


def ensure_visualizations_for_pipeline(
    pipeline_id: str,
    conformance_df: pd.DataFrame,
    date_str: str,
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

    contract, contract_status = load_pipeline_contract(pipeline_id)

    # Contract controls custom handoff event names. Defaults match Fracture vocabulary.
    producer_event = "DATA_AVAILABLE"
    consumer_event = "DATA_AVAILABLE"
    if contract is not None:
        producer_event = contract.log_contract.upstream_producer_event
        consumer_event = contract.log_contract.upstream_consumer_event

    gap_path = pipeline_dir / "bilateral_gap_timeline.png"
    gap_analysis_path = pipeline_dir / "bilateral_gap_analysis.png"
    if force or not gap_path.exists() or not gap_analysis_path.exists():
        producer_df, consumer_df, input_status = load_pipeline_events(
            inputs_dir=str(INPUTS_ROOT),
            pipeline_id=pipeline_id,
            date_str=date_str,
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
    if force or not dfg_path.exists():
        if contract is None:
            statuses.append(f"dfg skipped: {contract_status}")
        else:
            producer_df, consumer_df, input_status = load_pipeline_events(
                inputs_dir=str(INPUTS_ROOT),
                pipeline_id=pipeline_id,
                date_str=date_str,
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
    if force or not performance_path.exists() or not execution_drift_path.exists():
        if contract is None:
            statuses.append(f"performance skipped: {contract_status}")
        else:
            producer_df, consumer_df, input_status = load_pipeline_events(
                inputs_dir=str(INPUTS_ROOT),
                pipeline_id=pipeline_id,
                date_str=date_str,
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
        pipelines = sorted(conformance_df["pipeline_id"].dropna().unique())
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

    date_str = st.sidebar.text_input(
        "Input date",
        value=date.today().strftime("%Y%m%d"),
        help="Used for producer/consumer input files such as producer_YYYYMMDD.parquet.",
    )

    force_regenerate = st.sidebar.button("Regenerate visuals")

    with st.spinner("Preparing visualizations..."):
        generation_statuses = ensure_visualizations_for_pipeline(
            pipeline_id=selected_pipeline,
            conformance_df=conformance_df,
            date_str=date_str,
            force=force_regenerate,
        )

    pipeline_dir = VISUALIZATION_ROOT / selected_pipeline

    if generation_statuses:
        with st.expander("Generation status", expanded=False):
            for status in generation_statuses:
                st.write(status)

    gap_tab, drift_tab, petri_tab, dfg_tab, performance_tab, heatmap_tab = st.tabs([
        "Bilateral Gap",
        "Drift",
        "Petri Net",
        "Discovered DFG",
        "Performance DFG",
        "Fleet Heatmap",
    ])

    with gap_tab:
        render_visualization_image(
            pipeline_dir / "bilateral_gap_timeline.png",
            "Bilateral Gap Timeline",
            "No bilateral gap timeline exported yet.",
        )
        render_visualization_image(
            pipeline_dir / "bilateral_gap_analysis.png",
            "Bilateral Gap Analysis",
            "No bilateral gap analysis exported yet.",
        )

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

    with petri_tab:
        render_visualization_image(
            pipeline_dir / "contract_petri_net.png",
            "Contract Petri Net",
            "No contract Petri net exported yet.",
        )

    with dfg_tab:
        render_visualization_image(
            pipeline_dir / "discovered_dfg_producer.png",
            "Discovered Producer DFG",
            "No producer DFG exported yet.",
        )
        render_visualization_image(
            pipeline_dir / "discovered_dfg_consumer.png",
            "Discovered Consumer DFG",
            "No consumer DFG exported yet.",
        )

    with performance_tab:
        st.caption(
            "Performance DFG labels each actual arc with count, mean duration, "
            "and p95 duration. The highlighted arc is the slowest p95 bottleneck."
        )
        render_visualization_image(
            pipeline_dir / "performance_dfg_producer.png",
            "Producer Performance DFG",
            "No producer Performance DFG exported yet.",
        )
        render_visualization_image(
            pipeline_dir / "performance_dfg_consumer.png",
            "Consumer Performance DFG",
            "No consumer Performance DFG exported yet.",
        )
        render_visualization_image(
            pipeline_dir / "execution_time_drift_producer.png",
            "Producer Execution Time Drift",
            "No producer execution time drift exported yet.",
        )
        render_visualization_image(
            pipeline_dir / "execution_time_drift_consumer.png",
            "Consumer Execution Time Drift",
            "No consumer execution time drift exported yet.",
        )

    with heatmap_tab:
        render_visualization_image(
            VISUALIZATION_ROOT / "fleet_heatmap.png",
            "Fleet Weekday Heatmap",
            "No fleet heatmap exported yet.",
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

    conformance_df, conformance_status = load_conformance_log("conformance_log.csv")
    contracts_df, contracts_status = load_contracts("contracts")

    st.sidebar.caption(f"Conformance: {conformance_status}")
    st.sidebar.caption(f"Contracts: {contracts_status}")

    if page == "Fleet Overview":
        render_fleet_overview(conformance_df)

    elif page == "Pipeline Detail":
        render_pipeline_detail(conformance_df, contracts_df)

    elif page == "Visualizations":
        render_visualizations(conformance_df)


if __name__ == "__main__":
    main()
