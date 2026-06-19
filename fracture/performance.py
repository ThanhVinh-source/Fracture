"""
fracture/performance.py

Phase 5 performance-mining helpers.

This module answers:
"Which part of the actual process is slow?"

It builds on the same event format as discovery and conformance, but focuses on
durations between consecutive activities instead of only sequence/frequency.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fracture.discovery import _normalize_for_discovery, _ordered_variant


@dataclass
class PerformanceSummary:
    """
    Structured performance-mining result for one event log side.

    The fields are plain Python values so CLI, dashboard, JSON export, and tests
    can reuse the same object.
    """

    pipeline_id: str
    log_side: str
    status: str
    n_traces: int
    nodes: list[str]
    arcs: list[dict]
    run_duration_minutes: dict
    bottleneck_arc: dict | None

    @property
    def n_arcs(self) -> int:
        """Return the number of unique timed directly-follows arcs."""
        return len(self.arcs)

    def as_dataframe(self) -> pd.DataFrame:
        """Return performance arcs as a DataFrame for tables/charts."""
        return pd.DataFrame(self.arcs)

    def as_dict(self) -> dict:
        """Return plain Python values for JSON/report/dashboard use."""
        return {
            "pipeline_id": self.pipeline_id,
            "log_side": self.log_side,
            "status": self.status,
            "n_traces": self.n_traces,
            "nodes": self.nodes,
            "arcs": self.arcs,
            "n_arcs": self.n_arcs,
            "run_duration_minutes": self.run_duration_minutes,
            "bottleneck_arc": self.bottleneck_arc,
        }


def _duration_summary(values: list[float]) -> dict:
    """
    Summarise duration values in minutes.

    Percentiles are useful because process performance often has a long tail:
    p95 reveals rare slow arcs that a mean can hide.
    """
    if not values:
        return {}

    arr = np.array(values, dtype=float)

    return {
        "count": int(len(arr)),
        "mean_minutes": round(float(arr.mean()), 4),
        "p50_minutes": round(float(np.percentile(arr, 50)), 4),
        "p95_minutes": round(float(np.percentile(arr, 95)), 4),
        "max_minutes": round(float(arr.max()), 4),
    }


def build_performance_summary(
    events: pd.DataFrame,
    contract,
    log_side: str = "producer",
) -> PerformanceSummary:
    """
    Compute arc duration metrics from actual event traces.

    One trace is one pipeline_run_id. Each adjacent activity pair contributes a
    duration in minutes from source timestamp to target timestamp.
    """
    if events is None or events.empty:
        return PerformanceSummary(
            pipeline_id=contract.pipeline_id,
            log_side=log_side,
            status="no_input",
            n_traces=0,
            nodes=[],
            arcs=[],
            run_duration_minutes={},
            bottleneck_arc=None,
        )

    normalized = _normalize_for_discovery(events, contract)

    arc_durations = defaultdict(list)
    run_durations = []
    nodes = set()
    trace_count = 0

    for _run_id, trace_df in normalized.groupby("pipeline_run_id"):
        ordered = trace_df.sort_values("timestamp").reset_index(drop=True)
        activities = _ordered_variant(ordered)

        if not activities:
            continue

        trace_count += 1
        nodes.update(activities)

        if len(ordered) >= 2:
            run_minutes = (
                ordered["timestamp"].iloc[-1] - ordered["timestamp"].iloc[0]
            ).total_seconds() / 60.0

            if run_minutes >= 0:
                run_durations.append(run_minutes)

        for idx in range(len(ordered) - 1):
            source = str(ordered.loc[idx, "activity"])
            target = str(ordered.loc[idx + 1, "activity"])
            duration_minutes = (
                ordered.loc[idx + 1, "timestamp"] - ordered.loc[idx, "timestamp"]
            ).total_seconds() / 60.0

            # Negative durations usually mean clock skew or bad extraction.
            # Skip them here so bottleneck detection is not corrupted.
            if duration_minutes < 0:
                continue

            arc_durations[(source, target)].append(duration_minutes)

    if trace_count == 0:
        return PerformanceSummary(
            pipeline_id=contract.pipeline_id,
            log_side=log_side,
            status="no_traces",
            n_traces=0,
            nodes=[],
            arcs=[],
            run_duration_minutes={},
            bottleneck_arc=None,
        )

    if not arc_durations:
        return PerformanceSummary(
            pipeline_id=contract.pipeline_id,
            log_side=log_side,
            status="no_timed_arcs",
            n_traces=trace_count,
            nodes=sorted(nodes),
            arcs=[],
            run_duration_minutes=_duration_summary(run_durations),
            bottleneck_arc=None,
        )

    arcs = []
    for (source, target), values in arc_durations.items():
        arc = {
            "source": source,
            "target": target,
            **_duration_summary(values),
        }
        arcs.append(arc)

    # Slowest p95 is used as the bottleneck because tail latency is usually the
    # operational pain point for SLA-sensitive pipelines.
    arcs = sorted(
        arcs,
        key=lambda arc: (-arc["p95_minutes"], -arc["mean_minutes"], arc["source"], arc["target"]),
    )
    bottleneck_arc = arcs[0] if arcs else None

    return PerformanceSummary(
        pipeline_id=contract.pipeline_id,
        log_side=log_side,
        status="ok",
        n_traces=trace_count,
        nodes=sorted(nodes),
        arcs=arcs,
        run_duration_minutes=_duration_summary(run_durations),
        bottleneck_arc=bottleneck_arc,
    )


def format_performance_summary(summary: PerformanceSummary) -> str:
    """
    Format performance-mining output for CLI use.

    Keep this plain text so it is easy to paste into reports and compare with
    the static Performance DFG visual.
    """
    lines = [
        f"Performance Analysis: {summary.pipeline_id}",
        f"Log side: {summary.log_side}",
        f"Status: {summary.status}",
        f"Traces: {summary.n_traces}",
        f"Timed arcs: {summary.n_arcs}",
    ]

    if summary.run_duration_minutes:
        lines.extend(["", "Run duration minutes:"])
        for key, value in summary.run_duration_minutes.items():
            lines.append(f"  {key}: {value}")

    if summary.bottleneck_arc:
        arc = summary.bottleneck_arc
        lines.extend([
            "",
            "Bottleneck arc:",
            f"  {arc['source']} -> {arc['target']}",
            f"  mean_minutes: {arc['mean_minutes']}",
            f"  p95_minutes: {arc['p95_minutes']}",
            f"  count: {arc['count']}",
        ])

    if summary.arcs:
        lines.extend(["", "Top arcs by p95 duration:"])
        for arc in summary.arcs[:5]:
            lines.append(
                f"  {arc['source']} -> {arc['target']}: "
                f"mean={arc['mean_minutes']}m, p95={arc['p95_minutes']}m, "
                f"count={arc['count']}"
            )

    return "\n".join(lines)
