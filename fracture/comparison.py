"""
fracture/comparison.py

Comparative process-mining helpers.

This module answers:
"What is different between two process perspectives?"

V1 supports:
- producer vs consumer handoff comparison
- contract vs actual dominant variant comparison
- current period vs previous period comparison from conformance_log.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from fracture.discovery import discover_variants
from fracture.ingest import FRACTURE_COLUMNS, normalize_events


@dataclass
class ComparisonResult:
    """
    Structured comparison result for CLI, dashboard, tests, and reports.

    `metrics` contains numeric or short scalar values. `findings` contains
    human-readable interpretation of what changed or differs.
    """

    pipeline_id: str
    mode: str
    status: str
    metrics: dict
    findings: list[str]
    details: dict

    def as_dict(self) -> dict:
        """Return plain Python values for JSON/report/dashboard use."""
        return {
            "pipeline_id": self.pipeline_id,
            "mode": self.mode,
            "status": self.status,
            "metrics": self.metrics,
            "findings": self.findings,
            "details": self.details,
        }


def _empty_result(pipeline_id: str, mode: str, status: str, finding: str) -> ComparisonResult:
    """Build a small blocked/empty comparison result."""
    return ComparisonResult(
        pipeline_id=pipeline_id,
        mode=mode,
        status=status,
        metrics={},
        findings=[finding],
        details={},
    )


def _normalize_for_comparison(events: pd.DataFrame, contract) -> pd.DataFrame:
    """
    Normalize event logs before comparing process perspectives.

    Comparison must use the same activity vocabulary as conformance, discovery,
    and performance mining.
    """
    missing = [col for col in FRACTURE_COLUMNS if col not in events.columns]
    if missing:
        raise ValueError(
            "Comparison input is missing required columns: "
            + ", ".join(missing)
        )

    normalized = normalize_events(events, contract)
    normalized = normalized.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)

    return normalized


def _handoff_points(events: pd.DataFrame, activity: str, timestamp_name: str) -> pd.DataFrame:
    """
    Extract one handoff timestamp per pipeline_run_id.

    If a handoff event appears more than once, keep the latest timestamp because
    repeated handoff markers usually mean retry/re-publish behavior.
    """
    points = events[events["activity"] == activity].copy()

    if points.empty:
        return pd.DataFrame(columns=["pipeline_run_id", timestamp_name])

    points = (
        points
        .sort_values(["pipeline_run_id", "timestamp"])
        .drop_duplicates(["pipeline_run_id"], keep="last")
        [["pipeline_run_id", "timestamp"]]
        .rename(columns={"timestamp": timestamp_name})
    )

    return points


def compare_producer_consumer(
    producer_events: Optional[pd.DataFrame],
    consumer_events: Optional[pd.DataFrame],
    contract,
    producer_event: Optional[str] = None,
    consumer_event: Optional[str] = None,
    gap_warning_minutes: float = 10.0,
    gap_severe_minutes: float = 20.0,
) -> ComparisonResult:
    """
    Compare producer and consumer handoff timestamps.

    This is the flagship bilateral process-mining comparison: producer says data
    is available, consumer shows when that availability is actually observed.
    """
    pipeline_id = contract.pipeline_id

    if producer_events is None or producer_events.empty:
        return _empty_result(
            pipeline_id,
            "producer-consumer",
            "no_producer_input",
            "Producer log is missing, so bilateral handoff comparison cannot run.",
        )

    if consumer_events is None or consumer_events.empty:
        return _empty_result(
            pipeline_id,
            "producer-consumer",
            "no_consumer_input",
            "Consumer log is missing; producer-only mode cannot measure bilateral gap.",
        )

    producer_event = producer_event or contract.log_contract.upstream_producer_event
    consumer_event = consumer_event or contract.log_contract.upstream_consumer_event

    producer_normalized = _normalize_for_comparison(producer_events, contract)
    consumer_normalized = _normalize_for_comparison(consumer_events, contract)

    producer_points = _handoff_points(
        producer_normalized,
        producer_event,
        "producer_timestamp",
    )
    consumer_points = _handoff_points(
        consumer_normalized,
        consumer_event,
        "consumer_timestamp",
    )

    if producer_points.empty:
        return _empty_result(
            pipeline_id,
            "producer-consumer",
            "missing_producer_handoff",
            f"Producer event '{producer_event}' was not found.",
        )

    if consumer_points.empty:
        return _empty_result(
            pipeline_id,
            "producer-consumer",
            "missing_consumer_handoff",
            f"Consumer event '{consumer_event}' was not found.",
        )

    matched = producer_points.merge(
        consumer_points,
        on="pipeline_run_id",
        how="inner",
    )

    producer_runs = set(producer_points["pipeline_run_id"].astype(str))
    consumer_runs = set(consumer_points["pipeline_run_id"].astype(str))
    missing_consumer_runs = sorted(producer_runs - consumer_runs)
    missing_producer_runs = sorted(consumer_runs - producer_runs)

    if matched.empty:
        return ComparisonResult(
            pipeline_id=pipeline_id,
            mode="producer-consumer",
            status="no_matched_handoff",
            metrics={
                "producer_handoff_runs": len(producer_runs),
                "consumer_handoff_runs": len(consumer_runs),
            },
            findings=["No pipeline_run_id has both producer and consumer handoff events."],
            details={
                "missing_consumer_runs": missing_consumer_runs,
                "missing_producer_runs": missing_producer_runs,
            },
        )

    matched["gap_minutes"] = (
        matched["consumer_timestamp"] - matched["producer_timestamp"]
    ).dt.total_seconds() / 60.0
    matched = matched.sort_values("producer_timestamp").reset_index(drop=True)

    gaps = matched["gap_minutes"].astype(float)
    mean_gap = float(gaps.mean())
    p95_gap = float(np.percentile(gaps, 95))
    min_gap = float(gaps.min())
    max_gap = float(gaps.max())
    negative_count = int((gaps < 0).sum())

    metrics = {
        "matched_runs": int(len(matched)),
        "producer_handoff_runs": int(len(producer_runs)),
        "consumer_handoff_runs": int(len(consumer_runs)),
        "missing_consumer_runs": int(len(missing_consumer_runs)),
        "missing_producer_runs": int(len(missing_producer_runs)),
        "mean_gap_minutes": round(mean_gap, 4),
        "p95_gap_minutes": round(p95_gap, 4),
        "min_gap_minutes": round(min_gap, 4),
        "max_gap_minutes": round(max_gap, 4),
        "negative_gap_count": negative_count,
        "producer_event": producer_event,
        "consumer_event": consumer_event,
    }

    findings = []

    if negative_count:
        findings.append(
            f"{negative_count} matched runs have negative gap; check clock sync or timezone handling."
        )

    if p95_gap >= gap_severe_minutes:
        findings.append(
            f"p95 bilateral gap is severe at {p95_gap:.1f} minutes."
        )
    elif p95_gap >= gap_warning_minutes:
        findings.append(
            f"p95 bilateral gap is elevated at {p95_gap:.1f} minutes."
        )
    else:
        findings.append(
            f"Producer-consumer handoff is within warning threshold; p95 gap is {p95_gap:.1f} minutes."
        )

    if missing_consumer_runs:
        findings.append(
            f"{len(missing_consumer_runs)} producer handoff runs are missing consumer acknowledgement."
        )

    if missing_producer_runs:
        findings.append(
            f"{len(missing_producer_runs)} consumer handoff runs have no matching producer handoff."
        )

    return ComparisonResult(
        pipeline_id=pipeline_id,
        mode="producer-consumer",
        status="ok",
        metrics=metrics,
        findings=findings,
        details={
            "matched_gaps": matched.to_dict(orient="records"),
            "missing_consumer_runs": missing_consumer_runs,
            "missing_producer_runs": missing_producer_runs,
        },
    )


def compare_contract_actual(
    events: Optional[pd.DataFrame],
    contract,
    log_side: str = "producer",
) -> ComparisonResult:
    """
    Compare the discovered dominant actual variant against the contract path.

    This turns process discovery into a direct expected-vs-actual comparison.
    """
    summary = discover_variants(events, contract, log_side=log_side)

    metrics = {
        "log_side": log_side,
        "traces": summary.n_traces,
        "variants": summary.n_variants,
        "dominant_variant_count": summary.dominant_variant_count,
        "dominant_variant_frequency": round(summary.dominant_variant_frequency, 4),
        "dominant_matches_contract": summary.dominant_matches_contract,
    }

    if summary.status != "ok":
        return ComparisonResult(
            pipeline_id=contract.pipeline_id,
            mode="contract-actual",
            status=summary.status,
            metrics=metrics,
            findings=summary.deviations,
            details=summary.as_dict(),
        )

    findings = []
    if summary.dominant_matches_contract:
        findings.append("Dominant actual variant matches the contract path.")
    else:
        findings.extend(summary.deviations or ["Dominant actual variant differs from the contract path."])

    return ComparisonResult(
        pipeline_id=contract.pipeline_id,
        mode="contract-actual",
        status="ok",
        metrics=metrics,
        findings=findings,
        details=summary.as_dict(),
    )


def _parse_conformance_dates(values: pd.Series) -> pd.Series:
    """
    Parse conformance_log run_date values safely.

    Fracture stores dates as YYYYMMDD, but pandas may read them as integers.
    """
    as_text = values.astype(str).str.strip()
    compact_mask = as_text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    if compact_mask.any():
        parsed.loc[compact_mask] = pd.to_datetime(
            as_text.loc[compact_mask],
            format="%Y%m%d",
            errors="coerce",
        )

    if (~compact_mask).any():
        parsed.loc[~compact_mask] = pd.to_datetime(
            as_text.loc[~compact_mask],
            errors="coerce",
        )

    return parsed


def compare_periods(
    conformance_df: pd.DataFrame,
    pipeline_id: str,
    period_size: int = 7,
) -> ComparisonResult:
    """
    Compare latest N conformance rows with the previous N rows.

    This gives a lightweight current-vs-previous comparative mining view using
    already persisted conformance history.
    """
    required = {"pipeline_id", "run_date", "final_score"}
    missing = required - set(conformance_df.columns)
    if missing:
        return _empty_result(
            pipeline_id,
            "period",
            "missing_columns",
            f"conformance_log.csv is missing required columns: {sorted(missing)}",
        )

    history = conformance_df[conformance_df["pipeline_id"] == pipeline_id].copy()

    if history.empty:
        return _empty_result(
            pipeline_id,
            "period",
            "no_history",
            "No conformance history exists for this pipeline.",
        )

    history["run_date"] = _parse_conformance_dates(history["run_date"])
    history["final_score"] = pd.to_numeric(history["final_score"], errors="coerce")

    if "bilateral_gap_minutes" in history.columns:
        history["bilateral_gap_minutes"] = pd.to_numeric(
            history["bilateral_gap_minutes"],
            errors="coerce",
        )

    history = history.dropna(subset=["run_date", "final_score"])
    history = history.sort_values("run_date").reset_index(drop=True)

    if len(history) < period_size * 2:
        return ComparisonResult(
            pipeline_id=pipeline_id,
            mode="period",
            status="insufficient_history",
            metrics={
                "available_rows": int(len(history)),
                "required_rows": int(period_size * 2),
                "period_size": int(period_size),
            },
            findings=[
                f"Need at least {period_size * 2} scored rows to compare current vs previous periods."
            ],
            details={},
        )

    previous = history.iloc[-period_size * 2:-period_size]
    current = history.iloc[-period_size:]

    current_score = float(current["final_score"].mean())
    previous_score = float(previous["final_score"].mean())
    score_delta = current_score - previous_score

    metrics = {
        "period_size": int(period_size),
        "previous_start": str(previous["run_date"].iloc[0].date()),
        "previous_end": str(previous["run_date"].iloc[-1].date()),
        "current_start": str(current["run_date"].iloc[0].date()),
        "current_end": str(current["run_date"].iloc[-1].date()),
        "previous_mean_score": round(previous_score, 4),
        "current_mean_score": round(current_score, 4),
        "score_delta": round(score_delta, 4),
    }

    if "bilateral_gap_minutes" in history.columns:
        previous_gap = previous["bilateral_gap_minutes"].mean()
        current_gap = current["bilateral_gap_minutes"].mean()

        if not pd.isna(previous_gap) and not pd.isna(current_gap):
            metrics["previous_mean_gap_minutes"] = round(float(previous_gap), 4)
            metrics["current_mean_gap_minutes"] = round(float(current_gap), 4)
            metrics["gap_delta_minutes"] = round(float(current_gap - previous_gap), 4)

    findings = []

    if score_delta <= -0.05:
        findings.append(
            f"Current period score dropped by {abs(score_delta):.3f} versus previous period."
        )
    elif score_delta >= 0.05:
        findings.append(
            f"Current period score improved by {score_delta:.3f} versus previous period."
        )
    else:
        findings.append(
            f"Current period score is broadly stable; delta is {score_delta:.3f}."
        )

    gap_delta = metrics.get("gap_delta_minutes")
    if gap_delta is not None:
        if gap_delta >= 5:
            findings.append(
                f"Bilateral gap widened by {gap_delta:.1f} minutes on average."
            )
        elif gap_delta <= -5:
            findings.append(
                f"Bilateral gap narrowed by {abs(gap_delta):.1f} minutes on average."
            )
        else:
            findings.append(
                f"Bilateral gap is broadly stable; average delta is {gap_delta:.1f} minutes."
            )

    return ComparisonResult(
        pipeline_id=pipeline_id,
        mode="period",
        status="ok",
        metrics=metrics,
        findings=findings,
        details={
            "previous_rows": previous.to_dict(orient="records"),
            "current_rows": current.to_dict(orient="records"),
        },
    )


def format_comparison_result(result: ComparisonResult) -> str:
    """
    Format comparison output for CLI use.

    Keep this plain text so it is easy to paste into reports or terminal notes.
    """
    lines = [
        f"Comparative Process Mining: {result.pipeline_id}",
        f"Mode: {result.mode}",
        f"Status: {result.status}",
    ]

    if result.metrics:
        lines.extend(["", "Metrics:"])
        for key, value in result.metrics.items():
            lines.append(f"  {key}: {value}")

    if result.findings:
        lines.extend(["", "Findings:"])
        lines.extend([f"  - {finding}" for finding in result.findings])

    return "\n".join(lines)
