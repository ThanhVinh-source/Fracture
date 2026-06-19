"""
fracture/recommendation.py

Phase 5 action-oriented process-mining helpers.

This module answers:
"What should the pipeline owner do next?"

The rules intentionally stay transparent. Recommendations are derived from
conformance diagnostics, bilateral gap metrics, and optional prediction output.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SEVERITY_RANK = {
    "INFO": 0,
    "WATCH": 1,
    "ACTION": 2,
    "URGENT": 3,
    "BLOCKED": 4,
}


@dataclass
class RecommendationResult:
    """
    Structured recommendation result for CLI, dashboard, tests, and reports.

    `recommendations` is a list because one pipeline can have multiple
    findings, for example severe bilateral gap and weak score confidence.
    """

    pipeline_id: str
    status: str
    recommendations: list[dict]
    metrics: dict

    def as_dict(self) -> dict:
        """Return plain Python values for JSON/report/dashboard use."""
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
        }


def _parse_run_dates(values: pd.Series) -> pd.Series:
    """
    Parse conformance_log run_date values safely.

    Fracture normally stores YYYYMMDD, but reports/tests may use full
    timestamps. Recommendation only needs reliable ordering.
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


def _numeric(row: pd.Series, key: str):
    """
    Read optional numeric values from a conformance_log row.

    Empty CSV cells should behave like missing values, not zero.
    """
    if key not in row:
        return None

    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _text(row: pd.Series, key: str, default: str = "") -> str:
    """Read optional text values from a conformance_log row."""
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return str(value).strip()


def _recommendation(
    pipeline_id: str,
    severity: str,
    owner: str,
    finding: str,
    probable_cause: str,
    recommended_action: str,
    next_command: str,
    confidence: str,
) -> dict:
    """Build one recommendation row using the public output schema."""
    return {
        "pipeline_id": pipeline_id,
        "severity": severity,
        "owner": owner,
        "finding": finding,
        "probable_cause": probable_cause,
        "recommended_action": recommended_action,
        "next_command": next_command,
        "confidence": confidence,
    }


def _sort_recommendations(items: list[dict]) -> list[dict]:
    """Return recommendations from highest severity to lowest severity."""
    return sorted(
        items,
        key=lambda item: (-SEVERITY_RANK.get(item["severity"], 0), item["finding"]),
    )


def _latest_scored_row(conformance_df: pd.DataFrame, pipeline_id: str) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Return all valid rows and the latest row for one pipeline.

    Rows without final_score are kept in the history but are not considered the
    latest measured state because they represent NO_INPUT/BLOCKED runs.
    """
    if "pipeline_id" not in conformance_df.columns:
        return pd.DataFrame(), None

    history = conformance_df[conformance_df["pipeline_id"] == pipeline_id].copy()
    if history.empty or "run_date" not in history.columns:
        return history, None

    history["run_date_parsed"] = _parse_run_dates(history["run_date"])

    if "final_score" in history.columns:
        history["final_score_numeric"] = pd.to_numeric(
            history["final_score"],
            errors="coerce",
        )
    else:
        history["final_score_numeric"] = pd.NA

    history = history.sort_values("run_date_parsed").reset_index(drop=True)
    scored = history.dropna(subset=["run_date_parsed", "final_score_numeric"])

    if scored.empty:
        return history, None

    return history, scored.iloc[-1]


def _recommend_from_prediction(
    pipeline_id: str,
    owner: str,
    prediction_result,
    current_gap_minutes: float | None = None,
    gap_severe_threshold: float = 20.0,
) -> list[dict]:
    """
    Convert prediction records into action rows.

    Prediction is optional. If not available, recommendation still works from
    latest conformance diagnostics.
    """
    if not prediction_result or getattr(prediction_result, "status", "") != "ok":
        return []

    items = []
    for prediction in getattr(prediction_result, "predictions", []):
        prediction_type = prediction.get("prediction_type", "")
        days = prediction.get("days_until_projection")
        confidence = prediction.get("confidence", "UNKNOWN")

        if prediction_type == "score_critical" and days is not None and days <= 30:
            severity = "URGENT" if days <= 14 else "ACTION"
            items.append(_recommendation(
                pipeline_id=pipeline_id,
                severity=severity,
                owner=owner,
                finding=f"Score is projected to cross the critical threshold in {days} days.",
                probable_cause="Historical final_score trend is declining.",
                recommended_action="Review recent process changes and schedule a contract/conformance review.",
                next_command=f"python -m fracture.cli predict --pipeline-id {pipeline_id}",
                confidence=confidence,
            ))

        if prediction_type == "gap_severe" and days is not None:
            # If the latest measured gap is already severe, the direct
            # conformance rule gives a clearer action than repeating the
            # prediction signal.
            if (
                current_gap_minutes is not None
                and current_gap_minutes >= gap_severe_threshold
            ):
                continue

            severity = "URGENT" if days <= 14 else "ACTION"
            items.append(_recommendation(
                pipeline_id=pipeline_id,
                severity=severity,
                owner=owner,
                finding="Bilateral gap is projected to reach or remain in the severe zone.",
                probable_cause="Producer-consumer handoff delay is widening over time.",
                recommended_action="Review downstream pickup timing and the handoff contract with both teams.",
                next_command=f"python -m fracture.cli compare --pipeline-id {pipeline_id} --mode producer-consumer",
                confidence=confidence,
            ))

    return items


def recommend_pipeline(
    conformance_df: pd.DataFrame,
    pipeline_id: str,
    prediction_result=None,
    owner: str | None = None,
    gap_warning_threshold: float = 10.0,
    gap_severe_threshold: float = 20.0,
) -> RecommendationResult:
    """
    Generate action-oriented recommendations for one pipeline.

    Recommendations use latest measured conformance first. Prediction is used
    only as an extra signal when it is available.
    """
    required = {"pipeline_id", "run_date"}
    missing = required - set(conformance_df.columns)
    if missing:
        return RecommendationResult(
            pipeline_id=pipeline_id,
            status="missing_columns",
            recommendations=[_recommendation(
                pipeline_id=pipeline_id,
                severity="BLOCKED",
                owner=owner or "data-owner",
                finding=f"conformance_log.csv is missing required columns: {sorted(missing)}",
                probable_cause="The persisted measurement log is incomplete.",
                recommended_action="Regenerate conformance_log.csv by running conformance again.",
                next_command="python -m fracture.cli run-all",
                confidence="HIGH",
            )],
            metrics={},
        )

    history, latest = _latest_scored_row(conformance_df, pipeline_id)
    if history.empty:
        return RecommendationResult(
            pipeline_id=pipeline_id,
            status="no_history",
            recommendations=[_recommendation(
                pipeline_id=pipeline_id,
                severity="BLOCKED",
                owner=owner or "data-owner",
                finding="No conformance history exists for this pipeline.",
                probable_cause="The pipeline has not been measured yet.",
                recommended_action="Run conformance after input files are available.",
                next_command=f"python -m fracture.cli run --pipeline-id {pipeline_id}",
                confidence="HIGH",
            )],
            metrics={},
        )

    if latest is None:
        latest_status = _text(history.iloc[-1], "confidence_level", "NO_INPUT")
        return RecommendationResult(
            pipeline_id=pipeline_id,
            status="blocked",
            recommendations=[_recommendation(
                pipeline_id=pipeline_id,
                severity="BLOCKED",
                owner=owner or _text(history.iloc[-1], "alert_owner", "data-owner"),
                finding=f"Latest available rows are not scored; status is {latest_status}.",
                probable_cause="Input files are missing or the pipeline run could not be measured.",
                recommended_action="Check extraction output and rerun conformance for the missing date.",
                next_command=f"python -m fracture.cli status --pipeline-id {pipeline_id}",
                confidence="HIGH",
            )],
            metrics={"history_rows": int(len(history)), "scored_rows": 0},
        )

    effective_owner = (
        owner
        or _text(latest, "alert_owner")
        or _text(latest, "owner")
        or "pipeline-owner"
    )

    final_score = _numeric(latest, "final_score")
    sequence_fitness = _numeric(latest, "sequence_fitness")
    timing_score = _numeric(latest, "timing_score")
    completeness_score = _numeric(latest, "completeness_score")
    gap_minutes = _numeric(latest, "bilateral_gap_minutes")
    confidence_level = _text(latest, "confidence_level", "UNKNOWN").upper()
    timing_zone = _text(latest, "timing_zone", "UNKNOWN").upper()
    pattern = _text(latest, "pattern", "UNKNOWN").upper()
    run_date = _text(latest, "run_date")

    metrics = {
        "run_date": run_date,
        "final_score": round(final_score, 4) if final_score is not None else None,
        "confidence_level": confidence_level,
        "timing_zone": timing_zone,
        "pattern": pattern,
        "bilateral_gap_minutes": round(gap_minutes, 4) if gap_minutes is not None else None,
        "history_rows": int(len(history)),
        "scored_rows": int(history["final_score_numeric"].notna().sum()),
    }

    recommendations = []

    if confidence_level in {"LOW", "UNRELIABLE"}:
        recommendations.append(_recommendation(
            pipeline_id=pipeline_id,
            severity="BLOCKED",
            owner=effective_owner,
            finding=f"Measurement confidence is {confidence_level}.",
            probable_cause="Log extraction quality, trace coverage, or warmup history is insufficient.",
            recommended_action="Fix log extraction or collect more successful historical runs before acting on the score.",
            next_command=f"python -m fracture.cli explain --pipeline-id {pipeline_id} --verbose",
            confidence="HIGH",
        ))

    if timing_zone == "BREACH":
        recommendations.append(_recommendation(
            pipeline_id=pipeline_id,
            severity="URGENT",
            owner=effective_owner,
            finding="Latest run breached its timing SLA.",
            probable_cause="Observed completion time exceeded p99 plus grace window.",
            recommended_action="Escalate to the pipeline owner and inspect the slowest process segment.",
            next_command=f"python -m fracture.cli visualize --pipeline-id {pipeline_id} --kind performance",
            confidence=confidence_level,
        ))
    elif timing_zone in {"RED", "AMBER"}:
        severity = "ACTION" if timing_zone == "RED" else "WATCH"
        recommendations.append(_recommendation(
            pipeline_id=pipeline_id,
            severity=severity,
            owner=effective_owner,
            finding=f"Latest timing zone is {timing_zone}.",
            probable_cause="The run is approaching or entering its SLA risk boundary.",
            recommended_action="Monitor execution duration and inspect performance drift.",
            next_command=f"python -m fracture.cli visualize --pipeline-id {pipeline_id} --kind performance",
            confidence=confidence_level,
        ))

    if gap_minutes is not None:
        if gap_minutes < 0:
            recommendations.append(_recommendation(
                pipeline_id=pipeline_id,
                severity="ACTION",
                owner=effective_owner,
                finding=f"Bilateral gap is negative at {gap_minutes:.1f} minutes.",
                probable_cause="Producer and consumer clocks or timezones may be misaligned.",
                recommended_action="Check timestamp timezone normalization and system clock sync.",
                next_command=f"python -m fracture.cli compare --pipeline-id {pipeline_id} --mode producer-consumer",
                confidence="HIGH",
            ))
        elif gap_minutes >= gap_severe_threshold:
            recommendations.append(_recommendation(
                pipeline_id=pipeline_id,
                severity="URGENT",
                owner=effective_owner,
                finding=f"Bilateral gap is severe at {gap_minutes:.1f} minutes.",
                probable_cause="Consumer observes DATA_AVAILABLE much later than producer publishes it.",
                recommended_action="Review producer-consumer handoff, pickup schedule, and downstream acknowledgement.",
                next_command=f"python -m fracture.cli compare --pipeline-id {pipeline_id} --mode producer-consumer",
                confidence="HIGH",
            ))
        elif gap_minutes >= gap_warning_threshold:
            recommendations.append(_recommendation(
                pipeline_id=pipeline_id,
                severity="ACTION",
                owner=effective_owner,
                finding=f"Bilateral gap is elevated at {gap_minutes:.1f} minutes.",
                probable_cause="Handoff latency may become a downstream delay.",
                recommended_action="Monitor the handoff and check whether consumer polling can be tightened.",
                next_command=f"python -m fracture.cli visualize --pipeline-id {pipeline_id} --kind gap",
                confidence="HIGH",
            ))

    if pattern == "INTERMITTENT":
        owner_for_pattern = (
            "platform-infrastructure"
            if effective_owner == "platform-infrastructure"
            else effective_owner
        )
        recommendations.append(_recommendation(
            pipeline_id=pipeline_id,
            severity="ACTION",
            owner=owner_for_pattern,
            finding="Pipeline shows intermittent behavior.",
            probable_cause="Failures cluster on specific days or infrastructure windows.",
            recommended_action="Inspect weekday heatmap and route recurring day-specific failures to infrastructure.",
            next_command="python -m fracture.cli visualize --kind heatmap",
            confidence=confidence_level,
        ))

    if pattern == "DRIFTING":
        recommendations.append(_recommendation(
            pipeline_id=pipeline_id,
            severity="ACTION",
            owner=effective_owner,
            finding="Pipeline score pattern is drifting.",
            probable_cause="Historical conformance score is declining over time.",
            recommended_action="Review recent process changes and refresh the contract if the expected process changed.",
            next_command=f"python -m fracture.cli predict --pipeline-id {pipeline_id}",
            confidence=confidence_level,
        ))

    if sequence_fitness is not None and sequence_fitness < 1.0:
        recommendations.append(_recommendation(
            pipeline_id=pipeline_id,
            severity="ACTION",
            owner=effective_owner,
            finding=f"Sequence fitness is below perfect at {sequence_fitness:.3f}.",
            probable_cause="The actual trace has missing, repeated, or out-of-order activities.",
            recommended_action="Compare actual variants against the contract and check optional/retry settings.",
            next_command=f"python -m fracture.cli compare --pipeline-id {pipeline_id} --mode contract-actual",
            confidence=confidence_level,
        ))

    if completeness_score is not None and completeness_score < 0.9:
        recommendations.append(_recommendation(
            pipeline_id=pipeline_id,
            severity="ACTION",
            owner=effective_owner,
            finding=f"Completeness score is low at {completeness_score:.3f}.",
            probable_cause="Required or terminal events may be missing from extraction.",
            recommended_action="Fix event extraction or activity_name_map before changing the process contract.",
            next_command=f"python -m fracture.cli explain --pipeline-id {pipeline_id} --verbose",
            confidence=confidence_level,
        ))

    if timing_score is not None and timing_score < 0.85:
        recommendations.append(_recommendation(
            pipeline_id=pipeline_id,
            severity="ACTION",
            owner=effective_owner,
            finding=f"Timing score is weak at {timing_score:.3f}.",
            probable_cause="Execution time is slower than historical SLA percentiles.",
            recommended_action="Inspect performance DFG to locate the slowest arc.",
            next_command=f"python -m fracture.cli visualize --pipeline-id {pipeline_id} --kind performance",
            confidence=confidence_level,
        ))

    recommendations.extend(_recommend_from_prediction(
        pipeline_id=pipeline_id,
        owner=effective_owner,
        prediction_result=prediction_result,
        current_gap_minutes=gap_minutes,
        gap_severe_threshold=gap_severe_threshold,
    ))

    if not recommendations:
        recommendations.append(_recommendation(
            pipeline_id=pipeline_id,
            severity="INFO",
            owner=effective_owner,
            finding="No immediate action required.",
            probable_cause="Latest conformance, timing, and handoff metrics are within expected bounds.",
            recommended_action="Continue normal monitoring.",
            next_command=f"python -m fracture.cli status --pipeline-id {pipeline_id}",
            confidence=confidence_level,
        ))

    return RecommendationResult(
        pipeline_id=pipeline_id,
        status="ok",
        recommendations=_sort_recommendations(recommendations),
        metrics=metrics,
    )


def format_recommendation_result(result: RecommendationResult) -> str:
    """
    Format recommendations for CLI use.

    Keep this plain text so it can be pasted directly into reports.
    """
    lines = [
        f"Action-Oriented Process Mining: {result.pipeline_id}",
        f"Status: {result.status}",
    ]

    if result.metrics:
        lines.extend(["", "Metrics:"])
        for key, value in result.metrics.items():
            lines.append(f"  {key}: {value}")

    if result.recommendations:
        lines.extend(["", "Recommendations:"])
        for item in result.recommendations:
            lines.append(f"  - [{item['severity']}] {item['finding']}")
            lines.append(f"      owner: {item['owner']}")
            lines.append(f"      probable_cause: {item['probable_cause']}")
            lines.append(f"      recommended_action: {item['recommended_action']}")
            lines.append(f"      next_command: {item['next_command']}")
            lines.append(f"      confidence: {item['confidence']}")

    return "\n".join(lines)
