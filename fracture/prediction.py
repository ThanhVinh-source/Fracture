"""
fracture/prediction.py

Phase 5 predictive process-mining helpers.

This module answers:
"Is this pipeline drifting toward a future SLA/conformance problem?"

V1 intentionally uses deterministic trend rules instead of ML. That keeps the
prediction explainable for a course/demo setting and avoids pretending that a
small local CSV contains enough data for a reliable model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import ceil

import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    """
    Structured prediction result for CLI, dashboard, tests, and reports.

    `predictions` is a list because one pipeline can have separate score and
    bilateral-gap forecasts.
    """

    pipeline_id: str
    status: str
    metrics: dict
    predictions: list[dict]
    findings: list[str]
    details: dict

    def as_dict(self) -> dict:
        """Return plain Python values for JSON/report/dashboard use."""
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status,
            "metrics": self.metrics,
            "predictions": self.predictions,
            "findings": self.findings,
            "details": self.details,
        }


def _empty_result(
    pipeline_id: str,
    status: str,
    finding: str,
    metrics: dict | None = None,
) -> PredictionResult:
    """Build a small blocked/empty prediction result."""
    return PredictionResult(
        pipeline_id=pipeline_id,
        status=status,
        metrics=metrics or {},
        predictions=[],
        findings=[finding],
        details={},
    )


def _parse_run_dates(values: pd.Series) -> pd.Series:
    """
    Parse conformance_log run_date values safely.

    Fracture usually stores dates as YYYYMMDD, but pandas may read them as
    integers or as full timestamps depending on how the CSV was created.
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


def _trend_fit(history: pd.DataFrame, value_col: str) -> dict:
    """
    Fit a simple linear trend over calendar days.

    The slope is expressed in real units per day:
    - final_score points per day for score prediction
    - minutes per day for bilateral-gap prediction
    """
    clean = (
        history[["run_date", value_col]]
        .dropna()
        .sort_values("run_date")
        .reset_index(drop=True)
    )

    if len(clean) < 2:
        latest_date = clean["run_date"].iloc[-1] if len(clean) == 1 else None
        latest_value = float(clean[value_col].iloc[-1]) if len(clean) == 1 else None
        return {
            "n": int(len(clean)),
            "slope_per_day": 0.0,
            "intercept": None,
            "r2": 0.0,
            "latest_date": latest_date,
            "latest_value": latest_value,
        }

    # Use days since the first observed run so the slope is easy to interpret.
    x = (clean["run_date"] - clean["run_date"].iloc[0]).dt.days.astype(float)

    # If all rows have the same date, fall back to chronological row index.
    # This preserves test/demo behavior without pretending duplicate-date rows
    # are a real calendar trend.
    if float(x.max()) == 0.0:
        x = pd.Series(np.arange(len(clean)), dtype=float)

    y = clean[value_col].astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(((y - fitted) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 if total == 0 else max(0.0, 1.0 - residual / total)

    return {
        "n": int(len(clean)),
        "slope_per_day": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "latest_date": clean["run_date"].iloc[-1],
        "latest_value": float(y.iloc[-1]),
    }


def _prediction_confidence(n_points: int, r2: float) -> str:
    """
    Convert trend quality into a readable confidence label.

    This is prediction confidence, not the conformance confidence_level. It is
    based on amount of history and how well a straight line explains the trend.
    """
    if n_points >= 10 and r2 >= 0.50:
        return "HIGH"
    if n_points >= 5 and r2 >= 0.20:
        return "MEDIUM"
    return "LOW"


def _urgency(days_until_projection: int | None) -> str:
    """Classify how soon a projected risk date lands."""
    if days_until_projection is None:
        return "none"
    if days_until_projection <= 14:
        return "urgent"
    if days_until_projection <= 30:
        return "action"
    if days_until_projection <= 90:
        return "watch"
    return "low_urgency"


def _project_date(latest_date, latest_value: float, slope: float, threshold: float, direction: str) -> tuple[str | None, int | None]:
    """
    Project when a value crosses a threshold.

    direction="down" is used for final_score deterioration.
    direction="up" is used for widening bilateral gaps.
    """
    if latest_date is None:
        return None, None

    if direction == "down":
        if latest_value <= threshold:
            days = 0
        elif slope >= 0:
            return None, None
        else:
            # Floating point arithmetic can turn an exact 1-day crossing into
            # 1.00000000002. Subtract a tiny epsilon before ceil so projection
            # dates do not drift by one day.
            days = ceil(((threshold - latest_value) / slope) - 1e-9)
    else:
        if latest_value >= threshold:
            days = 0
        elif slope <= 0:
            return None, None
        else:
            # Same epsilon protection for widening-gap projections.
            days = ceil(((threshold - latest_value) / slope) - 1e-9)

    if days < 0:
        return None, None

    projected = latest_date + timedelta(days=int(days))
    return str(projected.date()), int(days)


def _prediction_record(
    prediction_type: str,
    projected_date: str | None,
    days_until_projection: int | None,
    slope_per_day: float,
    confidence: str,
    explanation: str,
    threshold: float | None = None,
) -> dict:
    """Create one prediction row using the public output schema."""
    record = {
        "prediction_type": prediction_type,
        "projected_date": projected_date,
        "days_until_projection": days_until_projection,
        "slope_per_day": round(float(slope_per_day), 6),
        "confidence": confidence,
        "explanation": explanation,
    }

    if threshold is not None:
        record["threshold"] = threshold

    record["urgency"] = _urgency(days_until_projection)
    return record


def predict_pipeline(
    conformance_df: pd.DataFrame,
    pipeline_id: str,
    min_score_points: int = 5,
    min_gap_points: int = 5,
    healthy_threshold: float = 0.85,
    critical_threshold: float = 0.70,
    gap_warning_threshold: float = 10.0,
    gap_severe_threshold: float = 20.0,
    min_score_slope: float = 0.001,
    min_gap_slope: float = 0.05,
    min_r2: float = 0.20,
) -> PredictionResult:
    """
    Predict score deterioration and bilateral-gap widening from history.

    The function reads persisted conformance history rather than raw event logs
    because prediction needs repeated scored runs over time.
    """
    required = {"pipeline_id", "run_date", "final_score"}
    missing = required - set(conformance_df.columns)
    if missing:
        return _empty_result(
            pipeline_id,
            "missing_columns",
            f"conformance_log.csv is missing required columns: {sorted(missing)}",
        )

    history = conformance_df[conformance_df["pipeline_id"] == pipeline_id].copy()
    if history.empty:
        return _empty_result(
            pipeline_id,
            "no_history",
            "No conformance history exists for this pipeline.",
        )

    history["run_date"] = _parse_run_dates(history["run_date"])
    history["final_score"] = pd.to_numeric(history["final_score"], errors="coerce")

    if "bilateral_gap_minutes" in history.columns:
        history["bilateral_gap_minutes"] = pd.to_numeric(
            history["bilateral_gap_minutes"],
            errors="coerce",
        )
    else:
        history["bilateral_gap_minutes"] = np.nan

    if "confidence_level" in history.columns:
        history["confidence_level"] = history["confidence_level"].astype(str).str.upper()
    else:
        history["confidence_level"] = ""

    history = history.dropna(subset=["run_date", "final_score"])
    history = history.sort_values("run_date").reset_index(drop=True)

    if history.empty:
        return _empty_result(
            pipeline_id,
            "no_valid_history",
            "No valid dated score rows exist for this pipeline.",
        )

    latest_confidence = str(history["confidence_level"].iloc[-1]).upper()
    if latest_confidence in {"LOW", "UNRELIABLE"}:
        return _empty_result(
            pipeline_id,
            "low_confidence",
            f"Latest conformance confidence is {latest_confidence}; prediction is skipped.",
            metrics={
                "available_score_points": int(history["final_score"].notna().sum()),
                "latest_confidence_level": latest_confidence,
            },
        )

    predictions = []
    findings = []
    score_fit = _trend_fit(history, "final_score")
    gap_history = history.dropna(subset=["bilateral_gap_minutes"])
    gap_fit = _trend_fit(gap_history, "bilateral_gap_minutes")

    metrics = {
        "available_score_points": int(score_fit["n"]),
        "required_score_points": int(min_score_points),
        "available_gap_points": int(gap_fit["n"]),
        "required_gap_points": int(min_gap_points),
        "latest_score": round(float(score_fit["latest_value"]), 4)
        if score_fit["latest_value"] is not None else None,
        "latest_gap_minutes": round(float(gap_fit["latest_value"]), 4)
        if gap_fit["latest_value"] is not None else None,
        "latest_confidence_level": latest_confidence or "UNKNOWN",
    }

    if score_fit["n"] < min_score_points:
        findings.append(
            f"Score prediction skipped: need at least {min_score_points} scored rows."
        )
    else:
        score_confidence = _prediction_confidence(score_fit["n"], score_fit["r2"])
        score_slope = score_fit["slope_per_day"]
        score_meaningful = abs(score_slope) >= min_score_slope and score_fit["r2"] >= min_r2

        metrics["score_slope_per_day"] = round(float(score_slope), 6)
        metrics["score_trend_r2"] = round(float(score_fit["r2"]), 4)

        if not score_meaningful:
            predictions.append(_prediction_record(
                "score_trend",
                None,
                None,
                score_slope,
                score_confidence,
                "Score trend is not strong enough to project a breach.",
            ))
            findings.append("Score trend is currently not meaningful enough for projection.")
        elif score_slope >= 0:
            predictions.append(_prediction_record(
                "score_trend",
                None,
                None,
                score_slope,
                score_confidence,
                "Score is flat or improving; no score breach projected.",
            ))
            findings.append("Score is flat or improving; no score breach projected.")
        else:
            warning_date, warning_days = _project_date(
                score_fit["latest_date"],
                score_fit["latest_value"],
                score_slope,
                healthy_threshold,
                "down",
            )
            critical_date, critical_days = _project_date(
                score_fit["latest_date"],
                score_fit["latest_value"],
                score_slope,
                critical_threshold,
                "down",
            )

            predictions.append(_prediction_record(
                "score_warning",
                warning_date,
                warning_days,
                score_slope,
                score_confidence,
                f"Projected date when final_score falls below {healthy_threshold:.2f}.",
                threshold=healthy_threshold,
            ))
            predictions.append(_prediction_record(
                "score_critical",
                critical_date,
                critical_days,
                score_slope,
                score_confidence,
                f"Projected date when final_score falls below {critical_threshold:.2f}.",
                threshold=critical_threshold,
            ))

            if critical_days is None:
                findings.append("Score is declining, but no critical date could be projected.")
            elif critical_days > 90:
                findings.append(
                    f"Score is declining, but critical projection is low urgency at {critical_days} days away."
                )
            else:
                findings.append(
                    f"Score is declining; critical threshold is projected in {critical_days} days."
                )

    if gap_fit["n"] < min_gap_points:
        findings.append(
            f"Gap prediction skipped: need at least {min_gap_points} bilateral-gap rows."
        )
    else:
        gap_confidence = _prediction_confidence(gap_fit["n"], gap_fit["r2"])
        gap_slope = gap_fit["slope_per_day"]
        gap_meaningful = abs(gap_slope) >= min_gap_slope and gap_fit["r2"] >= min_r2

        metrics["gap_slope_minutes_per_day"] = round(float(gap_slope), 6)
        metrics["gap_trend_r2"] = round(float(gap_fit["r2"]), 4)

        if not gap_meaningful:
            predictions.append(_prediction_record(
                "gap_trend",
                None,
                None,
                gap_slope,
                gap_confidence,
                "Bilateral gap trend is not strong enough to project widening risk.",
            ))
            findings.append("Bilateral gap trend is not meaningful enough for projection.")
        elif gap_slope <= 0:
            predictions.append(_prediction_record(
                "gap_trend",
                None,
                None,
                gap_slope,
                gap_confidence,
                "Bilateral gap is flat or narrowing; no widening breach projected.",
            ))
            findings.append("Bilateral gap is flat or narrowing.")
        else:
            warning_date, warning_days = _project_date(
                gap_fit["latest_date"],
                gap_fit["latest_value"],
                gap_slope,
                gap_warning_threshold,
                "up",
            )
            severe_date, severe_days = _project_date(
                gap_fit["latest_date"],
                gap_fit["latest_value"],
                gap_slope,
                gap_severe_threshold,
                "up",
            )

            predictions.append(_prediction_record(
                "gap_warning",
                warning_date,
                warning_days,
                gap_slope,
                gap_confidence,
                f"Projected date when bilateral gap reaches {gap_warning_threshold:.1f} minutes.",
                threshold=gap_warning_threshold,
            ))
            predictions.append(_prediction_record(
                "gap_severe",
                severe_date,
                severe_days,
                gap_slope,
                gap_confidence,
                f"Projected date when bilateral gap reaches {gap_severe_threshold:.1f} minutes.",
                threshold=gap_severe_threshold,
            ))

            if severe_days == 0:
                findings.append("Bilateral gap is already in the severe zone.")
            elif severe_days is not None and severe_days <= 90:
                findings.append(
                    f"Bilateral gap is widening; severe threshold is projected in {severe_days} days."
                )
            elif severe_days is not None:
                findings.append(
                    f"Bilateral gap is widening, but severe threshold is low urgency at {severe_days} days away."
                )

    if not predictions:
        return PredictionResult(
            pipeline_id=pipeline_id,
            status="insufficient_history",
            metrics=metrics,
            predictions=[],
            findings=findings or ["Not enough history to produce any prediction."],
            details={
                "history_rows": history.to_dict(orient="records"),
            },
        )

    return PredictionResult(
        pipeline_id=pipeline_id,
        status="ok",
        metrics=metrics,
        predictions=predictions,
        findings=findings,
        details={
            "history_rows": history.to_dict(orient="records"),
        },
    )


def format_prediction_result(result: PredictionResult) -> str:
    """
    Format prediction output for CLI use.

    Keep this plain text so it is easy to paste into a report or terminal note.
    """
    lines = [
        f"Predictive Process Mining: {result.pipeline_id}",
        f"Status: {result.status}",
    ]

    if result.metrics:
        lines.extend(["", "Metrics:"])
        for key, value in result.metrics.items():
            lines.append(f"  {key}: {value}")

    if result.predictions:
        lines.extend(["", "Predictions:"])
        for prediction in result.predictions:
            lines.append(f"  - {prediction['prediction_type']}")
            lines.append(f"      projected_date: {prediction['projected_date']}")
            lines.append(f"      days_until_projection: {prediction['days_until_projection']}")
            lines.append(f"      slope_per_day: {prediction['slope_per_day']}")
            lines.append(f"      confidence: {prediction['confidence']}")
            lines.append(f"      urgency: {prediction['urgency']}")
            if "threshold" in prediction:
                lines.append(f"      threshold: {prediction['threshold']}")
            lines.append(f"      explanation: {prediction['explanation']}")

    if result.findings:
        lines.extend(["", "Findings:"])
        lines.extend([f"  - {finding}" for finding in result.findings])

    return "\n".join(lines)
