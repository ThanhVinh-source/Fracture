"""
tests/test_recommendation.py

Tests for Action-Oriented Process Mining.

These tests verify that Fracture can turn conformance/prediction signals into
owner-facing next actions.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from argparse import Namespace
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from fracture.cli import cmd_recommend
from fracture.prediction import predict_pipeline
from fracture.recommendation import (
    format_recommendation_result,
    recommend_pipeline,
)


passed = 0
failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"v {name}")
    except AssertionError as e:
        failed += 1
        print(f"x {name}")
        print(f"  {e}")
    except Exception as e:
        failed += 1
        print(f"x {name}: {type(e).__name__}: {e}")


def make_rows(
    scores,
    gaps=None,
    confidence="HIGH",
    timing_zone="GREEN",
    pattern="STABLE",
    pipeline_id="payment_batch",
    owner="pipeline-owner",
) -> pd.DataFrame:
    """
    Build minimal conformance_log-like rows.

    Recommendation reads the latest scored row and optional historical trend
    signals, so this fake data mirrors the columns written by fracture run.
    """
    start = datetime(2026, 6, 1)
    gaps = gaps if gaps is not None else [0.0] * len(scores)
    rows = []

    for index, score in enumerate(scores):
        rows.append({
            "pipeline_id": pipeline_id,
            "run_date": (start + timedelta(days=index)).strftime("%Y%m%d"),
            "final_score": score,
            "confidence_level": confidence,
            "timing_zone": timing_zone,
            "pattern": pattern,
            "bilateral_gap_minutes": gaps[index],
            "sequence_fitness": 1.0,
            "timing_score": 1.0,
            "completeness_score": 1.0,
            "alert_owner": owner,
        })

    return pd.DataFrame(rows)


def severities(result):
    """Return recommendation severities in display order."""
    return [item["severity"] for item in result.recommendations]


def test_recommendation_flags_severe_bilateral_gap():
    df = make_rows(
        scores=[1.0],
        gaps=[27.5],
    )

    result = recommend_pipeline(df, "payment_batch")

    assert result.status == "ok"
    assert result.recommendations[0]["severity"] == "URGENT"
    assert "Bilateral gap is severe" in result.recommendations[0]["finding"]
    assert "producer-consumer" in result.recommendations[0]["next_command"]


def test_recommendation_blocks_low_confidence_measurement():
    df = make_rows(
        scores=[0.99],
        gaps=[3.0],
        confidence="UNRELIABLE",
    )

    result = recommend_pipeline(df, "payment_batch")

    assert "BLOCKED" in severities(result)
    assert any("Measurement confidence" in item["finding"] for item in result.recommendations)


def test_recommendation_uses_prediction_gap_risk():
    df = make_rows(
        scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        gaps=[4.0, 7.0, 10.0, 13.0, 16.0],
    )
    prediction = predict_pipeline(df, "payment_batch")

    result = recommend_pipeline(
        conformance_df=df,
        pipeline_id="payment_batch",
        prediction_result=prediction,
    )

    assert result.status == "ok"
    assert "URGENT" in severities(result)
    assert any("projected" in item["finding"] for item in result.recommendations)


def test_recommendation_returns_info_when_healthy():
    df = make_rows(
        scores=[1.0],
        gaps=[2.0],
    )

    result = recommend_pipeline(df, "payment_batch")

    assert result.status == "ok"
    assert result.recommendations == [{
        "pipeline_id": "payment_batch",
        "severity": "INFO",
        "owner": "pipeline-owner",
        "finding": "No immediate action required.",
        "probable_cause": "Latest conformance, timing, and handoff metrics are within expected bounds.",
        "recommended_action": "Continue normal monitoring.",
        "next_command": "python -m fracture.cli status --pipeline-id payment_batch",
        "confidence": "HIGH",
    }]


def test_recommendation_handles_no_history():
    df = make_rows(
        scores=[1.0],
        gaps=[2.0],
        pipeline_id="other_pipeline",
    )

    result = recommend_pipeline(df, "payment_batch")

    assert result.status == "no_history"
    assert result.recommendations[0]["severity"] == "BLOCKED"
    assert "No conformance history" in result.recommendations[0]["finding"]


def test_format_recommendation_result_is_human_readable():
    df = make_rows(
        scores=[1.0],
        gaps=[27.5],
    )
    result = recommend_pipeline(df, "payment_batch")

    text = format_recommendation_result(result)

    assert "Action-Oriented Process Mining: payment_batch" in text
    assert "Recommendations:" in text
    assert "recommended_action" in text


def test_cmd_recommend_reads_conformance_log_csv():
    tmp = Path(tempfile.mkdtemp())
    try:
        log_path = tmp / "conformance_log.csv"
        make_rows(
            scores=[1.0],
            gaps=[27.5],
        ).to_csv(log_path, index=False)

        args = Namespace(
            key=None,
            pipeline_id="payment_batch",
            log_path=str(log_path),
            skip_prediction=True,
            min_score_points=5,
            min_gap_points=5,
        )

        output = StringIO()
        with redirect_stdout(output):
            exit_code = cmd_recommend(args)

        assert exit_code == 0
        assert "Action-Oriented Process Mining: payment_batch" in output.getvalue()
        assert "Bilateral gap is severe" in output.getvalue()

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print()
    print("Phase 5 action-oriented recommendation tests")
    print("=" * 50)

    check("recommendation flags severe bilateral gap",
          test_recommendation_flags_severe_bilateral_gap)
    check("recommendation blocks low confidence measurement",
          test_recommendation_blocks_low_confidence_measurement)
    check("recommendation uses prediction gap risk",
          test_recommendation_uses_prediction_gap_risk)
    check("recommendation returns info when healthy",
          test_recommendation_returns_info_when_healthy)
    check("recommendation handles no history",
          test_recommendation_handles_no_history)
    check("format recommendation result is human readable",
          test_format_recommendation_result_is_human_readable)
    check("cmd_recommend reads conformance_log.csv",
          test_cmd_recommend_reads_conformance_log_csv)

    print()
    print(f"Results: {passed + failed} tests  v {passed}  x {failed}")

    if failed:
        sys.exit(1)
