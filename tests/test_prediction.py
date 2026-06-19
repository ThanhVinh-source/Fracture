"""
tests/test_prediction.py

Tests for Predictive Process Mining.

These tests use small fake conformance_log rows because prediction needs a
multi-day history. The production command reads the real conformance_log.csv.
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

from fracture.cli import cmd_predict
from fracture.prediction import (
    format_prediction_result,
    predict_pipeline,
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


def make_history(
    scores,
    gaps=None,
    confidence="HIGH",
    pipeline_id="payment_batch",
) -> pd.DataFrame:
    """
    Build minimal conformance_log-like rows for one pipeline.

    Prediction needs run_date + final_score. bilateral_gap_minutes and
    confidence_level are optional but used by gap prediction and skip rules.
    """
    start = datetime(2026, 6, 1)
    gaps = gaps if gaps is not None else [None] * len(scores)
    rows = []

    for index, score in enumerate(scores):
        rows.append({
            "pipeline_id": pipeline_id,
            "run_date": (start + timedelta(days=index)).strftime("%Y%m%d"),
            "final_score": score,
            "bilateral_gap_minutes": gaps[index],
            "confidence_level": confidence,
        })

    return pd.DataFrame(rows)


def find_prediction(result, prediction_type):
    """Return one prediction record by type for concise assertions."""
    for prediction in result.predictions:
        if prediction["prediction_type"] == prediction_type:
            return prediction
    raise AssertionError(f"Missing prediction_type={prediction_type}")


def test_score_decline_projects_warning_and_critical_dates():
    df = make_history(
        scores=[1.00, 0.97, 0.94, 0.91, 0.88],
    )

    result = predict_pipeline(df, "payment_batch")

    assert result.status == "ok"

    warning = find_prediction(result, "score_warning")
    critical = find_prediction(result, "score_critical")

    assert warning["projected_date"] == "2026-06-06"
    assert warning["days_until_projection"] == 1
    assert critical["days_until_projection"] == 6
    assert warning["confidence"] == "MEDIUM"
    assert any("Score is declining" in finding for finding in result.findings)


def test_gap_widening_projects_severe_threshold():
    df = make_history(
        scores=[1.00, 1.00, 1.00, 1.00, 1.00],
        gaps=[4.0, 7.0, 10.0, 13.0, 16.0],
    )

    result = predict_pipeline(df, "payment_batch")

    assert result.status == "ok"

    severe = find_prediction(result, "gap_severe")

    assert severe["projected_date"] == "2026-06-07"
    assert severe["days_until_projection"] == 2
    assert severe["slope_per_day"] == 3.0
    assert any("Bilateral gap is widening" in finding for finding in result.findings)


def test_prediction_skips_when_history_is_insufficient():
    df = make_history(
        scores=[1.0, 0.98, 0.96, 0.94],
        gaps=[4.0, 5.0, 6.0, 7.0],
    )

    result = predict_pipeline(df, "payment_batch")

    assert result.status == "insufficient_history"
    assert result.predictions == []
    assert any("Score prediction skipped" in finding for finding in result.findings)
    assert any("Gap prediction skipped" in finding for finding in result.findings)


def test_prediction_skips_low_conformance_confidence():
    df = make_history(
        scores=[1.0, 0.97, 0.94, 0.91, 0.88],
        confidence="LOW",
    )

    result = predict_pipeline(df, "payment_batch")

    assert result.status == "low_confidence"
    assert result.predictions == []
    assert "LOW" in result.findings[0]


def test_format_prediction_result_is_human_readable():
    df = make_history(
        scores=[1.00, 0.97, 0.94, 0.91, 0.88],
    )
    result = predict_pipeline(df, "payment_batch")

    text = format_prediction_result(result)

    assert "Predictive Process Mining: payment_batch" in text
    assert "Predictions:" in text
    assert "score_warning" in text


def test_cmd_predict_reads_conformance_log_csv():
    tmp = Path(tempfile.mkdtemp())
    try:
        log_path = tmp / "conformance_log.csv"
        make_history(
            scores=[1.00, 0.97, 0.94, 0.91, 0.88],
        ).to_csv(log_path, index=False)

        args = Namespace(
            key=None,
            pipeline_id="payment_batch",
            log_path=str(log_path),
            min_score_points=5,
            min_gap_points=5,
        )

        output = StringIO()
        with redirect_stdout(output):
            exit_code = cmd_predict(args)

        assert exit_code == 0
        assert "Predictive Process Mining: payment_batch" in output.getvalue()
        assert "score_warning" in output.getvalue()

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print()
    print("Phase 5 predictive process mining tests")
    print("=" * 50)

    check("score decline projects warning and critical dates",
          test_score_decline_projects_warning_and_critical_dates)
    check("gap widening projects severe threshold",
          test_gap_widening_projects_severe_threshold)
    check("prediction skips when history is insufficient",
          test_prediction_skips_when_history_is_insufficient)
    check("prediction skips low conformance confidence",
          test_prediction_skips_low_conformance_confidence)
    check("format prediction result is human readable",
          test_format_prediction_result_is_human_readable)
    check("cmd_predict reads conformance_log.csv",
          test_cmd_predict_reads_conformance_log_csv)

    print()
    print(f"Results: {passed + failed} tests  v {passed}  x {failed}")

    if failed:
        sys.exit(1)
