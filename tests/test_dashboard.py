"""
tests/test_dashboard.py

Smoke tests for the Phase 4 Streamlit dashboard.

These tests do not start a Streamlit server. They only verify that the
dashboard module imports and that helper functions handle missing files,
empty values, and basic dashboard data shapes safely.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

import dashboard


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


def test_dashboard_module_imports():
    # Importing dashboard should expose the expected Phase 4 entry points.
    assert callable(dashboard.main)
    assert callable(dashboard.render_fleet_overview)
    assert callable(dashboard.render_pipeline_detail)
    assert callable(dashboard.render_visualizations)


def test_resolve_data_root_prefers_runtime_log():
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "conformance_log.csv").write_text("pipeline_id,run_date\n", encoding="utf-8")
        (tmp / "demo_data").mkdir()
        (tmp / "demo_data" / "conformance_log.csv").write_text(
            "pipeline_id,run_date\n",
            encoding="utf-8",
        )

        # Local development should keep using the generated runtime files at
        # project root when they exist.
        assert dashboard.resolve_data_root(base_dir=tmp) == tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_data_root_falls_back_to_demo_data():
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "demo_data").mkdir()
        (tmp / "demo_data" / "conformance_log.csv").write_text(
            "pipeline_id,run_date\n",
            encoding="utf-8",
        )

        # Streamlit Cloud will not have generated root runtime files, so the
        # dashboard should automatically read the committed demo bundle.
        assert dashboard.resolve_data_root(base_dir=tmp) == tmp / "demo_data"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_data_root_uses_explicit_override():
    tmp = Path(tempfile.mkdtemp())
    try:
        explicit = tmp / "custom_data"

        # FRACTURE_DATA_ROOT lets tests and hosted deployments force a specific
        # data bundle without changing dashboard code.
        assert dashboard.resolve_data_root(
            explicit_root=str(explicit),
            base_dir=tmp,
        ) == explicit
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_format_score_handles_numbers_and_empty_values():
    # Dashboard scores come from CSV, so values may be numeric, strings, or empty.
    assert dashboard.format_score(0.9876) == "0.99"
    assert dashboard.format_score("1.0048") == "1.00"
    assert dashboard.format_score(None) == "n/a"
    assert dashboard.format_score("not-a-number") == "n/a"


def test_numeric_value_handles_missing_and_bad_values():
    row = pd.Series({
        "bilateral_gap_minutes": "14.2",
        "bad_value": "n/a",
    })

    # Numeric strings become floats for warnings and metric cards.
    assert dashboard.numeric_value(row, "bilateral_gap_minutes") == 14.2

    # Bad or missing values become None instead of crashing.
    assert dashboard.numeric_value(row, "bad_value") is None
    assert dashboard.numeric_value(row, "missing_column") is None


def test_get_latest_pipeline_row_uses_latest_run_date():
    df = pd.DataFrame({
        "pipeline_id": ["payment_batch", "payment_batch", "other_pipeline"],
        "run_date": ["20260617", "20260618", "20260618"],
        "final_score": [0.80, 0.95, 0.50],
    })

    latest = dashboard.get_latest_pipeline_row(df, "payment_batch")

    # The dashboard should summarize the newest measured row for the selected pipeline.
    assert latest["run_date"] == "20260618"
    assert latest["final_score"] == 0.95


def test_build_pipeline_takeaway_flags_high_gap_first():
    row = pd.Series({
        "final_score": "0.93",
        "timing_zone": "AMBER",
        "confidence_level": "HIGH",
        "bilateral_gap_minutes": "33.6",
    })

    kind, message = dashboard.build_pipeline_takeaway(row)

    # A severe producer-consumer gap is the most important demo finding.
    assert kind == "error"
    assert "handoff delay" in message


def test_build_pipeline_takeaway_marks_healthy_pipeline_success():
    row = pd.Series({
        "final_score": "0.96",
        "timing_zone": "GREEN",
        "confidence_level": "HIGH",
        "bilateral_gap_minutes": "4.0",
    })

    kind, message = dashboard.build_pipeline_takeaway(row)

    assert kind == "success"
    assert "broadly conformant" in message


def test_sorted_filter_options_drops_empty_values():
    df = pd.DataFrame({
        "timing_zone": ["GREEN", "AMBER", "GREEN", "", None],
    })

    assert dashboard.sorted_filter_options(df, "timing_zone") == [
        "AMBER",
        "GREEN",
    ]
    assert dashboard.sorted_filter_options(df, "missing") == []


def test_get_pipeline_options_prefers_conformance_history():
    df = pd.DataFrame({
        "pipeline_id": ["z_pipeline", "a_pipeline", "z_pipeline"],
    })

    # conformance_log.csv history is the preferred pipeline selector source.
    # Pipelines with more history come first so dashboard trend views are useful
    # immediately when the app opens.
    assert dashboard.get_pipeline_options(df) == [
        "z_pipeline",
        "a_pipeline",
    ]


def test_get_pipeline_options_falls_back_to_visualization_folders():
    tmp = Path(tempfile.mkdtemp())
    try:
        original_root = dashboard.VISUALIZATION_ROOT

        # Point the dashboard helper at an isolated fake output directory.
        dashboard.VISUALIZATION_ROOT = tmp / "outputs" / "visualizations"
        (dashboard.VISUALIZATION_ROOT / "beta_pipeline").mkdir(parents=True)
        (dashboard.VISUALIZATION_ROOT / "alpha_pipeline").mkdir(parents=True)
        (dashboard.VISUALIZATION_ROOT / "fleet_heatmap.png").write_text("fake")

        assert dashboard.get_pipeline_options(pd.DataFrame()) == [
            "alpha_pipeline",
            "beta_pipeline",
        ]
    finally:
        dashboard.VISUALIZATION_ROOT = original_root
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_pipeline_contract_missing_file_is_safe():
    tmp = Path(tempfile.mkdtemp())
    try:
        original_contracts_root = dashboard.CONTRACTS_ROOT

        # Missing contracts should produce a clear status, not an exception.
        dashboard.CONTRACTS_ROOT = tmp / "contracts"
        contract, status = dashboard.load_pipeline_contract("missing_pipeline")

        assert contract is None
        assert "missing contract" in status
    finally:
        dashboard.CONTRACTS_ROOT = original_contracts_root
        shutil.rmtree(tmp, ignore_errors=True)


def test_build_event_visual_scope_tracks_single_date():
    first_scope = dashboard.build_event_visual_scope(
        pipeline_id="payment_batch",
        date_str="20260616",
        available_dates=["20260615", "20260616"],
        use_all_event_dates=False,
    )
    second_scope = dashboard.build_event_visual_scope(
        pipeline_id="payment_batch",
        date_str="20260617",
        available_dates=["20260615", "20260616"],
        use_all_event_dates=False,
    )

    # Single-date mode must change scope when the user edits Input date.
    assert first_scope != second_scope
    assert first_scope == ("payment_batch", "single", ("20260616",))


def test_build_event_visual_scope_tracks_all_dates():
    scope = dashboard.build_event_visual_scope(
        pipeline_id="payment_batch",
        date_str="20260616",
        available_dates=["20260615", "20260616"],
        use_all_event_dates=True,
    )

    # All-dates mode ignores the text field and follows the available file set.
    assert scope == ("payment_batch", "all", ("20260615", "20260616"))


def test_describe_event_scope_availability_flags_missing_date():
    available, message = dashboard.describe_event_scope_availability(
        date_str="20260619",
        available_dates=["20260617", "20260618"],
        use_all_event_dates=False,
    )

    # A typed date with no producer_YYYYMMDD file must not reuse old PNG output.
    assert available is False
    assert "Unavailable" in message
    assert "20260619" in message


def test_describe_event_scope_availability_all_dates_is_available():
    available, message = dashboard.describe_event_scope_availability(
        date_str="20260619",
        available_dates=["20260617", "20260618"],
        use_all_event_dates=True,
    )

    # All-dates mode follows the discovered input range, so the text field is
    # not used as a single required file date.
    assert available is True
    assert "full available input range" in message


def test_load_json_artifact_missing_file_is_safe():
    tmp = Path(tempfile.mkdtemp())
    try:
        payload, status = dashboard.load_json_artifact(
            tmp / "outputs" / "prediction.json"
        )

        # Missing prediction/recommendation artifacts should show a dashboard
        # hint instead of crashing the Streamlit page.
        assert payload == {}
        assert "missing artifact" in status
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_json_artifact_reads_valid_json():
    tmp = Path(tempfile.mkdtemp())
    try:
        artifact_path = tmp / "recommendations.json"
        artifact_path.write_text(
            json.dumps({
                "pipeline_id": "payment_batch",
                "status": "ok",
                "recommendations": [],
            }),
            encoding="utf-8",
        )

        payload, status = dashboard.load_json_artifact(artifact_path)

        # Dashboard JSON loader should preserve structured fields for cards
        # and tables in the Prediction & Actions tab.
        assert status == "ok"
        assert payload["pipeline_id"] == "payment_batch"
        assert payload["status"] == "ok"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_save_json_artifact_writes_pipeline_file():
    tmp = Path(tempfile.mkdtemp())
    try:
        original_root = dashboard.VISUALIZATION_ROOT

        # Point artifact writing at a temporary output root so the test does
        # not touch real demo files in outputs/visualizations.
        dashboard.VISUALIZATION_ROOT = tmp / "outputs" / "visualizations"
        output_path = dashboard.save_json_artifact(
            {"pipeline_id": "payment_batch", "status": "ok"},
            pipeline_id="payment_batch",
            filename="prediction.json",
        )

        payload = json.loads(output_path.read_text(encoding="utf-8"))

        assert output_path.exists()
        assert output_path.name == "prediction.json"
        assert payload["pipeline_id"] == "payment_batch"
        assert payload["status"] == "ok"
        assert "generated_at" in payload
    finally:
        dashboard.VISUALIZATION_ROOT = original_root
        shutil.rmtree(tmp, ignore_errors=True)


def test_dashboard_cli_help_works():
    # The CLI route should exist without starting a Streamlit server.
    result = subprocess.run(
        [sys.executable, "-m", "fracture.cli", "dashboard", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage: fracture dashboard" in result.stdout
    assert "--port" in result.stdout


if __name__ == "__main__":
    print()
    print("Phase 4 dashboard smoke tests")
    print("=" * 40)

    check("dashboard module imports", test_dashboard_module_imports)
    check("resolve_data_root prefers runtime log",
          test_resolve_data_root_prefers_runtime_log)
    check("resolve_data_root falls back to demo_data",
          test_resolve_data_root_falls_back_to_demo_data)
    check("resolve_data_root uses explicit override",
          test_resolve_data_root_uses_explicit_override)
    check("format_score handles numbers and empty values",
          test_format_score_handles_numbers_and_empty_values)
    check("numeric_value handles missing and bad values",
          test_numeric_value_handles_missing_and_bad_values)
    check("get_latest_pipeline_row uses latest run date",
          test_get_latest_pipeline_row_uses_latest_run_date)
    check("build_pipeline_takeaway flags high gap first",
          test_build_pipeline_takeaway_flags_high_gap_first)
    check("build_pipeline_takeaway marks healthy pipeline success",
          test_build_pipeline_takeaway_marks_healthy_pipeline_success)
    check("sorted_filter_options drops empty values",
          test_sorted_filter_options_drops_empty_values)
    check("get_pipeline_options prefers conformance history",
          test_get_pipeline_options_prefers_conformance_history)
    check("get_pipeline_options falls back to visualization folders",
          test_get_pipeline_options_falls_back_to_visualization_folders)
    check("load_pipeline_contract missing file is safe",
          test_load_pipeline_contract_missing_file_is_safe)
    check("build_event_visual_scope tracks single date",
          test_build_event_visual_scope_tracks_single_date)
    check("build_event_visual_scope tracks all dates",
          test_build_event_visual_scope_tracks_all_dates)
    check("describe_event_scope_availability flags missing date",
          test_describe_event_scope_availability_flags_missing_date)
    check("describe_event_scope_availability all-dates is available",
          test_describe_event_scope_availability_all_dates_is_available)
    check("load_json_artifact missing file is safe",
          test_load_json_artifact_missing_file_is_safe)
    check("load_json_artifact reads valid json",
          test_load_json_artifact_reads_valid_json)
    check("save_json_artifact writes pipeline file",
          test_save_json_artifact_writes_pipeline_file)
    check("dashboard CLI help works", test_dashboard_cli_help_works)

    print()
    print(f"Results: {passed + failed} tests  v {passed}  x {failed}")

    if failed:
        raise SystemExit(1)
