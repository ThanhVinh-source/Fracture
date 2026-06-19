"""
scripts/09_run_final_demo.py

Run the full Fracture demo from one command.

This script intentionally calls the public CLI commands instead of importing
private internals. That keeps the final demo aligned with how a real user would
run the project from VSCode or a terminal.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(label: str, command: list[str]) -> None:
    """
    Run one demo command and stop immediately if it fails.

    A final demo should fail loudly. Continuing after a broken setup command
    would make later output confusing and harder to debug.
    """
    print()
    print(f"== {label}")
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full Fracture final demo flow.",
    )
    parser.add_argument(
        "--pipeline-id",
        default="trade_positions_sftp",
        help="Pipeline used for single-pipeline demo commands.",
    )
    parser.add_argument(
        "--start-date",
        default="20260519",
        help="Inclusive backfill start date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--end-date",
        default="20260618",
        help="Inclusive backfill/demo date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip contract/input generation and conformance backfill.",
    )
    parser.add_argument(
        "--with-dashboard",
        action="store_true",
        help="Open Streamlit dashboard at the end. This keeps running.",
    )
    parser.add_argument(
        "--dashboard-port",
        default="8501",
        help="Dashboard port when --with-dashboard is used.",
    )
    args = parser.parse_args()

    py = sys.executable
    pid = args.pipeline_id
    demo_date = args.end_date

    if not args.skip_setup:
        run_step(
            "Generate demo contracts and inputs",
            [py, "scripts/06_generate_team_contracts.py", "--clean", "--days", "30"],
        )
        run_step(
            "Backfill conformance history",
            [
                py,
                "scripts/08_backfill_conformance_history.py",
                "--start-date",
                args.start_date,
                "--end-date",
                args.end_date,
            ],
        )

    run_step(
        "Latest pipeline status",
        [py, "-m", "fracture.cli", "status", "--pipeline-id", pid],
    )
    run_step(
        "Process discovery",
        [py, "-m", "fracture.cli", "discover", "--pipeline-id", pid, "--date", demo_date],
    )
    run_step(
        "Producer-consumer comparison",
        [
            py,
            "-m",
            "fracture.cli",
            "compare",
            "--pipeline-id",
            pid,
            "--mode",
            "producer-consumer",
            "--date",
            demo_date,
        ],
    )
    run_step(
        "Contract-actual comparison",
        [
            py,
            "-m",
            "fracture.cli",
            "compare",
            "--pipeline-id",
            pid,
            "--mode",
            "contract-actual",
            "--date",
            demo_date,
        ],
    )
    run_step(
        "Period comparison",
        [py, "-m", "fracture.cli", "compare", "--pipeline-id", pid, "--mode", "period"],
    )
    run_step(
        "Performance mining",
        [py, "-m", "fracture.cli", "performance", "--pipeline-id", pid, "--date", demo_date],
    )
    run_step(
        "Predictive mining",
        [py, "-m", "fracture.cli", "predict", "--pipeline-id", pid],
    )
    run_step(
        "Action recommendations",
        [py, "-m", "fracture.cli", "recommend", "--pipeline-id", pid],
    )
    run_step(
        "Static visualization export",
        [
            py,
            "-m",
            "fracture.cli",
            "visualize",
            "--pipeline-id",
            pid,
            "--kind",
            "all",
            "--all-dates",
        ],
    )

    if args.with_dashboard:
        run_step(
            "Open dashboard",
            [py, "-m", "fracture.cli", "dashboard", "--port", args.dashboard_port],
        )
    else:
        print()
        print("Demo complete.")
        print("Open the dashboard with:")
        print(f"  {py} -m fracture.cli dashboard")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

