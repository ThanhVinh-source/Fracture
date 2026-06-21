"""
scripts/10_prepare_streamlit_demo_data.py

Create a small, committed demo data bundle for Streamlit Cloud.

Local Fracture runs write runtime files at the project root:
- conformance_log.csv
- contracts/
- inputs/

Those paths are intentionally ignored by git. Streamlit Cloud therefore needs a
tracked snapshot under demo_data/ so the dashboard can open with meaningful data
immediately after deployment.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_PIPELINE = "trade_positions_sftp"


def discover_input_pipelines(inputs_dir: Path) -> list[str]:
    """
    Return pipeline folders that have runtime input files.

    The dashboard can render event-log visuals for any pipeline with producer
    and consumer files under inputs/{pipeline_id}/.
    """
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Required inputs directory is missing: {inputs_dir}")

    pipeline_ids = sorted(path.name for path in inputs_dir.iterdir() if path.is_dir())
    if not pipeline_ids:
        raise FileNotFoundError(f"No pipeline input folders found in {inputs_dir}")

    return pipeline_ids


def copy_file_if_exists(source: Path, target: Path, required: bool = False) -> bool:
    """
    Copy one runtime file into demo_data.

    Required files fail the script when missing. Optional files are skipped with
    a visible message because not every demo needs clustering or registry data.
    """
    if not source.exists():
        if required:
            raise FileNotFoundError(f"Required demo source is missing: {source}")

        print(f"! skipped optional file: {source.name}")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"v copied {source.relative_to(ROOT)} -> {target.relative_to(ROOT)}")
    return True


def copy_contracts(source_dir: Path, target_dir: Path) -> int:
    """
    Copy all contract YAML files.

    Fleet and Pipeline Detail pages need contract metadata for many pipelines,
    so this keeps the cloud dashboard useful beyond one selected pipeline.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Required contracts directory is missing: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for contract_path in sorted(source_dir.glob("*.yaml")):
        shutil.copy2(contract_path, target_dir / contract_path.name)
        copied += 1

    if copied == 0:
        raise FileNotFoundError(f"No contract YAML files found in {source_dir}")

    print(f"v copied {copied} contract YAML files")
    return copied


def copy_pipeline_inputs(source_dir: Path, target_dir: Path, pipeline_id: str) -> int:
    """
    Copy producer/consumer input files for one pipeline.

    Event-log visualizations such as bilateral gap, DFG, and performance need
    raw producer_YYYYMMDD and consumer_YYYYMMDD files. The demo bundle keeps
    these scoped to selected pipelines so the repository stays lightweight.
    """
    pipeline_source = source_dir / pipeline_id
    pipeline_target = target_dir / pipeline_id

    if not pipeline_source.exists():
        print(f"! skipped inputs for {pipeline_id}: missing {pipeline_source}")
        return 0

    pipeline_target.mkdir(parents=True, exist_ok=True)
    copied = 0

    for pattern in ("producer_*.parquet", "consumer_*.parquet", "producer_*.csv", "consumer_*.csv"):
        for event_file in sorted(pipeline_source.glob(pattern)):
            shutil.copy2(event_file, pipeline_target / event_file.name)
            copied += 1

    if copied == 0:
        print(f"! skipped inputs for {pipeline_id}: no producer/consumer files found")
    else:
        print(f"v copied {copied} input files for {pipeline_id}")

    return copied


def prepare_demo_data(
    output_dir: Path,
    pipeline_ids: list[str],
    clean: bool = False,
) -> Path:
    """
    Build the demo_data folder from the current local runtime state.

    The generated folder is safe to commit. It intentionally excludes generated
    PNG/JSON outputs because the dashboard can regenerate those artifacts on
    Streamlit Cloud.
    """
    output_dir = output_dir.resolve()

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"v removed existing {output_dir.relative_to(ROOT)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    copy_file_if_exists(
        ROOT / "conformance_log.csv",
        output_dir / "conformance_log.csv",
        required=True,
    )
    copy_file_if_exists(
        ROOT / "cluster_assignments.csv",
        output_dir / "cluster_assignments.csv",
        required=False,
    )
    copy_file_if_exists(
        ROOT / "pipeline_registry.csv",
        output_dir / "pipeline_registry.csv",
        required=False,
    )

    copy_contracts(ROOT / "contracts", output_dir / "contracts")

    total_inputs = 0
    for pipeline_id in pipeline_ids:
        total_inputs += copy_pipeline_inputs(
            ROOT / "inputs",
            output_dir / "inputs",
            pipeline_id,
        )

    if total_inputs == 0:
        print("! no event input files were copied; visual event charts may be limited")

    return output_dir


def parse_args() -> argparse.Namespace:
    """
    Parse CLI options for demo bundle creation.

    Defaults target the strongest demo pipeline because it has multi-day
    conformance history and producer/consumer input files. Use --all-pipelines
    to make demo_data/inputs mirror every local pipeline input folder.
    """
    parser = argparse.ArgumentParser(
        description="Prepare demo_data/ for Streamlit Cloud deployment.",
    )
    parser.add_argument(
        "--pipeline-id",
        action="append",
        default=None,
        help=(
            "Pipeline whose input files should be copied. Can be used multiple "
            "times. Defaults to trade_positions_sftp."
        ),
    )
    parser.add_argument(
        "--all-pipelines",
        action="store_true",
        help="Copy input files for every pipeline folder under inputs/.",
    )
    parser.add_argument(
        "--output-dir",
        default="demo_data",
        help="Demo data output directory. Default: demo_data",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing demo_data directory before copying.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all_pipelines:
        pipeline_ids = discover_input_pipelines(ROOT / "inputs")
    else:
        pipeline_ids = args.pipeline_id or [DEFAULT_DEMO_PIPELINE]

    output_dir = prepare_demo_data(
        output_dir=ROOT / args.output_dir,
        pipeline_ids=pipeline_ids,
        clean=args.clean,
    )

    print()
    print(f"Streamlit demo data ready: {output_dir.relative_to(ROOT)}")
    print("Next local check:")
    print("  FRACTURE_DATA_ROOT=demo_data python -m fracture.cli dashboard")


if __name__ == "__main__":
    main()
