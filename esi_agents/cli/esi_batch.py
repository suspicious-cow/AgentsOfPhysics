"""Batch scoring CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..workflows.batch_pipeline import run_batch_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch anomaly detection pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration YAML")
    parser.add_argument("--input", required=False, help="Optional override for input file")
    parser.add_argument("--out", required=True, help="Output directory for artifacts")
    args = parser.parse_args()

    result = run_batch_pipeline(args.config, args.input, args.out)
    print(f"Artifacts written to {result.output_dir}")
    print(f"Report available at {result.report_path}")


if __name__ == "__main__":
    main()
