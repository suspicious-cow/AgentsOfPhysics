"""CLI entry point for batch scoring."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..workflows.batch_pipeline import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch anomaly detection")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--input", required=False, help="Input data path override")
    parser.add_argument("--out", required=True, help="Output directory for artifacts")
    parser.add_argument("--labels", required=False, help="Optional labels CSV")
    args = parser.parse_args()
    run_batch(args.config, args.input, args.out, args.labels)


if __name__ == "__main__":
    main()
