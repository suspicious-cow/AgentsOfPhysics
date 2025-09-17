"""Streaming CLI."""
from __future__ import annotations

import argparse

from ..workflows.stream_pipeline import run_stream_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run streaming anomaly detection demo")
    parser.add_argument("--config", required=True, help="Path to configuration YAML")
    parser.add_argument("--out", required=False, help="Optional output directory")
    args = parser.parse_args()
    run_stream_pipeline(args.config, args.out)


if __name__ == "__main__":
    main()
