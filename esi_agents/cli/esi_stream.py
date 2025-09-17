"""CLI for running the streaming pipeline."""
from __future__ import annotations

import argparse
import asyncio

from ..workflows.stream_pipeline import run_stream


def main() -> None:
    parser = argparse.ArgumentParser(description="Run streaming anomaly detection")
    parser.add_argument("--config", required=True, help="Path to stream YAML config")
    args = parser.parse_args()
    asyncio.run(run_stream(args.config))


if __name__ == "__main__":
    main()
