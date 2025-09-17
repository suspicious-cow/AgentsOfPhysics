"""Report viewer CLI."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Display generated report")
    parser.add_argument("--report", required=True, help="Path to report.md")
    args = parser.parse_args()
    path = Path(args.report)
    if not path.exists():
        raise FileNotFoundError(path)
    print(path.read_text())


if __name__ == "__main__":
    main()
