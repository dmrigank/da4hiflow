#!/usr/bin/env python
"""DA4HiFlow experiment runner."""

import sys
import argparse
from pathlib import Path


def main():
    """Run the DA4HiFlow experiment."""
    parser = argparse.ArgumentParser(
        description="DA4HiFlow - Data Assimilation for High-Speed Flow"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file (optional)"
    )

    args = parser.parse_args()

    print("DA4HiFlow runner stub")
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            print(f"Config loaded from: {args.config}")
        else:
            print(f"Warning: Config file not found: {args.config}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
