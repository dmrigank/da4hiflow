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
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="runs",
        help="Directory where run outputs are stored",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (defaults to timestamp)",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    import json

    with open(config_path) as f:
        config = json.load(f)

    from da4hiflow.core.runner import run_experiment

    out_dir = run_experiment(
        config, output_root=args.output_root, run_name=args.run_name
    )

    print(f"Output directory: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
