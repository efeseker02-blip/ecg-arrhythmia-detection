#!/usr/bin/env python
"""Convenience entry point to download the MIT-BIH Arrhythmia Database.

Usage
-----
    python download_data.py                 # download all records
    python download_data.py --records 100 101 102

The database (~100 MB) is fetched from PhysioNet into the directory configured
under ``data.raw_dir`` in ``config.yaml`` (default ``data/mitdb``).
"""

from __future__ import annotations

import argparse

from src.preprocessing import download_database
from src.utils import get_logger, load_config

logger = get_logger("download")


def main() -> None:
    """Parse args and download the requested records."""
    parser = argparse.ArgumentParser(description="Download the MIT-BIH database.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--records",
        nargs="*",
        default=None,
        help="Specific record ids to download (default: the whole database).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = config.data["raw_dir"]
    download_database(raw_dir, records=args.records)
    logger.info("MIT-BIH data is ready in %s", raw_dir)


if __name__ == "__main__":
    main()
