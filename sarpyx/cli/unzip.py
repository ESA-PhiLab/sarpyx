"""Command-line wrapper for unpacking Sentinel-1 ZIP products."""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger("sarpyx.unzip")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sarpyx-unzip",
        description="Extract Sentinel-1 ZIP products from a directory.",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing Sentinel-1 ZIP products.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep ZIP files after extraction.",
    )
    return parser


def find_zip_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.name.startswith("S1") and path.suffix.lower() == ".zip"
    )


def extract_zip(zip_path: Path, output_dir: Path, keep: bool = False) -> None:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        LOGGER.info("Extracting %s to %s", zip_path, output_dir)
        zip_ref.extractall(output_dir)
    if not keep:
        zip_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")

    zip_files = find_zip_files(args.input_dir)
    if not zip_files:
        parser.error(f"no Sentinel-1 ZIP files found in {args.input_dir}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for zip_path in zip_files:
        extract_zip(zip_path, args.input_dir, keep=args.keep)
    return 0


__all__ = ["create_parser", "extract_zip", "find_zip_files", "main"]
