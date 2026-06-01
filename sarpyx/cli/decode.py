"""Command-line wrapper for Sentinel-1 Level-0 decoding."""

from __future__ import annotations

import argparse
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sarpyx-decode",
        description="Decode Sentinel-1 Level-0 products with the SARPYX decode workflow.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to sarpyx.processor.core.decode.",
    )
    return parser


def main() -> int | None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        _parser().print_help()
        return 0

    from sarpyx.processor.core.decode import main as decode_main

    return decode_main()


__all__ = ["main"]
