"""Compatibility entrypoints for removed legacy commands."""

from __future__ import annotations

import argparse


def _main(prog: str, description: str, replacement: str, argv=None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--version", action="store_true", help="Show sarpyx version and exit.")
    args = parser.parse_args(argv)
    if args.version:
        from sarpyx import __version__

        print(__version__)
        return 0
    parser.print_help()
    print(f"\nThis legacy command is no longer implemented directly. Use `{replacement}`.")
    return 0


def decode_main(argv=None) -> int:
    return _main("sarpyx-decode", "Legacy decode command.", "sarpyx worldsar", argv)


def focus_main(argv=None) -> int:
    return _main("sarpyx-focus", "Legacy focus command.", "sarpyx worldsar", argv)


def shipdet_main(argv=None) -> int:
    return _main("sarpyx-shipdet", "Legacy ship detection command.", "sarpyx worldsar", argv)


def unzip_main(argv=None) -> int:
    return _main("sarpyx-unzip", "Legacy unzip command.", "sarpyx worldsar", argv)


def upload_main(argv=None) -> int:
    return _main("sarpyx-upload", "Legacy upload command.", "sarpyx worldsar", argv)
