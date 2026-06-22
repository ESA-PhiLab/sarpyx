"""Top-level sarpyx command dispatcher."""

from __future__ import annotations

import argparse
import sys

from sarpyx.cli.pipeline import add_pipeline_arguments
from sarpyx.cli.pipeline import run as run_pipeline
from sarpyx.snapflow.parser import add_worldsar_arguments


def run_worldsar(args):
    from sarpyx.snapflow.runner import run

    return run(args)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sarpyx")
    parser.add_argument("--version", action="store_true", help="Show sarpyx version and exit.")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    worldsar = subparsers.add_parser("worldsar", help="Run the WorldSAR processing pipeline.")
    add_worldsar_arguments(worldsar)
    worldsar.set_defaults(func=run_worldsar)

    pipeline = subparsers.add_parser("pipeline", help="Run an explicit pipeline recipe.")
    add_pipeline_arguments(pipeline)
    pipeline.set_defaults(func=run_pipeline)

    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.version:
        from sarpyx import __version__

        print(__version__)
        return 0
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
