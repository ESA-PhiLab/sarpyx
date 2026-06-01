"""Command-line interface for YAML-configured SNAP pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sarpyx.snapflow.config_pipeline import (
    ConfigPipelineError,
    list_config_pipelines,
    load_pipeline_config,
    run_config_pipeline,
    validate_pipeline_config,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sarpyx-pipeline",
        description="Run alternate YAML-configured SNAP pipelines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a pipeline YAML file.")
    validate_parser.add_argument("config", type=Path)

    list_parser = subparsers.add_parser("list", help="List pipelines defined in a YAML file.")
    list_parser.add_argument("config", type=Path)

    run_parser = subparsers.add_parser("run", help="Run a configured pipeline.")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--pipeline", "-p", default=None, help="Pipeline name to run.")
    run_parser.add_argument(
        "--set-input",
        dest="inputs",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Bind a declared pipeline input to a product path. May be repeated.",
    )
    run_parser.add_argument("--outdir", "-o", type=Path, default=None, help="Pipeline output directory.")
    run_parser.add_argument("--dry-run", action="store_true", help="Plan steps without executing SNAP.")
    run_parser.add_argument("--resume", action="store_true", help="Reuse matching existing step outputs.")
    run_parser.add_argument("--overwrite", action="store_true", help="Replace existing step outputs.")
    run_parser.add_argument("--json", action="store_true", help="Print run result as JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            config = load_pipeline_config(args.config)
            validate_pipeline_config(config)
            print(f"valid: {args.config}")
            return 0

        if args.command == "list":
            config = load_pipeline_config(args.config)
            validate_pipeline_config(config)
            for name in list_config_pipelines(config):
                print(name)
            return 0

        if args.command == "run":
            result = run_config_pipeline(
                args.config,
                pipeline=args.pipeline,
                inputs=_parse_inputs(args.inputs),
                outdir=args.outdir,
                dry_run=args.dry_run,
                resume=True if args.resume else None,
                overwrite=True if args.overwrite else None,
            )
            if args.json:
                print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
            else:
                for record in result.records:
                    print(f"{record.action}: {record.pipeline}.{record.step_id} -> {record.output}")
                print(f"output: {result.output}")
            return 0

    except ConfigPipelineError as exc:
        parser.exit(2, f"sarpyx-pipeline: error: {exc}\n")

    parser.error(f"Unhandled command: {args.command}")
    return 2


def _parse_inputs(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ConfigPipelineError(f"--set-input must use NAME=PATH, got '{value}'")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ConfigPipelineError(f"--set-input must use NAME=PATH, got '{value}'")
        parsed[name] = path
    return parsed


if __name__ == "__main__":
    sys.exit(main())
