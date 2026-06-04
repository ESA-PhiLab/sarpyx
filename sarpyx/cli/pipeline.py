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
        usage=(
            "sarpyx-pipeline CONFIG.yaml "
            "[--master MASTER.SAFE --slave SLAVE.SAFE --output OUTDIR]\n"
            "       sarpyx-pipeline {validate,list,run} ..."
        ),
        description=(
            "Run YAML-configured SNAP pipelines. Direct run form: "
            "sarpyx-pipeline pipeline.yaml --master MASTER.SAFE --slave SLAVE.SAFE --output OUTDIR"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a pipeline YAML file.")
    validate_parser.add_argument("config", type=Path)

    list_parser = subparsers.add_parser("list", help="List pipelines defined in a YAML file.")
    list_parser.add_argument("config", type=Path)

    run_parser = subparsers.add_parser("run", help="Run a configured pipeline.")
    run_parser.add_argument("config", type=Path)
    _add_run_arguments(run_parser)

    return parser


def create_direct_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sarpyx-pipeline",
        description="Run a YAML-configured SNAP pipeline.",
    )
    parser.add_argument("config", type=Path)
    _add_run_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser: argparse.ArgumentParser | None = None

    try:
        if _is_direct_run(raw_argv):
            parser = create_direct_run_parser()
            args = parser.parse_args(raw_argv)
            return _run_from_args(args)

        parser = create_parser()
        args = parser.parse_args(raw_argv)

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
            return _run_from_args(args)

    except ConfigPipelineError as exc:
        if parser is None:
            parser = create_parser()
        parser.exit(2, f"sarpyx-pipeline: error: {exc}\n")

    parser.error(f"Unhandled command: {args.command}")
    return 2


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pipeline", "-p", default=None, help="Pipeline name to run.")
    parser.add_argument(
        "--set-input",
        dest="inputs",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Bind a declared pipeline input to a product path. May be repeated.",
    )
    parser.add_argument("--master", type=Path, default=None, help="Bind the 'master' input path.")
    parser.add_argument("--slave", type=Path, default=None, help="Bind the 'slave' input path.")
    parser.add_argument(
        "--outdir",
        "--output",
        "-o",
        dest="outdir",
        type=Path,
        default=None,
        help="Pipeline output directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan steps without executing SNAP.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse matching existing step outputs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing step outputs.")
    parser.add_argument("--json", action="store_true", help="Print run result as JSON.")


def _is_direct_run(argv: list[str]) -> bool:
    return bool(argv) and not argv[0].startswith("-") and argv[0] not in {"validate", "list", "run"}


def _run_from_args(args: argparse.Namespace) -> int:
    result = run_config_pipeline(
        args.config,
        pipeline=args.pipeline,
        inputs=_parse_inputs(args.inputs, master=args.master, slave=args.slave),
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


def _parse_inputs(
    values: list[str],
    *,
    master: Path | None = None,
    slave: Path | None = None,
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ConfigPipelineError(f"--set-input must use NAME=PATH, got '{value}'")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ConfigPipelineError(f"--set-input must use NAME=PATH, got '{value}'")
        parsed[name] = path
    _set_shortcut_input(parsed, "master", master)
    _set_shortcut_input(parsed, "slave", slave)
    return parsed


def _set_shortcut_input(parsed: dict[str, str], name: str, value: Path | None) -> None:
    if value is None:
        return
    path = str(value)
    if name in parsed and parsed[name] != path:
        raise ConfigPipelineError(f"Input '{name}' was provided twice with different paths")
    parsed[name] = path


if __name__ == "__main__":
    sys.exit(main())
