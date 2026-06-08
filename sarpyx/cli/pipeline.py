"""Generic pipeline command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import sys

from sarpyx.pipelines.runner import BUILTIN_PIPELINES, run_pipeline_target


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pipeline", nargs="?", help="Built-in pipeline name, dotted module, or path to a pipeline .py file.")
    parser.add_argument("--input", dest="input_path", help="Input product for single-product pipelines.")
    parser.add_argument("--master", help="Master product for double-product pipelines.")
    parser.add_argument("--slave", help="Slave product for double-product pipelines.")
    parser.add_argument("--output", "-o", dest="output_dir", help="Pipeline output directory.")
    parser.add_argument(
        "--param",
        "-P",
        dest="params",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Recipe parameter. VALUE is parsed as JSON when possible.",
    )
    parser.add_argument("--gpt-path", dest="gpt_path", type=str, default=None)
    parser.add_argument("--gpt-memory", dest="gpt_memory", type=str, default="16G")
    parser.add_argument("--gpt-cache-size", dest="gpt_cache_size", type=str, default="8G")
    parser.add_argument("--gpt-parallelism", dest="gpt_parallelism", type=int, default=6)
    parser.add_argument("--gpt-timeout", dest="gpt_timeout", type=int, default=None)
    parser.add_argument("--snap-userdir", dest="snap_userdir", type=str, default=None)
    parser.add_argument("--product-wkt", dest="product_wkt", type=str, default=None)
    parser.add_argument("--grid-path", dest="grid_path", type=str, default=None)
    parser.add_argument("--cuts-outdir", dest="cuts_outdir", type=str, default=None)
    parser.add_argument("--tile-writer", dest="tile_writer", type=str, default="zarr")
    parser.add_argument("--keep-intermediate", dest="keep_intermediate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--list", dest="list_pipelines", action="store_true", help="List built-in pipelines and exit.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an explicit sarpyx pipeline recipe.")
    add_pipeline_arguments(parser)
    return parser


def parse_params(values: list[str]) -> dict[str, object]:
    params: dict[str, object] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--param must use NAME=VALUE, got: {value}")
        name, raw = value.split("=", 1)
        if not name:
            raise ValueError(f"--param name cannot be empty: {value}")
        params[name] = _parse_value(raw)
    return params


def _parse_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def run(args) -> int:
    if args.list_pipelines:
        for name, spec in sorted(BUILTIN_PIPELINES.items()):
            print(f"{name}\t{spec.input_kind}")
        return 0
    if not args.pipeline:
        raise ValueError("pipeline is required unless --list is used.")
    if not args.output_dir:
        raise ValueError("--output is required unless --list is used.")
    result = run_pipeline_target(
        args.pipeline,
        input_path=args.input_path,
        master=args.master,
        slave=args.slave,
        output_dir=args.output_dir,
        params=parse_params(args.params),
        gpt_path=args.gpt_path,
        gpt_memory=args.gpt_memory,
        gpt_parallelism=args.gpt_parallelism,
        gpt_timeout=args.gpt_timeout,
        gpt_cache_size=args.gpt_cache_size,
        snap_userdir=args.snap_userdir,
        product_wkt=args.product_wkt,
        grid_path=args.grid_path,
        cuts_outdir=args.cuts_outdir,
        tile_writer=args.tile_writer,
        keep_intermediate=args.keep_intermediate,
    )
    print(f"Pipeline result: {result}")
    return 0


def main(argv=None):
    args = create_parser().parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
