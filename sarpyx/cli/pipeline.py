"""Generic pipeline command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import sys


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
    if raw in {"True", "False"}:
        return raw == "True"
    if raw == "None":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _builtin_pipelines():
    from sarpyx.pipelines.runner import BUILTIN_PIPELINES

    return BUILTIN_PIPELINES


def run_pipeline_target(*args, **kwargs):
    from sarpyx.pipelines.runner import run_pipeline_target as dispatch

    return dispatch(*args, **kwargs)


def _has_worldsar_tiling_step(pipeline: str, params: dict[str, object]) -> bool:
    spec = _builtin_pipelines().get(pipeline)
    if spec is None:
        return False
    steps = getattr(spec.module, "steps", None)
    if steps is None:
        return False
    return any(getattr(step, "name", None) == "WorldSARTiling" for step in steps(**params))


def _pipeline_warnings(args, params: dict[str, object]) -> list[str]:
    if not _has_worldsar_tiling_step(args.pipeline, params):
        return []
    warnings = []
    if args.cuts_outdir:
        if not args.grid_path:
            warnings.append("--grid-path was not supplied; using the default grid when cuts are requested.")
        if not args.product_wkt:
            warnings.append("--product-wkt was not supplied; sarpyx will derive the tiling footprint from the processed raster when possible.")
    elif args.grid_path:
        warnings.append("--cuts-outdir was not supplied; using the default output tiles directory.")
        if not args.product_wkt:
            warnings.append("--product-wkt was not supplied; sarpyx will derive the tiling footprint from the processed raster when possible.")
    else:
        warnings.append("--cuts-outdir and --grid-path were not supplied; WorldSARTiling will be skipped for this explicit pipeline run.")
    return warnings


def run(args) -> int:
    if args.list_pipelines:
        for name, spec in sorted(_builtin_pipelines().items()):
            print(f"{name}\t{spec.input_kind}")
        return 0
    if not args.pipeline:
        raise ValueError("pipeline is required unless --list is used.")
    if not args.output_dir:
        raise ValueError("--output is required unless --list is used.")
    params = parse_params(args.params)
    warnings = _pipeline_warnings(args, params)
    result = run_pipeline_target(
        args.pipeline,
        input_path=args.input_path,
        master=args.master,
        slave=args.slave,
        output_dir=args.output_dir,
        params=params,
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
    if warnings:
        print("Pipeline warnings:")
        for warning in warnings:
            print(f"WARNING: {warning}")
    return 0


def main(argv=None):
    args = create_parser().parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
