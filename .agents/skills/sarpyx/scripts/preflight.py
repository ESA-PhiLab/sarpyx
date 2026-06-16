#!/usr/bin/env python3
"""Read-only preflight report for sarpyx processing requests."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


BUILTIN_FALLBACK = {"s1_tops": "single", "s1_strip": "single", "tsx": "single", "csg": "single", "biomass": "single", "nisar": "single", "s1_insar": "double"}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    mode = infer_mode(args)
    uv = shutil.which("uv")
    report: dict[str, Any] = {
        "repo": str(repo),
        "mode": mode,
        "tools": {"uv": uv, "python": sys.executable},
        "cli": {},
        "pipelines": {},
        "paths": {},
        "grid": {},
        "snap_gpt": {},
        "recommended_command": [],
        "blockers": [],
        "warnings": [],
    }

    if not repo.is_dir():
        report["blockers"].append(f"repo does not exist: {repo}")
        print_json(report)
        return 2

    if not args.skip_cli_check:
        report["cli"]["sarpyx_help"] = command_result(cli_command(uv, "sarpyx", "--help"), repo)
        list_result = command_result(cli_command(uv, "sarpyx", "pipeline", "--list"), repo)
        report["cli"]["pipeline_list"] = list_result
        report["pipelines"] = parse_pipeline_list(list_result.get("stdout", "")) or parse_runner_file(repo)
        if report["cli"]["sarpyx_help"].get("returncode") != 0:
            report["warnings"].append("sarpyx --help check failed; inspect cli.sarpyx_help.stderr")
        if list_result.get("returncode") != 0:
            report["warnings"].append("pipeline list check failed; using source fallback if available")
    else:
        report["pipelines"] = parse_runner_file(repo)

    if not report["pipelines"]:
        report["pipelines"] = dict(BUILTIN_FALLBACK)
        report["warnings"].append("could not inspect pipeline list; using built-in fallback names")

    add_path_status(report, args)
    report["grid"] = resolve_grid(args, repo)
    report["snap_gpt"] = resolve_gpt(args)
    report["recommended_command"] = recommended_command(args, uv, mode, report)
    add_blockers(report, args, mode)
    print_json(report)
    return 2 if report["blockers"] else 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight sarpyx processing inputs without running processing.")
    parser.add_argument("--repo", default=".", help="Path to the sarpyx repository.")
    parser.add_argument("--mode", choices=("auto", "worldsar", "pipeline", "make"), default="auto")
    parser.add_argument("--pipeline", help="Built-in recipe, dotted module, or pipeline .py file.")
    parser.add_argument("--make-target", help="Makefile target for validation or operations.")
    parser.add_argument("--input", dest="input_path", help="Input product for single-product processing.")
    parser.add_argument("--master", help="Master product for double-product processing.")
    parser.add_argument("--slave", help="Slave product for double-product processing.")
    parser.add_argument("--output", dest="output_dir", help="Processing output root.")
    parser.add_argument("--cuts-outdir", help="Tile output directory.")
    parser.add_argument("--grid-path", help="Grid GeoJSON path.")
    parser.add_argument("--gpt-path", help="SNAP GPT executable path.")
    parser.add_argument("--snap-userdir", help="SNAP user directory.")
    parser.add_argument("--product-wkt", help="Product WKT supplied by the caller.")
    parser.add_argument("--require-grid", action="store_true", help="Require a resolved grid path.")
    parser.add_argument("--require-product-wkt", action="store_true", help="Require explicit product WKT.")
    parser.add_argument("--h5-to-zarr-only", action="store_true", help="Preflight a WorldSAR H5-to-Zarr conversion.")
    parser.add_argument("--skip-cli-check", action="store_true", help="Skip uv/CLI subprocess checks.")
    return parser.parse_args(argv)


def infer_mode(args: argparse.Namespace) -> str:
    if args.mode != "auto":
        return args.mode
    if args.make_target:
        return "make"
    if args.pipeline:
        return "pipeline"
    if not any([args.input_path, args.master, args.slave, args.output_dir]):
        return "inspect"
    return "worldsar"


def cli_command(uv: str | None, *parts: str) -> list[str]:
    if uv:
        return [uv, "run", *parts]
    executable = shutil.which(parts[0])
    return [executable or parts[0], *parts[1:]]


def command_result(command: list[str], cwd: Path, timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        return {"command": command, "returncode": proc.returncode, "stdout": trim(proc.stdout), "stderr": trim(proc.stderr)}
    except FileNotFoundError as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"command": command, "returncode": None, "stdout": trim(exc.stdout or ""), "stderr": f"timed out after {timeout}s"}


def trim(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def parse_pipeline_list(stdout: str) -> dict[str, str]:
    pipelines: dict[str, str] = {}
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and parts[1] in {"single", "double"}:
            pipelines[parts[0]] = parts[1]
    return pipelines


def parse_runner_file(repo: Path) -> dict[str, str]:
    path = repo / "sarpyx" / "pipelines" / "runner.py"
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    pattern = r'"([A-Za-z0-9_]+)":\s*PipelineSpec\("[^"]+",\s*"([^"]+)"'
    return {name: kind for name, kind in re.findall(pattern, text) if kind in {"single", "double"}}


def add_path_status(report: dict[str, Any], args: argparse.Namespace) -> None:
    fields = {
        "input": args.input_path,
        "master": args.master,
        "slave": args.slave,
        "output": args.output_dir,
        "cuts_outdir": args.cuts_outdir,
        "grid_path": args.grid_path,
        "gpt_path": args.gpt_path,
        "snap_userdir": args.snap_userdir,
    }
    report["paths"] = {name: path_status(value) for name, value in fields.items() if value}


def path_status(value: str) -> dict[str, Any]:
    path = Path(value).expanduser()
    return {
        "path": str(path.resolve() if path.exists() else path.absolute()),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "executable": os.access(path, os.X_OK) if path.exists() else False,
    }


def resolve_grid(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    candidates = [
        args.grid_path,
        os.getenv("GRID_PATH"),
        os.getenv("grid_path"),
        str(repo / "grid" / "grid_10km.geojson"),
    ]
    grid_dir = repo / "grid"
    if grid_dir.is_dir():
        candidates.extend(str(path) for path in sorted(grid_dir.glob("*.geojson")))
    checked = [path_status(value) for value in dict.fromkeys(c for c in candidates if c)]
    resolved = next((item for item in checked if item["exists"] and item["is_file"]), None)
    return {"resolved": resolved, "checked": checked}


def resolve_gpt(args: argparse.Namespace) -> dict[str, Any]:
    candidates = [
        args.gpt_path,
        os.getenv("GPT_PATH"),
        os.getenv("gpt_path"),
        str(Path(os.environ["CONDA_PREFIX"]) / "opt" / "esa-snap" / "bin" / "gpt")
        if os.getenv("CONDA_PREFIX")
        else None,
        str(Path(os.environ["SNAP_HOME"]) / "bin" / "gpt") if os.getenv("SNAP_HOME") else None,
        shutil.which("gpt"),
    ]
    checked = [path_status(value) for value in dict.fromkeys(c for c in candidates if c)]
    resolved = next((item for item in checked if valid_gpt(Path(item["path"]))), None)
    return {"resolved": resolved, "checked": checked}


def valid_gpt(path: Path) -> bool:
    if not (path.exists() and os.access(path, os.X_OK)):
        return False
    if str(path) == "/usr/sbin/gpt":
        return False
    return any(part in {"esa-snap", "snap"} for part in path.parts)


def recommended_command(args: argparse.Namespace, uv: str | None, mode: str, report: dict[str, Any]) -> list[str]:
    prefix = ["uv", "run"] if uv else []
    if mode == "inspect":
        return []
    if mode == "make":
        return ["make", args.make_target or "<target>"]
    if mode == "pipeline":
        command = [*prefix, "sarpyx", "pipeline", args.pipeline or "<pipeline>"]
        pipelines = report.get("pipelines", {})
        kind = pipelines.get(args.pipeline)
        if kind == "double" or args.pipeline == "s1_insar":
            command.extend(["--master", args.master or "<master>", "--slave", args.slave or "<slave>"])
        else:
            command.extend(["--input", args.input_path or "<input>"])
        append_common(command, args)
        return command
    command = [*prefix, "sarpyx", "worldsar", "--input", args.input_path or "<input>"]
    append_common(command, args)
    if args.h5_to_zarr_only:
        command.append("--h5-to-zarr-only")
    return command


def append_common(command: list[str], args: argparse.Namespace) -> None:
    pairs = [
        ("--output", args.output_dir),
        ("--cuts-outdir", args.cuts_outdir),
        ("--grid-path", args.grid_path),
        ("--gpt-path", args.gpt_path),
        ("--snap-userdir", args.snap_userdir),
        ("--product-wkt", args.product_wkt),
    ]
    for flag, value in pairs:
        if value:
            command.extend([flag, value])


def add_blockers(report: dict[str, Any], args: argparse.Namespace, mode: str) -> None:
    blockers = report["blockers"]
    if report["tools"]["uv"] is None:
        blockers.append("uv is not available in PATH")
    if mode == "inspect":
        return
    if mode == "make":
        if not args.make_target:
            blockers.append("make target is required")
        return
    if not args.output_dir:
        blockers.append("--output is required for autonomous processing")
    if mode == "worldsar":
        require_existing(blockers, args.input_path, "--input")
        if report["grid"]["resolved"] is None:
            blockers.append("grid path could not be resolved")
    elif mode == "pipeline":
        kind = report.get("pipelines", {}).get(args.pipeline)
        if not args.pipeline:
            blockers.append("pipeline name or file is required")
        elif kind == "double" or args.pipeline == "s1_insar":
            require_existing(blockers, args.master, "--master")
            require_existing(blockers, args.slave, "--slave")
        else:
            require_existing(blockers, args.input_path, "--input")
        if args.require_grid and report["grid"]["resolved"] is None:
            blockers.append("grid path could not be resolved")
    if report["snap_gpt"]["resolved"] is None:
        blockers.append("SNAP GPT executable could not be resolved")
    if args.require_product_wkt and not args.product_wkt:
        blockers.append("--product-wkt is required for this request")


def require_existing(blockers: list[str], value: str | None, label: str) -> None:
    if not value:
        blockers.append(f"{label} is required")
        return
    if not Path(value).expanduser().exists():
        blockers.append(f"{label} does not exist: {value}")


def print_json(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
