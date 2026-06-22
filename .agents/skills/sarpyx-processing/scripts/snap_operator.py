#!/usr/bin/env python3
"""Run any SNAP GPT operator declared in sarpyx/snapflow/op.py."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    operators = load_operators(repo)
    report: dict[str, Any] = {
        "repo": str(repo),
        "operator_count": len(operators),
        "operator": args.operator,
        "command": [],
        "blockers": [],
        "warnings": [],
    }

    if args.list:
        print("\n".join(operators))
        return 0

    if not args.operator:
        report["blockers"].append("--operator is required unless --list is used")
        print_json(report)
        return 2
    if args.operator not in operators:
        matches = [op for op in operators if args.operator.lower() in op.lower()]
        report["blockers"].append(f"operator is not declared in sarpyx/snapflow/op.py: {args.operator}")
        report["matches"] = matches[:20]
        print_json(report)
        return 2

    try:
        gpt = resolve_gpt(args.gpt_path)
    except FileNotFoundError as exc:
        if not args.dry_run:
            report["blockers"].append(str(exc))
            print_json(report)
            return 2
        gpt = args.gpt_path or "<gpt>"
        report["warnings"].append(str(exc))

    if args.help_operator:
        command = [gpt, "-h", args.operator]
    else:
        try:
            command = build_command(args, gpt)
        except ValueError as exc:
            report["blockers"].append(str(exc))
            print_json(report)
            return 2
    report["command"] = command

    if args.dry_run:
        print_json(report)
        return 0

    if args.target:
        Path(args.target).expanduser().absolute().parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.timeout)
    report.update({"returncode": result.returncode, "stdout": trim(result.stdout), "stderr": trim(result.stderr)})
    print_json(report)
    return result.returncode


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a SNAP GPT operator listed in sarpyx/snapflow/op.py.")
    parser.add_argument("--repo", default=".", help="Path to the sarpyx repository.")
    parser.add_argument("--operator", help="Exact SNAP operator name from op.py.")
    parser.add_argument("--list", action="store_true", help="List supported operators and exit.")
    parser.add_argument("--help-operator", action="store_true", help="Run `gpt -h OPERATOR` instead of processing.")
    parser.add_argument("--input", dest="input_path", help="Default input product, passed as -Ssource=PATH.")
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH", help="Additional source parameter.")
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="Operator parameter.")
    parser.add_argument("--target", help="Output target path passed via -t.")
    parser.add_argument("--format", default="BEAM-DIMAP", help="Output format passed via -f when --target is set.")
    parser.add_argument("--gpt-path", help="Explicit SNAP GPT executable.")
    parser.add_argument("--snap-userdir", help="SNAP user directory.")
    parser.add_argument("--memory", help="Java heap, e.g. 16G.")
    parser.add_argument("--cache-size", help="Tile cache size, e.g. 8G.")
    parser.add_argument("--parallelism", type=int, help="GPT -q parallelism.")
    parser.add_argument("--raw-arg", action="append", default=[], help="Raw argument appended after the operator.")
    parser.add_argument("--timeout", type=int, default=7200, help="Subprocess timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON command report without executing.")
    return parser.parse_args(argv)


def load_operators(repo: Path) -> list[str]:
    path = repo / "sarpyx" / "snapflow" / "op.py"
    if not path.is_file():
        raise FileNotFoundError(f"op.py not found: {path}")
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "snap_operators" in names:
                value = ast.literal_eval(node.value)
                return [str(item) for item in value]
    raise ValueError(f"snap_operators list not found in {path}")


def resolve_gpt(gpt_path: str | None) -> str:
    explicit = Path(gpt_path).expanduser() if gpt_path else None
    if explicit:
        if explicit.exists() and os.access(explicit, os.X_OK) and str(explicit) != "/usr/sbin/gpt":
            return str(explicit)
        raise FileNotFoundError(f"SNAP GPT executable not found or invalid: {explicit}")
    candidates = [
        os.getenv("GPT_PATH"),
        os.getenv("gpt_path"),
        str(Path(os.environ["CONDA_PREFIX"]) / "opt" / "esa-snap" / "bin" / "gpt") if os.getenv("CONDA_PREFIX") else None,
        str(Path(os.environ["SNAP_HOME"]) / "bin" / "gpt") if os.getenv("SNAP_HOME") else None,
        shutil.which("gpt"),
    ]
    for candidate in [c for c in candidates if c]:
        path = Path(candidate).expanduser()
        if valid_gpt(path):
            return str(path)
    raise FileNotFoundError("SNAP GPT executable not found. Pass --gpt-path explicitly.")


def valid_gpt(path: Path) -> bool:
    if not (path.exists() and os.access(path, os.X_OK)) or str(path) == "/usr/sbin/gpt":
        return False
    return any(part in {"esa-snap", "snap"} for part in path.parts)


def build_command(args: argparse.Namespace, gpt: str) -> list[str]:
    command = [gpt]
    if args.snap_userdir:
        Path(args.snap_userdir).expanduser().mkdir(parents=True, exist_ok=True)
        command.append(f"-J-Dsnap.userdir={Path(args.snap_userdir).expanduser().as_posix()}")
    if args.parallelism:
        command.extend(["-q", str(args.parallelism)])
    if args.memory:
        command.append(f"-J-Xmx{args.memory}")
    if args.cache_size:
        command.extend(["-c", args.cache_size])
    command.extend(["-x", "-e"])
    for source in collect_sources(args):
        command.append(f"-S{source[0]}={source[1]}")
    command.append(args.operator)
    for name, value in parse_assignments(args.param, "--param"):
        command.append(f"-P{name}={value}")
    command.extend(args.raw_arg)
    if args.target:
        command.extend(["-t", str(Path(args.target).expanduser().absolute()), "-f", args.format])
    return command


def collect_sources(args: argparse.Namespace) -> list[tuple[str, str]]:
    sources = [("source", args.input_path)] if args.input_path else []
    sources.extend(parse_assignments(args.source, "--source"))
    missing = [value for _, value in sources if not Path(value).expanduser().exists()]
    if missing:
        raise ValueError(f"source path does not exist: {missing[0]}")
    return [(name, str(Path(value).expanduser().resolve())) for name, value in sources]


def parse_assignments(values: list[str], label: str) -> list[tuple[str, str]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use NAME=VALUE, got: {value}")
        name, raw = value.split("=", 1)
        if not name:
            raise ValueError(f"{label} name cannot be empty: {value}")
        parsed.append((name, raw))
    return parsed


def trim(value: str, limit: int = 4000) -> str:
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"


def print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
