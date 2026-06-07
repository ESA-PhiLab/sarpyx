"""StaMPS command-line preparation helpers."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

_DATE_RE = re.compile(r"(?<!\d)(20\d{6})(?:T|\D|$)")
_METHOD_COMMANDS = {"snap": "mt_prep_snap", "gamma": "mt_prep_gamma"}


def infer_stamps_master_date(master_product: str | Path | None, target_folder: str | Path | None = None) -> str | None:
    """Infer StaMPS master date as YYYYMMDD from product paths when unambiguous."""
    for candidate in _product_date_candidates(master_product):
        match = _DATE_RE.search(candidate)
        if match:
            return match.group(1)

    if target_folder is None:
        return None
    rslc_dir = Path(target_folder).expanduser() / "rslc"
    dates = {
        path.name.split(".", 1)[0]
        for path in rslc_dir.glob("*.rslc")
        if re.fullmatch(r"20\d{6}", path.name.split(".", 1)[0])
    }
    if len(dates) == 1:
        return next(iter(dates))
    return None


def validate_stamps_export_folder(target_folder: str | Path) -> Path:
    """Validate the folder layout written by SNAP ``StampsExport``."""
    target = Path(target_folder).expanduser()
    missing_dirs = [name for name in ("rslc", "diff0", "geo", "dem") if not (target / name).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            f"StaMPS export folder {target} is missing required directories: {', '.join(missing_dirs)}"
        )

    required_patterns = {
        "rslc/*.rslc": target / "rslc",
        "diff0/*.diff": target / "diff0",
        "geo/*.lat": target / "geo",
        "geo/*.lon": target / "geo",
        "dem/*": target / "dem",
    }
    missing_files = [pattern for pattern, folder in required_patterns.items() if not list(folder.glob(pattern.split("/", 1)[1]))]
    if missing_files:
        raise FileNotFoundError(
            f"StaMPS export folder {target} is missing required files: {', '.join(missing_files)}"
        )
    return target


def run_stamps_prep(
    *,
    target_folder: str | Path,
    master_product: str | Path | None = None,
    master_date: str | None = None,
    method: str = "snap",
    da_threshold: float | str = 0.4,
    rg_patches: int | None = 1,
    az_patches: int | None = 1,
    rg_overlap: int | None = 50,
    az_overlap: int | None = 50,
    maskfile: str | Path | None = None,
    command: str | Path | None = None,
    config_file: str | Path | None = None,
    workdir: str | Path | None = None,
    log_path: str | Path | None = None,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the StaMPS post-export preparation command."""
    target = validate_stamps_export_folder(target_folder).resolve()
    resolved_master_date = _validate_master_date(
        master_date or infer_stamps_master_date(master_product, target)
    )
    prep_command = _resolve_prep_command(method, command, config_file=config_file)
    args = _prep_args(
        prep_command,
        resolved_master_date,
        target,
        da_threshold,
        rg_patches,
        az_patches,
        rg_overlap,
        az_overlap,
        maskfile,
    )
    run_dir = Path(workdir).expanduser() if workdir else target
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_path).expanduser() if log_path else run_dir / f"{prep_command}_prep.log"
    completed = _run_command(args, config_file=config_file, cwd=run_dir, timeout=timeout, env=env)
    _write_prep_log(log_file, args, completed)
    if completed.returncode != 0:
        raise RuntimeError(
            f"StaMPS prep failed with exit code {completed.returncode}. See log: {log_file}"
        )
    patch_dirs = sorted(path.name for path in run_dir.glob("PATCH_*") if path.is_dir())
    return {
        "method": method,
        "command": args,
        "target_folder": target,
        "workdir": run_dir,
        "log_path": log_file,
        "master_date": resolved_master_date,
        "patch_dirs": patch_dirs,
    }


def _product_date_candidates(master_product: str | Path | None) -> list[str]:
    if master_product is None:
        return []
    path = Path(master_product).expanduser()
    candidates = [path.name, path.stem, path.as_posix()]
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return candidates
    return [*candidates, resolved.name, resolved.stem, resolved.as_posix()]


def _validate_master_date(master_date: str | None) -> str:
    if master_date is None:
        raise ValueError(
            "Could not infer StaMPS master date. Pass stamps_prep_master_date=YYYYMMDD."
        )
    if not re.fullmatch(r"20\d{6}", str(master_date)):
        raise ValueError(f"StaMPS master date must be YYYYMMDD, got: {master_date!r}")
    return str(master_date)


def _resolve_prep_command(method: str, command: str | Path | None, *, config_file: str | Path | None) -> str:
    if command is not None:
        prep_command = Path(command).expanduser().as_posix() if os.sep in str(command) else str(command)
    else:
        try:
            prep_command = _METHOD_COMMANDS[method]
        except KeyError as exc:
            available = ", ".join(sorted(_METHOD_COMMANDS))
            raise ValueError(f"Unknown StaMPS prep method {method!r}. Available: {available}") from exc
    if config_file is None and shutil.which(prep_command) is None:
        raise FileNotFoundError(
            f"StaMPS prep command not found on PATH: {prep_command}. "
            "Install StaMPS or pass stamps_prep_command/stamps_prep_config_file."
        )
    return prep_command


def _prep_args(
    command: str,
    master_date: str,
    target: Path,
    da_threshold: float | str,
    rg_patches: int | None,
    az_patches: int | None,
    rg_overlap: int | None,
    az_overlap: int | None,
    maskfile: str | Path | None,
) -> list[str]:
    args = [command, master_date, target.as_posix(), str(da_threshold)]
    patch_args = [rg_patches, az_patches, rg_overlap, az_overlap]
    if any(value is not None for value in patch_args):
        if any(value is None for value in patch_args):
            raise ValueError("StaMPS prep patch parameters must be all set or all None.")
        args.extend(str(value) for value in patch_args)
    if maskfile is not None:
        args.append(Path(maskfile).expanduser().as_posix())
    return args


def _run_command(
    args: list[str],
    *,
    config_file: str | Path | None,
    cwd: Path,
    timeout: int | None,
    env: dict[str, str] | None,
) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    if config_file is None:
        return subprocess.run(
            args,
            cwd=cwd,
            env=run_env,
            timeout=timeout,
            text=True,
            capture_output=True,
            check=False,
        )

    config_path = Path(config_file).expanduser()
    suffix = config_path.suffix.lower()
    if suffix in {".csh", ".tcsh"}:
        shell = shutil.which("tcsh") or shutil.which("csh")
        if shell is None:
            raise FileNotFoundError("StaMPS config file is csh/tcsh, but tcsh/csh was not found.")
        shell_args = [shell, "-c", f"source {shlex.quote(config_path.as_posix())}; {shlex.join(args)}"]
    else:
        shell = shutil.which("bash") or shutil.which("sh")
        if shell is None:
            raise FileNotFoundError("No shell found to source StaMPS config file.")
        shell_args = [shell, "-lc", f"source {shlex.quote(config_path.as_posix())} && {shlex.join(args)}"]
    return subprocess.run(
        shell_args,
        cwd=cwd,
        env=run_env,
        timeout=timeout,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_prep_log(log_path: Path, args: list[str], completed: subprocess.CompletedProcess[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"command: {shlex.join(args)}",
                f"returncode: {completed.returncode}",
                "",
                "stdout:",
                completed.stdout or "",
                "",
                "stderr:",
                completed.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
