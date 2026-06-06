"""Runtime configuration for WorldSAR workflows."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ORBIT_TYPE = "Sentinel Precise (Auto Download)"
DEFAULT_ZARR_CHUNK_SIZE = (128, 128)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env(*names: str, default=None):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _expand_path(path_value) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).absolute()
    return path


def _ensure_existing_path(path_value, label: str) -> Path:
    path = _expand_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _is_snap_gpt(path: Path) -> bool:
    if not (path.exists() and os.access(path, os.X_OK)):
        return False
    if str(path) == "/usr/sbin/gpt":
        return False
    return any(part in {"esa-snap", "snap"} for part in path.parts)


def resolve_gpt_path(gpt_path=None) -> str:
    explicit_path = Path(gpt_path).expanduser() if gpt_path else None
    if explicit_path:
        if explicit_path.exists() and os.access(explicit_path, os.X_OK) and str(explicit_path) != "/usr/sbin/gpt":
            return str(explicit_path)
        raise FileNotFoundError(f"SNAP GPT executable not found or invalid: {explicit_path}")
    candidates = []
    candidates.extend(
        value
        for value in (
            _env("gpt_path", "GPT_PATH"),
            str(Path(os.environ["CONDA_PREFIX"]) / "opt" / "esa-snap" / "bin" / "gpt")
            if os.environ.get("CONDA_PREFIX")
            else None,
            str(Path(sys.prefix) / "opt" / "esa-snap" / "bin" / "gpt"),
            str(Path(os.environ["SNAP_HOME"]) / "bin" / "gpt") if os.environ.get("SNAP_HOME") else None,
            shutil.which("gpt"),
        )
        if value
    )

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if _is_snap_gpt(path):
            return str(path)
    raise FileNotFoundError(
        "SNAP GPT executable not found. Activate the conda environment containing SNAP "
        "or pass --gpt-path explicitly."
    )


GPT_PATH = _env("gpt_path", "GPT_PATH")
GRID_PATH = _env("grid_path", "GRID_PATH")
DB_DIR = _env("db_dir", "DB_DIR")
CUTS_OUTDIR = _env("cuts_outdir", "OUTPUT_CUTS_DIR")
BASE_PATH = _env("base_path", "BASE_PATH", default=str(PROJECT_ROOT))
SNAP_USERDIR = _env("SNAP_USERDIR", "snap_userdir", default=str(PROJECT_ROOT / ".snap"))
os.environ.setdefault("SNAP_USERDIR", SNAP_USERDIR)

prepro = True
tiling = True
db_indexing = True


def apply_runtime_overrides(args) -> None:
    global GPT_PATH, GRID_PATH, DB_DIR, SNAP_USERDIR
    if args.gpt_path:
        GPT_PATH = resolve_gpt_path(args.gpt_path)
    else:
        try:
            GPT_PATH = resolve_gpt_path()
        except FileNotFoundError:
            GPT_PATH = _env("gpt_path", "GPT_PATH")
    if args.grid_path:
        GRID_PATH = args.grid_path
    if args.db_dir:
        DB_DIR = args.db_dir
    if args.snap_userdir:
        SNAP_USERDIR = args.snap_userdir
        os.environ["SNAP_USERDIR"] = SNAP_USERDIR


def validate_runtime_args(args) -> None:
    from sarpyx.snapflow.tile_writers import normalize_tile_writer

    normalize_tile_writer(getattr(args, "tile_writer", "zarr"))
    if args.gpt_parallelism is not None and args.gpt_parallelism <= 0:
        raise ValueError(f"--gpt-parallelism must be > 0, got {args.gpt_parallelism}")
    if args.gpt_timeout is not None and args.gpt_timeout < 0:
        raise ValueError(f"--gpt-timeout must be >= 0, got {args.gpt_timeout}")
    if len(args.zarr_chunk_size) != 2 or any(size <= 0 for size in args.zarr_chunk_size):
        raise ValueError(f"--zarr-chunk-size must contain two positive integers, got {args.zarr_chunk_size}")
    if args.sentinel_first_burst is not None and args.sentinel_first_burst < 1:
        raise ValueError(f"--sentinel-first-burst must be >= 1, got {args.sentinel_first_burst}")
    if args.sentinel_last_burst is not None and args.sentinel_last_burst < 1:
        raise ValueError(f"--sentinel-last-burst must be >= 1, got {args.sentinel_last_burst}")
    if (
        args.sentinel_first_burst is not None
        and args.sentinel_last_burst is not None
        and args.sentinel_last_burst < args.sentinel_first_burst
    ):
        raise ValueError(
            "--sentinel-last-burst must be greater than or equal to "
            f"--sentinel-first-burst, got {args.sentinel_last_burst} < {args.sentinel_first_burst}"
        )
    if args.sentinel_subap_decompositions is not None:
        invalid = [n for n in args.sentinel_subap_decompositions if n < 2]
        if invalid:
            raise ValueError(f"--sentinel-subap-decompositions values must be >= 2, got {invalid}")
    if args.sentinel_subap_feature_window_size < 1 or args.sentinel_subap_feature_window_size % 2 == 0:
        raise ValueError(
            "--sentinel-subap-feature-window-size must be a positive odd integer, "
            f"got {args.sentinel_subap_feature_window_size}"
        )


def resolve_db_dir(cuts_outdir=None) -> Path:
    global DB_DIR
    if DB_DIR:
        db_dir = _expand_path(DB_DIR)
    else:
        if cuts_outdir is None:
            raise ValueError("db_dir not provided and no default output root is available.")
        db_dir = _expand_path(Path(cuts_outdir) / "_db")
        DB_DIR = str(db_dir)
        print(f"DB_DIR not configured; defaulting to {db_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


def ensure_grid_file(grid_path: Path, base_path: Path) -> Path:
    if grid_path.exists():
        return grid_path
    grid_dir = base_path / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    print(f"Grid file not found at {grid_path}. Generating grid_10km.geojson in {grid_dir}.")
    subprocess.run([sys.executable, "-m", "sarpyx.utils.grid"], cwd=grid_dir, check=True)
    generated = grid_dir / "grid_10km.geojson"
    if not generated.exists():
        raise FileNotFoundError(f"Grid generation completed, but {generated} was not created.")
    return generated
