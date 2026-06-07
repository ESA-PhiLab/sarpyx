"""Generic execution helpers for declared sarpyx pipeline recipes."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from sarpyx.pipelines.double_product import s1_insar, snap2stamps
from sarpyx.pipelines.single_product import biomass, csg, nisar, s1_strip, s1_tops, tsx
from sarpyx.hooks.subap_features import SubapFeatureConfig
from sarpyx.hooks.worldsar import make_worldsar_zarr_tile_hook, product_output_name
from sarpyx.snapflow import config
from sarpyx.snapflow.insar import run_insar_pipeline
from sarpyx.snapflow.preprocessing import (
    run_biomass_pipeline,
    run_nisar_pipeline,
    run_sentinel_strip_pipeline,
    run_sentinel_tops_pipeline,
    run_tsx_csg_pipeline,
)
from sarpyx.snapflow.product import resolve_product_wkt
from sarpyx.snapflow.runtime import make_context, run_steps


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    input_kind: str
    module: ModuleType
    runner: Callable[..., Any] | None = None
    product_mode: str | None = None


BUILTIN_PIPELINES: dict[str, PipelineSpec] = {
    "s1_tops": PipelineSpec("s1_tops", "single", s1_tops, run_sentinel_tops_pipeline, "S1TOPS"),
    "s1_strip": PipelineSpec("s1_strip", "single", s1_strip, run_sentinel_strip_pipeline, "S1STRIP"),
    "tsx": PipelineSpec("tsx", "single", tsx, run_tsx_csg_pipeline, "TSX"),
    "csg": PipelineSpec("csg", "single", csg, run_tsx_csg_pipeline, "CSG"),
    "biomass": PipelineSpec("biomass", "single", biomass, run_biomass_pipeline, "BM"),
    "nisar": PipelineSpec("nisar", "single", nisar, run_nisar_pipeline, "NISAR"),
    "s1_insar": PipelineSpec("s1_insar", "double", s1_insar, product_mode="S1INSAR"),
    "2stamps": PipelineSpec("2stamps", "double", snap2stamps, product_mode="S1STAMPS"),
}


def load_pipeline(target: str) -> PipelineSpec:
    if target in BUILTIN_PIPELINES:
        return BUILTIN_PIPELINES[target]
    path = Path(target).expanduser()
    module = _load_module_from_path(path) if path.exists() else importlib.import_module(target)
    input_kind = getattr(module, "INPUT_KIND", None)
    if input_kind not in {"single", "double"}:
        raise ValueError(f"Pipeline {target!r} must declare INPUT_KIND = 'single' or 'double'.")
    return PipelineSpec(path.stem if path.exists() else target.rsplit(".", 1)[-1], input_kind, module)


def _load_module_from_path(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline file does not exist: {path}")
    spec = importlib.util.spec_from_file_location(f"sarpyx_external_pipeline_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pipeline file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_pipeline_target(
    target: str,
    *,
    output_dir,
    input_path=None,
    master=None,
    slave=None,
    params: dict[str, Any] | None = None,
    gpt_path=None,
    gpt_memory="16G",
    gpt_parallelism=6,
    gpt_timeout=None,
    gpt_cache_size="8G",
    snap_userdir=None,
    product_wkt=None,
    grid_path=None,
    cuts_outdir=None,
    tile_writer="zarr",
    keep_intermediate=True,
):
    spec = load_pipeline(target)
    params = dict(params or {})
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_snap(gpt_path, snap_userdir)
    if spec.input_kind == "single":
        return _run_single(spec, input_path, output_dir, params, gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size, product_wkt, grid_path, cuts_outdir, tile_writer, keep_intermediate)
    return _run_double(spec, master, slave, output_dir, params, gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size, product_wkt, grid_path, cuts_outdir, tile_writer)


def _configure_snap(gpt_path, snap_userdir) -> None:
    config.GPT_PATH = config.resolve_gpt_path(gpt_path)
    if snap_userdir:
        config.SNAP_USERDIR = str(Path(snap_userdir).expanduser())


def _run_single(spec, input_path, output_dir, params, gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size, product_wkt, grid_path, cuts_outdir, tile_writer, keep_intermediate):
    if input_path is None:
        raise ValueError(f"Pipeline {spec.name!r} requires --input.")
    input_path = Path(input_path).expanduser()
    if spec.runner is not None:
        cuts_outdir = Path(cuts_outdir).expanduser() if cuts_outdir else output_dir / "tiles"
        params.setdefault("pre_write_hook", _pre_write_hook(spec, input_path, tile_writer))
        params.setdefault("product_name", product_output_name(input_path))
        return spec.runner(
            input_path,
            output_dir,
            gpt_memory=gpt_memory,
            gpt_parallelism=gpt_parallelism,
            gpt_timeout=gpt_timeout,
            gpt_cache_size=gpt_cache_size,
            product_wkt=product_wkt,
            grid_path=Path(grid_path).expanduser() if grid_path else None,
            cuts_outdir=cuts_outdir,
            product_mode=spec.product_mode,
            tile_writer=tile_writer,
            keep_intermediate=keep_intermediate,
            **params,
        )
    recipe = _recipe(spec.module, params)
    gpt_kwargs = dict(gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout, gpt_cache_size=gpt_cache_size)
    metadata = _tiling_metadata(spec, product_wkt, grid_path, cuts_outdir, tile_writer)
    ctx = make_context(input_path, output_dir, getattr(spec.module, "OUTPUT_FORMAT", "BEAM-DIMAP"), gpt_kwargs, metadata=metadata)
    run_steps(ctx, recipe)
    return ctx.current_product


def _run_double(spec, master, slave, output_dir, params, gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size, product_wkt, grid_path, cuts_outdir, tile_writer):
    if master is None or slave is None:
        raise ValueError(f"Pipeline {spec.name!r} requires --master and --slave.")
    master_path = Path(master).expanduser()
    if product_wkt is None and grid_path is not None and cuts_outdir is not None:
        product_wkt = resolve_product_wkt(type("Args", (), {"product_wkt": None})(), master_path, spec.product_mode or "CUSTOM")
    cuts_outdir = Path(cuts_outdir).expanduser() if cuts_outdir else output_dir / "tiles"
    return run_insar_pipeline(
        master_path,
        slave,
        output_dir,
        recipe=_recipe(spec.module, params),
        gpt_path=config.GPT_PATH,
        memory=gpt_memory,
        parallelism=gpt_parallelism,
        timeout=gpt_timeout,
        snap_userdir=config.SNAP_USERDIR,
        cache_size=gpt_cache_size,
        metadata=_tiling_metadata(
            spec,
            product_wkt,
            grid_path,
            cuts_outdir,
            tile_writer,
            pre_write_hook=_pre_write_hook(spec, master_path, tile_writer),
            product_name=product_output_name(master_path),
        ),
    )


def _recipe(module: ModuleType, params: dict[str, Any]):
    steps = getattr(module, "steps", None)
    if steps is None:
        raise ValueError(f"Pipeline module {module.__name__!r} must define steps(**kwargs).")
    return steps(**params)


def _tiling_metadata(spec, product_wkt, grid_path, cuts_outdir, tile_writer, pre_write_hook=None, product_name=None):
    if product_wkt is None or grid_path is None or cuts_outdir is None:
        return {}
    metadata = {
        "product_wkt": product_wkt,
        "grid_path": Path(grid_path).expanduser(),
        "cuts_outdir": Path(cuts_outdir).expanduser(),
        "product_mode": getattr(spec.module, "PRODUCT_MODE", spec.product_mode or "CUSTOM"),
        "tile_writer": tile_writer,
    }
    if pre_write_hook is not None:
        metadata["pre_write_hook"] = pre_write_hook
    if product_name is not None:
        metadata["product_name"] = product_name
    return metadata


def _pre_write_hook(spec, product_path, tile_writer):
    product_mode = getattr(spec.module, "PRODUCT_MODE", spec.product_mode)
    if tile_writer != "zarr" or product_mode not in {"S1TOPS", "S1STRIP", "S1INSAR"}:
        return None
    subap_features = SubapFeatureConfig(enabled=False) if product_mode == "S1INSAR" else None
    return make_worldsar_zarr_tile_hook(
        product_path,
        product_mode=product_mode,
        product_name=product_output_name(product_path),
        subap_features=subap_features,
    )
