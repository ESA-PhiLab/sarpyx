"""Runtime for declared InSAR pipeline recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from xml.etree.ElementTree import ParseError

from sarpyx.pipelines.double_product import s1_insar
from sarpyx.snapflow.dimap import get_data_dir_from_dim, materialized_band_names, spectral_band_names
from sarpyx.snapflow.engine import GPT
from sarpyx.snapflow.runtime import PipelineStep
from sarpyx.snapflow.stamps import run_stamps_prep


@dataclass
class InSARContext:
    master: Path
    slave: Path
    outdir: Path
    gpt_path: str
    memory: str | None = None
    parallelism: int | None = 14
    cache_size: str | None = None
    timeout: int | None = 7200
    snap_userdir: Path | None = None
    format: str = "BEAM-DIMAP"
    saved: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    gpt_kwargs: dict[str, Any] = field(default_factory=dict)
    original_product: Path | None = None


def _source(ctx: InSARContext, name: str):
    if name == "master":
        return ctx.master
    if name == "slave":
        return ctx.slave
    return ctx.saved[name]


def _gpt(ctx: InSARContext, product, outdir: str):
    return GPT(
        product=product,
        outdir=ctx.outdir / outdir,
        format=ctx.format,
        gpt_path=ctx.gpt_path,
        memory=ctx.memory,
        parallelism=ctx.parallelism,
        cache_size=ctx.cache_size,
        timeout=ctx.timeout,
        snap_userdir=ctx.snap_userdir,
    )


def _require(result, gpt: GPT, label: str):
    if result is None:
        raise RuntimeError(f"{label} failed: {gpt.last_error_summary()}")
    return result


def _clean_params(params: dict[str, Any], *remove: str) -> dict[str, Any]:
    return {key: value for key, value in params.items() if key not in remove and value is not None}


def _resolve_outdir_path(ctx: InSARContext, value: str | Path | None, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else ctx.outdir / path


def _dimap_bands(path: Path) -> set[str]:
    try:
        return set(spectral_band_names(path)) | set(materialized_band_names(path))
    except (FileNotFoundError, OSError, ParseError, RuntimeError) as exc:
        raise RuntimeError(f"Could not read BEAM-DIMAP bands from {path}: {exc}") from exc


def _has_phase_ifg_band(names: set[str]) -> bool:
    return any(name.lower().startswith("phase_ifg") for name in names)


def _has_elevation_band(names: set[str]) -> bool:
    return any(name.lower() == "elevation" or name.lower().startswith("elevation_") for name in names)


def _has_lat_lon_bands(names: set[str]) -> bool:
    lowered = {name.lower() for name in names}
    has_lat = any("lat" in name and ("orthorectified" in name or name in {"lat", "latitude"}) for name in lowered)
    has_lon = any("lon" in name and ("orthorectified" in name or name in {"lon", "longitude"}) for name in lowered)
    return has_lat and has_lon


def _validate_stamps_product(path: Path, label: str, require_data_dir: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} product path does not exist: {path}")
    if require_data_dir and path.suffix.lower() == ".dim":
        data_dir = get_data_dir_from_dim(path)
        if not data_dir.exists():
            raise FileNotFoundError(f"{label} BEAM-DIMAP data directory does not exist: {data_dir}")


def _validate_stamps_inputs(ctx: InSARContext, params: dict[str, Any]):
    coreg_ref = params.pop("coreg_ref")
    ifg_ref = params.pop("ifg_ref")
    require_data_dir = params.pop("require_data_dir", True)
    coreg_path = Path(_source(ctx, coreg_ref))
    ifg_path = Path(_source(ctx, ifg_ref))
    _validate_stamps_product(coreg_path, "Coregistered SLC", require_data_dir)
    _validate_stamps_product(ifg_path, "Interferogram", require_data_dir)

    ifg_bands = _dimap_bands(ifg_path)
    missing = []
    if params.pop("require_phase", True) and not _has_phase_ifg_band(ifg_bands):
        missing.append("Phase_ifg_*")
    if params.pop("require_elevation", True) and not _has_elevation_band(ifg_bands):
        missing.append("elevation")
    if params.pop("require_lat_lon", True) and not _has_lat_lon_bands(ifg_bands):
        missing.append("orthorectified latitude/longitude")
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"StaMPS IFG product {ifg_path} is missing required bands: {missing_text}")
    return {"coreg": coreg_path, "ifg": ifg_path, "ifg_bands": sorted(ifg_bands)}


def _run_stamps_export(ctx: InSARContext, params: dict[str, Any]):
    outdir = params.pop("outdir", "export")
    coreg_ref = params.pop("coreg_ref")
    ifg_ref = params.pop("ifg_ref")
    target_folder = _resolve_outdir_path(ctx, params.pop("target_folder", None), "stamps")
    coreg_product = _source(ctx, coreg_ref)
    ifg_product = _source(ctx, ifg_ref)
    gpt = _gpt(ctx, coreg_product, outdir)
    result = gpt.stamps_export_pair(
        coreg_product=coreg_product,
        ifg_product=ifg_product,
        target_folder=target_folder,
        **_clean_params(params),
    )
    return _require(result, gpt, "StaMPS Export")


def _run_stamps_prep(ctx: InSARContext, params: dict[str, Any]):
    target_folder = _resolve_outdir_path(ctx, params.pop("target_folder", None), "stamps")
    return run_stamps_prep(
        target_folder=target_folder,
        master_product=ctx.master,
        timeout=ctx.timeout,
        **_clean_params(params),
    )


def run_insar_step(ctx: InSARContext, step: PipelineStep):
    params = dict(step.params)
    if step.name == "WorldSARTiling":
        from sarpyx.snapflow.tiling_runtime import run_tiling_step

        intermediate_ref = params.pop("intermediate_ref", None)
        swath = params.pop("swath", None)
        collect = params.pop("collect", False)
        intermediate = _source(ctx, intermediate_ref) if intermediate_ref else ctx.saved.get("terrain_corrected")
        return run_tiling_step(ctx, intermediate, swath=swath or ctx.metadata.get("swath"), collect=collect)
    if step.name == "ValidateStampsInputs":
        return _validate_stamps_inputs(ctx, params)
    if step.name == "StampsExport":
        return _run_stamps_export(ctx, params)
    if step.name == "StampsPrep":
        return _run_stamps_prep(ctx, params)
    outdir = params.pop("outdir", "pair")
    if step.name == "TopsarCoregistration":
        master_ref = params.pop("master_ref")
        slave_ref = params.pop("slave_ref")
        gpt = _gpt(ctx, _source(ctx, master_ref), outdir)
        result = gpt.topsar_coregistration(
            master_product=_source(ctx, master_ref),
            slave_product=_source(ctx, slave_ref),
            **_clean_params(params),
        )
        return _require(result, gpt, "TOPSAR coregistration")
    source_ref = params.pop("source_ref")
    gpt = _gpt(ctx, _source(ctx, source_ref), outdir)
    if step.name == "TopsarSplit":
        return _require(gpt.topsar_split(**_clean_params(params)), gpt, "TOPSAR-Split")
    if step.name == "ApplyOrbitFile":
        return _require(gpt.apply_orbit_file(**_clean_params(params)), gpt, "Apply-Orbit-File")
    if step.name == "Deburst":
        return _require(gpt.deburst(**_clean_params(params)), gpt, "TOPSAR-Deburst")
    if step.name == "AddElevation":
        return _require(gpt.add_elevation(**_clean_params(params)), gpt, "AddElevation")
    if step.name == "AddStampsLatLonBands":
        return _require(gpt.stamps_lat_lon_bands(**_clean_params(params)), gpt, "StaMPS lat/lon band creation")
    if step.name == "Interferogram":
        return _require(gpt.interferogram(**_clean_params(params)), gpt, "Interferogram")
    if step.name == "TopoPhaseRemoval":
        return _require(gpt.topo_phase_removal(**_clean_params(params)), gpt, "TopoPhaseRemoval")
    if step.name == "Subset":
        return _require(gpt.subset(**_clean_params(params)), gpt, "Subset")
    if step.name == "TerrainCorrection":
        result = _require(gpt.terrain_correction(**_clean_params(params)), gpt, "Terrain-Correction")
        runs = ctx.metadata.setdefault("terrain_correction_runs", {})
        runs[str(Path(result))] = {
            "source_product": str(_source(ctx, source_ref)),
            "params": _clean_params(params),
            "output_dir": str(ctx.outdir / outdir),
        }
        return result
    raise KeyError(f"Unknown InSAR pipeline step: {step.name}")


def run_insar_pipeline(
    master,
    slave,
    outdir,
    *,
    recipe: Sequence[PipelineStep] | None = None,
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    cache_size: str | None = None,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
    format: str = "BEAM-DIMAP",
    metadata: dict[str, Any] | None = None,
):
    ctx = InSARContext(
        master=Path(master),
        slave=Path(slave),
        outdir=Path(outdir),
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        cache_size=cache_size,
        timeout=timeout,
        snap_userdir=Path(snap_userdir) if snap_userdir else None,
        format=format,
        metadata=dict(metadata or {}),
        gpt_kwargs={
            "gpt_memory": memory,
            "gpt_parallelism": parallelism,
            "gpt_timeout": timeout,
            "gpt_cache_size": cache_size,
        },
        original_product=Path(master),
    )
    ctx.saved.update({"master": ctx.master, "slave": ctx.slave})
    last_result = None
    for step in recipe or s1_insar.steps():
        result = run_insar_step(ctx, step)
        if result is not None:
            last_result = result
        if step.save_as:
            ctx.saved[step.save_as] = result
    return ctx.saved.get("terrain_corrected") or last_result
