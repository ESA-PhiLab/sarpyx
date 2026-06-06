"""Small execution runtime for package-owned WorldSAR pipeline definitions."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from xml.etree.ElementTree import ParseError

from sarpyx.snapflow.dim_updater import update_dim_add_bands_from_data_dir
from sarpyx.snapflow.dimap import materialized_band_names
from sarpyx.snapflow.gpt import create_gpt_operator
from sarpyx.snapflow.merge import merge_iq_into_pdec

ALL_SOURCE_BANDS = "__all_source_bands__"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    save_as: str | None = None


@dataclass
class PipelineContext:
    original_product: Any
    current_product: Any
    output_dir: Path
    op: Any
    saved: dict[str, Any]
    metadata: dict[str, Any]
    gpt_kwargs: dict[str, Any]


CreateGptOperator = Callable[..., Any]


def make_context(
    product_path,
    output_dir,
    output_format: str,
    gpt_kwargs: Mapping[str, Any],
    *,
    create_operator: CreateGptOperator = create_gpt_operator,
    metadata: Mapping[str, Any] | None = None,
) -> PipelineContext:
    op = create_operator(product_path, output_dir, output_format, **dict(gpt_kwargs))
    return PipelineContext(
        original_product=product_path,
        current_product=product_path,
        output_dir=Path(output_dir),
        op=op,
        saved={},
        metadata=dict(metadata or {}),
        gpt_kwargs=dict(gpt_kwargs),
    )


def run_steps(ctx: PipelineContext, steps: Sequence[PipelineStep]) -> PipelineContext:
    for step in steps:
        result = run_step(ctx, step)
        if step.save_as:
            ctx.saved[step.save_as] = result
        if result is not None:
            ctx.current_product = result
    return ctx


def run_step(ctx: PipelineContext, step: PipelineStep):
    try:
        handler = STEP_REGISTRY[step.name]
    except KeyError as exc:
        raise KeyError(f"Unknown WorldSAR pipeline step: {step.name}") from exc
    return handler(ctx, **dict(step.params))


def _op_result(ctx: PipelineContext):
    return getattr(ctx.op, "prod_path", ctx.current_product)


def _last_error(ctx: PipelineContext) -> str:
    return ctx.op.last_error_summary() if hasattr(ctx.op, "last_error_summary") else "unknown error"


def _require(result, ctx: PipelineContext, label: str):
    if result is None:
        raise RuntimeError(f"{label} failed: {_last_error(ctx)}")
    return result


def _ref(ctx: PipelineContext, name: str):
    return ctx.saved[name]


def _refs(ctx: PipelineContext, names: Sequence[str]) -> list[Any]:
    return [_ref(ctx, name) for name in names]


def _dimap_data_dir(dim_path: Path) -> Path:
    return dim_path.with_suffix(".data")


def _cleanup_saved_dimap_products(ctx: PipelineContext, keep_products: Sequence[Any]) -> None:
    output_dir = ctx.output_dir.resolve()
    keep = {Path(path).resolve() for path in keep_products if path is not None}
    for product in list(ctx.saved.values()):
        try:
            dim_path = Path(product)
        except TypeError:
            continue
        if dim_path.suffix.lower() != ".dim":
            continue
        resolved = dim_path.resolve()
        if resolved in keep or resolved.parent != output_dir:
            continue
        data_dir = _dimap_data_dir(resolved)
        if data_dir.exists():
            shutil.rmtree(data_dir)
        if resolved.exists():
            resolved.unlink()


def step_apply_orbit_file(
    ctx: PipelineContext,
    orbit_type: str = "Sentinel Precise (Auto Download)",
    orbit_continue_on_fail: bool = False,
):
    result = ctx.op.ApplyOrbitFile(orbit_type=orbit_type, continue_on_fail=orbit_continue_on_fail)
    if result is not None:
        return result
    error_summary = _last_error(ctx)
    normalized = error_summary.lower()
    offline_markers = (
        "network is unreachable",
        "unable to connect to http://step.esa.int/auxdata/orbits/",
        "unable to connect to https://step.esa.int/auxdata/orbits/",
    )
    missing_markers = ("no valid orbit file found", "orbit files may be downloaded from copernicus dataspaces")
    recoverable = any(marker in normalized for marker in offline_markers) or all(
        marker in normalized for marker in missing_markers
    )
    if orbit_continue_on_fail or recoverable:
        print(f"WARNING: Apply-Orbit-File failed but continuing without orbit correction: {error_summary}")
        return _op_result(ctx)
    raise RuntimeError(f"Apply-Orbit-File failed: {error_summary}")


def step_gpt_method(ctx: PipelineContext, method: str, label: str | None = None, **kwargs):
    return _require(getattr(ctx.op, method)(**kwargs), ctx, label or method)


def step_do_subaps(ctx: PipelineContext, dim_ref: str | None = None, safe_ref: str | None = None, **kwargs):
    if dim_ref:
        kwargs["dim_path"] = _ref(ctx, dim_ref)
    elif "dim_path" not in kwargs:
        kwargs["dim_path"] = _op_result(ctx)
    if safe_ref:
        kwargs["safe_path"] = _ref(ctx, safe_ref)
    elif "safe_path" not in kwargs:
        kwargs["safe_path"] = ctx.original_product
    if ctx.metadata.get("cleanup_before_subaps") and not ctx.metadata.get("keep_intermediate", True):
        _cleanup_saved_dimap_products(ctx, [kwargs["dim_path"]])
    result = ctx.op.do_subaps(**kwargs)
    return result if result is not None else _op_result(ctx)


def step_merge_iq_into_pdec(
    ctx: PipelineContext,
    src_ref: str,
    pdec_ref: str,
    is_tops: bool,
    overwrite_copied_files: bool = False,
    backup: bool = False,
    fallback_steps: Sequence[PipelineStep] | None = None,
):
    merge_func = ctx.metadata.get("merge_iq_into_pdec", merge_iq_into_pdec)
    try:
        merge_func(
            src_dim=_ref(ctx, src_ref),
            pdec_dim=_ref(ctx, pdec_ref),
            is_tops=is_tops,
            overwrite_copied_files=overwrite_copied_files,
            backup=backup,
        )
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != "merge_iq_into_pdec":
            raise
        if not fallback_steps:
            raise RuntimeError(
                "merge_iq_into_pdec module is required for TOPS flow. "
                "TOPS fallback to DIM metadata rewrite is disabled to avoid malformed DEB metadata."
            )
        run_steps(ctx, fallback_steps)
    return _ref(ctx, pdec_ref)


def step_update_dim_from_data_dir(ctx: PipelineContext, dim_ref: str, save_ref: str | None = None, verbose: bool = False):
    result = update_dim_add_bands_from_data_dir(_ref(ctx, dim_ref), verbose=verbose)
    if save_ref:
        ctx.saved[save_ref] = result
    return result


def step_band_merge(
    ctx: PipelineContext,
    source_refs: Sequence[str],
    output_name_ref: str | None = None,
    output_name: str | None = None,
):
    if output_name_ref:
        output_name = f"{Path(_ref(ctx, output_name_ref)).stem}_MERGED"
    result = ctx.op.BandMerge(source_products=_refs(ctx, source_refs), output_name=output_name)
    return _require(result, ctx, "BandMerge")


def step_write(ctx: PipelineContext, **kwargs):
    return _require(ctx.op.write(**kwargs), ctx, "Write")


def step_terrain_correction(ctx: PipelineContext, **kwargs):
    kwargs = dict(kwargs)
    source_product = _op_result(ctx)
    include_subap_source_bands = bool(kwargs.pop("include_subap_source_bands", False))
    if kwargs.get("source_bands") == ALL_SOURCE_BANDS:
        # SNAP Terrain-Correction rejects cloned sub-aperture bands in sourceBands.
        try:
            source_bands = [
                name for name in materialized_band_names(Path(source_product))
                if include_subap_source_bands or "_SA" not in name
            ]
        except (FileNotFoundError, OSError, ParseError):
            source_bands = []
        if source_bands:
            kwargs["source_bands"] = source_bands
        else:
            kwargs.pop("source_bands", None)
    result = _require(ctx.op.TerrainCorrection(**kwargs), ctx, "Terrain Correction")
    runs = ctx.metadata.setdefault("terrain_correction_runs", {})
    runs[str(Path(result))] = {
        "source_product": str(source_product),
        "params": dict(kwargs),
        "output_dir": str(ctx.output_dir),
    }
    return result


def step_worldsar_tiling(
    ctx: PipelineContext,
    intermediate_ref: str | None = None,
    swath: str | None = None,
    collect: bool = False,
):
    from sarpyx.snapflow.tiling_runtime import run_tiling_step

    intermediate = _ref(ctx, intermediate_ref) if intermediate_ref else _op_result(ctx)
    return run_tiling_step(ctx, intermediate, swath=swath or ctx.metadata.get("swath"), collect=collect)


STEP_REGISTRY = {
    "ApplyOrbitFile": step_apply_orbit_file,
    "Calibration": lambda ctx, **kw: step_gpt_method(ctx, "Calibration", "Calibration", **kw),
    "TopsarDerampDemod": lambda ctx, **kw: step_gpt_method(ctx, "TopsarDerampDemod", "TOPSAR-DerampDemod", **kw),
    "Deburst": lambda ctx, **kw: step_gpt_method(ctx, "Deburst", "TOPSAR-Deburst", **kw),
    "TerrainCorrection": step_terrain_correction,
    "TopsarSplit": lambda ctx, **kw: step_gpt_method(ctx, "TopsarSplit", "TOPSAR-Split", **kw),
    "polarimetric_decomposition": lambda ctx, **kw: step_gpt_method(
        ctx, "polarimetric_decomposition", "Polarimetric decomposition", **kw
    ),
    "do_subaps": step_do_subaps,
    "merge_iq_into_pdec": step_merge_iq_into_pdec,
    "update_dim_add_bands_from_data_dir": step_update_dim_from_data_dir,
    "BandMerge": step_band_merge,
    "Write": step_write,
    "WorldSARTiling": step_worldsar_tiling,
}
