"""Runtime for declared InSAR pipeline recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from sarpyx.pipelines.double_product import s1_insar
from sarpyx.snapflow.engine import GPT
from sarpyx.snapflow.runtime import PipelineStep


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


def run_insar_step(ctx: InSARContext, step: PipelineStep):
    params = dict(step.params)
    if step.name == "WorldSARTiling":
        from sarpyx.snapflow.tiling_runtime import run_tiling_step

        intermediate_ref = params.pop("intermediate_ref", None)
        swath = params.pop("swath", None)
        collect = params.pop("collect", False)
        intermediate = _source(ctx, intermediate_ref) if intermediate_ref else ctx.saved.get("terrain_corrected")
        return run_tiling_step(ctx, intermediate, swath=swath or ctx.metadata.get("swath"), collect=collect)
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
    for step in recipe or s1_insar.steps():
        result = run_insar_step(ctx, step)
        if step.save_as:
            ctx.saved[step.save_as] = result
    return ctx.saved.get("terrain_corrected")
