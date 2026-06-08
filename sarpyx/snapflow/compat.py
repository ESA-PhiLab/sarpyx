"""Compatibility facade for historical WorldSAR imports."""

from __future__ import annotations

from pathlib import Path

from sarpyx.pipelines.single_product.s1_tops import steps as sentinel_tops_steps
from sarpyx.snapflow.config import DEFAULT_ORBIT_TYPE
from sarpyx.snapflow.gpt import create_gpt_operator as _create_gpt_operator
from sarpyx.snapflow.merge import merge_iq_into_pdec
from sarpyx.snapflow.preprocessing import (
    run_biomass_pipeline,
    run_nisar_pipeline,
    run_sentinel_strip_pipeline,
    run_sentinel_tops_pipeline,
    run_tsx_csg_pipeline,
)
from sarpyx.snapflow.runtime import PipelineContext, run_steps, step_apply_orbit_file


def _apply_sentinel_orbit_file(op, orbit_type=DEFAULT_ORBIT_TYPE, orbit_continue_on_fail=False):
    ctx = PipelineContext(None, getattr(op, "prod_path", None), None, op, {}, {}, {})
    return step_apply_orbit_file(ctx, orbit_type=orbit_type, orbit_continue_on_fail=orbit_continue_on_fail)


def _sentinel_post_chain(
    op,
    product_path,
    orbit_type=DEFAULT_ORBIT_TYPE,
    orbit_continue_on_fail=False,
    sentinel_tc_source_band=None,
    sentinel_subap_decompositions=None,
):
    ctx = PipelineContext(product_path, getattr(op, "prod_path", product_path), None, op, {}, {"merge_iq_into_pdec": merge_iq_into_pdec}, {})
    run_steps(
        ctx,
        sentinel_tops_steps(
            orbit_type=orbit_type,
            orbit_continue_on_fail=orbit_continue_on_fail,
            sentinel_tc_source_band=sentinel_tc_source_band,
            sentinel_subap_decompositions=sentinel_subap_decompositions,
        )[1:],
    )
    return op.prod_path


def pipeline_sentinel(product_path, output_dir, is_TOPS=False, **kwargs):
    impl = run_sentinel_tops_pipeline if is_TOPS else run_sentinel_strip_pipeline
    return impl(product_path, output_dir, create_operator=_create_gpt_operator, merge_func=merge_iq_into_pdec, **kwargs)


def pipeline_tsx_csg(product_path, output_dir, **kwargs):
    return run_tsx_csg_pipeline(product_path, output_dir, create_operator=_create_gpt_operator, **kwargs)


def pipeline_biomass(product_path, output_dir, **kwargs):
    return run_biomass_pipeline(product_path, output_dir, create_operator=_create_gpt_operator, **kwargs)


def pipeline_nisar(product_path, output_dir=None, **kwargs):
    return run_nisar_pipeline(product_path, output_dir, **kwargs)
