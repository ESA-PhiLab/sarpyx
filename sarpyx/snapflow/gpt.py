"""GPT construction helpers for WorldSAR workflows."""

from __future__ import annotations

from pathlib import Path

from sarpyx.snapflow.engine import GPT
from sarpyx.snapflow import config


def build_gpt_kwargs(gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size=None) -> dict:
    pairs = [
        ("memory", gpt_memory),
        ("parallelism", gpt_parallelism),
        ("timeout", gpt_timeout),
        ("cache_size", gpt_cache_size),
    ]
    return {key: value for key, value in pairs if value}


def create_gpt_operator(
    product_path,
    output_dir,
    output_format,
    gpt_memory=None,
    gpt_parallelism=None,
    gpt_timeout=None,
    gpt_cache_size=None,
):
    return GPT(
        product=product_path,
        outdir=output_dir,
        format=output_format,
        gpt_path=config.GPT_PATH,
        snap_userdir=config.SNAP_USERDIR,
        **build_gpt_kwargs(gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size),
    )


def run_gpt_op(
    product_path,
    output_dir,
    output_format,
    op_name,
    gpt_memory=None,
    gpt_parallelism=None,
    gpt_timeout=None,
    gpt_cache_size=None,
    **op_kwargs,
) -> Path:
    op = create_gpt_operator(product_path, output_dir, output_format, gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size)
    result = getattr(op, op_name)(**op_kwargs)
    if result is None:
        error_summary = op.last_error_summary()
        timeout_hint = ""
        if "timed out" in error_summary.lower():
            timeout_hint = " Increase --gpt-timeout (e.g. 14400) or disable it with --gpt-timeout 0."
        raise RuntimeError(f"GPT {op_name} failed: {error_summary}{timeout_hint}")
    output_path = Path(result)
    if not output_path.exists():
        raise RuntimeError(f"GPT {op_name} reported {output_path} but output file is missing.")
    return output_path
