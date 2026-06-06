"""BIOMASS WorldSAR recipe."""

from __future__ import annotations

from sarpyx.snapflow.runtime import PipelineStep

INPUT_KIND = "single"
PRODUCT_MODE = "BM"


def steps():
    return [
        PipelineStep("Write", {}, "write"),
        PipelineStep("WorldSARTiling", {"intermediate_ref": "write"}, "tiling"),
    ]
