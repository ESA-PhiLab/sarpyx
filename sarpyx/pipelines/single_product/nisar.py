"""NISAR WorldSAR recipe."""

from __future__ import annotations

from sarpyx.snapflow.runtime import PipelineStep

INPUT_KIND = "single"
PRODUCT_MODE = "NISAR"


def steps():
    return [PipelineStep("WorldSARTiling", {}, "tiling")]
