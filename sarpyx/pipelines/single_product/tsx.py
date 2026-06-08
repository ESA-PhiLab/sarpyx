"""TerraSAR-X/TanDEM-X WorldSAR recipe."""

from __future__ import annotations

from pathlib import Path

from sarpyx.snapflow.runtime import PipelineStep

DEFAULT_MAP_PROJECTION = "AUTO:42001"
INPUT_KIND = "single"
PRODUCT_MODE = "TSX"
DEFAULT_PIXEL_SPACING_M = 5.0


def steps(geocoded: bool = False, output_complex: bool = True, output_file: Path | None = None):
    if geocoded:
        return [
            PipelineStep("Write", {"output_file": output_file, "format_name": "BEAM-DIMAP"}, "write"),
            PipelineStep("WorldSARTiling", {"intermediate_ref": "write"}, "tiling"),
        ]
    return [
        PipelineStep("Calibration", {"output_complex": output_complex}, "cal"),
        PipelineStep(
            "TerrainCorrection",
            {"map_projection": DEFAULT_MAP_PROJECTION, "pixel_spacing_in_meter": DEFAULT_PIXEL_SPACING_M, "output_complex": output_complex},
            "tc",
        ),
        PipelineStep("WorldSARTiling", {"intermediate_ref": "tc"}, "tiling"),
    ]
