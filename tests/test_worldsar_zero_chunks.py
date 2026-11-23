"""WorldSAR stores even fill chunks so missing writes remain detectable."""
from pathlib import Path

import numpy as np
import zarr

from sarpyx.hooks.worldsar import make_worldsar_zarr_tile_hook
from sarpyx.snapflow.tile_writers import TilePayload, write_tile_payloads
from sarpyx.snapflow.raster_quality import summarize_zarr_raster_quality


def test_worldsar_fill_chunks_are_stored(tmp_path: Path):
    target = tmp_path / "cell.zarr"
    payload = TilePayload(
        tile_name="cell", output_path=target,
        arrays={"i_ifg_VV_dates": np.zeros((256, 256), dtype=np.float32)},
        abstract_attrs={}, band_attrs={},
    )
    write_tile_payloads(
        [payload], "zarr",
        make_worldsar_zarr_tile_hook("S1_TEST.SAFE", product_mode="S1INSAR"),
    )
    array = zarr.open_group(str(target), mode="r")["bands"]["i_ifg_VV_dates"]
    assert array.nchunks_initialized == array.nchunks == 4
    assert not np.any(array[:])


def test_insar_ancillary_data_cannot_hide_empty_signal(tmp_path: Path):
    target = tmp_path / "empty.zarr"
    root = zarr.open_group(str(target), mode="w")
    root.attrs["product_mode"] = "S1INSAR"
    bands = root.create_group("bands")
    for name in ("i_ifg_VV_dates", "q_ifg_VV_dates", "coh_VV_dates"):
        bands.create_array(name, data=np.zeros((16, 16), dtype=np.float32))
    bands.create_array("elevation", data=np.ones((16, 16), dtype=np.float32))
    result = summarize_zarr_raster_quality(target)
    assert result["raster_data_ok"] is False
    assert result["valid_pixels"] == 0
