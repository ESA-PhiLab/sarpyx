from __future__ import annotations

from pathlib import Path

import numpy as np
import pyproj
import pytest
import zarr

from sarpyx.hooks.worldsar import make_worldsar_zarr_tile_hook, validate_worldsar_zarr_tile, validate_worldsar_zarr_tile_group
from sarpyx.snapflow.h5_quality import summarize_zarr_raster_quality
from sarpyx.snapflow.tile_writers import TilePayload, write_tile_payloads
from sarpyx.snapflow.tiling import _cut_single_tile, _validate_tile_result


def test_worldsar_zarr_hook_writes_minimal_metadata_and_128_chunks(tmp_path: Path) -> None:
    payload = TilePayload(
        tile_name="tile-a",
        output_path=tmp_path / "tile-a.zarr",
        arrays={"Sigma0_VV": np.ones((256, 256), dtype=np.float32)},
        abstract_attrs={
            "PASS": "ASCENDING",
            "mds1_tx_rx_polar": "VV",
            "MISSION": "S1A",
            "first_line_time": "2026-01-01T00:00:00Z",
            "antenna_pointing": "right",
        },
        band_attrs={"Sigma0_VV": {"unit": "linear", "CLASS": "large-unused-value"}},
        crs_wkt=pyproj.CRS.from_epsg(32633).to_wkt(),
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 4600000.0),
    )
    hook = make_worldsar_zarr_tile_hook("S1_TEST.SAFE", product_mode="S1TOPS")

    write_tile_payloads([payload], "zarr", hook)

    root = zarr.open((tmp_path / "tile-a.zarr").as_posix(), mode="r")
    assert root.attrs["pass_direction"] == "ASC"
    assert root.attrs["polarizations"] == ["VV"]
    assert root.attrs["chunk_size"] == [128, 128]
    assert "metadata" not in root
    assert root["bands"]["Sigma0_VV"].chunks == (128, 128)
    assert dict(root["bands"]["Sigma0_VV"].attrs) == {"polarization": "VV", "unit": "linear"}


def test_cut_single_tc_tile_to_zarr_from_dimap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sarpyx.snapflow import tiling as tiling_mod

    dim_path = _write_synthetic_tc_dimap(tmp_path / "TC.dim")
    cuts_dir = tmp_path / "tiles" / "S1_TEST.SAFE"
    rect = {
        "BL": {
            "type": "Feature",
            "properties": {"name": "tile-tc", "epsg": "EPSG:32633"},
            "geometry": {"type": "Point", "coordinates": [15.0, 0.0]},
        }
    }
    monkeypatch.setattr(tiling_mod, "grid_cell_utm_bbox", lambda *_args, **_kwargs: (500000.0, 0.0, 501280.0, 1280.0))
    hook = make_worldsar_zarr_tile_hook("S1_TEST.SAFE", product_mode="S1STRIP")

    result = _cut_single_tile(
        rect,
        dim_path,
        cuts_dir,
        "S1STRIP",
        gpt_memory=None,
        gpt_parallelism=None,
        gpt_timeout=None,
        tile_writer="zarr",
        pre_write_hook=hook,
    )

    tile_path = cuts_dir / "tile-tc.zarr"
    assert result["status"] == "success"
    assert tile_path.is_dir()
    root = zarr.open(tile_path.as_posix(), mode="r")
    assert root.attrs["pass_direction"] == "DESC"
    assert root.attrs["polarizations"] == ["VH"]
    assert root["bands"]["Sigma0_VH"].shape == (128, 128)
    assert root["bands"]["Sigma0_VH"].chunks == (128, 128)
    band_names = sorted(root["bands"].keys())
    assert "i_IW1_VH_SA1" not in band_names
    assert "q_IW1_VH_SA1" not in band_names
    assert "subap_coherence_VH_gamma12" in band_names
    assert "subap_coherence_VH_gamma_mean" in band_names
    assert "subap_covariance_IW1_VH_C1" not in band_names
    assert "subap_covariance_VH_C11" in band_names
    assert "subap_phase_variance_VH" in band_names
    assert root.attrs["subap_feature_bands"]
    zarr.create_group(store=(cuts_dir / "other-swath-tile.zarr").as_posix(), zarr_format=3, overwrite=True)
    group = validate_worldsar_zarr_tile_group(
        cuts_dir,
        {
            "name": "S1_TEST.SAFE",
            "report_path": tmp_path / "report.txt",
            "actual_tiles": ["tile-tc"],
            "expected_tiles": ["tile-tc"],
            "expected_tile_geometries": {"tile-tc": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]},
        },
        dim_path,
    )
    assert [result["tile"] for result in group["results"]] == ["tile-tc"]
    assert group["rows"][0]["PASS"] == "DESC"
    assert group["rows"][0]["mds1_tx_rx_polar"] == "VH"


def test_zarr_raster_quality_counts_nan_and_all_band_zero_pixels(tmp_path: Path) -> None:
    tile_path = tmp_path / "quality.zarr"
    _write_zarr_tile(
        tile_path,
        {
            "Band_A": np.array([[0.0, np.nan], [1.0, 2.0]], dtype=np.float32),
            "Band_B": np.array([[0.0, 3.0], [1.0, np.nan]], dtype=np.float32),
        },
    )

    summary = summarize_zarr_raster_quality(tile_path)

    assert summary["total_pixels"] == 4
    assert summary["zero_all_pixels"] == 1
    assert summary["nan_pixels"] == 2
    assert summary["nodata_pixels"] == 3
    assert summary["valid_pixels"] == 1
    assert summary["raster_data_ok"] is False


def test_validate_worldsar_zarr_tile_fails_all_nodata_tile(tmp_path: Path) -> None:
    tile_path = tmp_path / "empty.zarr"
    _write_zarr_tile(
        tile_path,
        {
            "Band_A": np.zeros((2, 2), dtype=np.float32),
            "Band_B": np.zeros((2, 2), dtype=np.float32),
        },
    )

    result = validate_worldsar_zarr_tile(tile_path)

    assert result["status"] == "failed"
    assert result["raster_data_ok"] is False
    assert result["valid_pixels"] == 0


def test_validate_tile_result_removes_partial_zarr_tile(tmp_path: Path) -> None:
    tile_path = tmp_path / "partial.zarr"
    _write_zarr_tile(
        tile_path,
        {
            "Band_A": np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32),
            "Band_B": np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32),
        },
    )

    result = _validate_tile_result("partial", tile_path, "tile cut", tile_writer="zarr")

    assert result["status"] == "partial"
    assert result["valid_pixels"] == 3
    assert not tile_path.exists()


def _write_zarr_tile(path: Path, arrays: dict[str, np.ndarray]) -> None:
    root = zarr.create_group(store=path.as_posix(), zarr_format=3, overwrite=True)
    root.attrs.update({"pass_direction": "ASC", "polarizations": ["VV"]})
    bands = root.create_group("bands")
    for name, data in arrays.items():
        bands.create_array(name, data=data, chunks=tuple(min(128, dim) for dim in data.shape))


def _write_synthetic_tc_dimap(dim_path: Path) -> Path:
    data_dir = dim_path.with_suffix(".data")
    data_dir.mkdir(parents=True)
    data = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    _write_envi_band(data_dir / "Sigma0_VH.img", data)
    _write_envi_band(data_dir / "i_IW1_VH_SA1.img", np.ones((128, 128), dtype=np.float32))
    _write_envi_band(data_dir / "q_IW1_VH_SA1.img", np.full((128, 128), 0.25, dtype=np.float32))
    _write_envi_band(data_dir / "i_IW1_VH_SA2.img", np.full((128, 128), 0.5, dtype=np.float32))
    _write_envi_band(data_dir / "q_IW1_VH_SA2.img", np.full((128, 128), 0.75, dtype=np.float32))

    crs_wkt = pyproj.CRS.from_epsg(32633).to_wkt()
    band_xml = "\n".join(_spectral_band_xml(name) for name in ("Sigma0_VH", "i_IW1_VH_SA1", "q_IW1_VH_SA1", "i_IW1_VH_SA2", "q_IW1_VH_SA2"))
    data_xml = "\n".join(
        f'<Data_File><BAND_INDEX>{index}</BAND_INDEX><DATA_FILE_PATH href="./TC.data/{name}.hdr" /></Data_File>'
        for index, name in enumerate(("Sigma0_VH", "i_IW1_VH_SA1", "q_IW1_VH_SA1", "i_IW1_VH_SA2", "q_IW1_VH_SA2"))
    )
    dim_path.write_text(
        f"""<Dimap_Document>
  <RasterDataNode>
    <Raster_Dimensions><NCOLS>128</NCOLS><NROWS>128</NROWS></Raster_Dimensions>
  </RasterDataNode>
  <Coordinate_Reference_System><WKT>{crs_wkt}</WKT></Coordinate_Reference_System>
  <IMAGE_TO_MODEL_TRANSFORM>10.0,0.0,0.0,-10.0,500000.0,1280.0</IMAGE_TO_MODEL_TRANSFORM>
  <Image_Interpretation>{band_xml}</Image_Interpretation>
  <Data_Access>{data_xml}</Data_Access>
  <MDElem name="Abstracted_Metadata">
    <MDATTR name="PASS">DESCENDING</MDATTR>
    <MDATTR name="mds1_tx_rx_polar">VH</MDATTR>
    <MDATTR name="MISSION">S1A</MDATTR>
    <MDATTR name="first_line_time">2026-01-01T00:00:00Z</MDATTR>
  </MDElem>
</Dimap_Document>
""",
        encoding="utf-8",
    )
    return dim_path


def _write_envi_band(path: Path, data: np.ndarray) -> None:
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="ENVI",
        width=128,
        height=128,
        count=1,
        dtype="float32",
        transform=from_origin(500000.0, 1280.0, 10.0, 10.0),
        crs="EPSG:32633",
    ) as dst:
        dst.write(data, 1)


def _spectral_band_xml(name: str) -> str:
    return f"""<Spectral_Band_Info>
      <BAND_NAME>{name}</BAND_NAME>
      <PHYSICAL_UNIT>linear</PHYSICAL_UNIT>
      <SCALING_FACTOR>1.0</SCALING_FACTOR>
      <SCALING_OFFSET>0.0</SCALING_OFFSET>
      <LOG10_SCALED>false</LOG10_SCALED>
    </Spectral_Band_Info>"""
