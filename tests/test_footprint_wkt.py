import zipfile
from pathlib import Path

import h5py
import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely import wkt

from sarpyx.pipelines import runner as pipeline_runner
from sarpyx.snapflow import preprocessing
from sarpyx.snapflow.footprint_wkt import resolve_source_product_wkt, resolve_tiling_wkt


def _tsx_product_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<level1Product>
  <productInfo>
    <productVariantInfo>
      <productType>EEC_SE_HS_S</productType>
      <productVariant>EEC</productVariant>
      <projection>MAP</projection>
      <mapProjection>UTM</mapProjection>
    </productVariantInfo>
  </productInfo>
  <sceneCornerCoord><lon>11.9</lon><lat>51.4</lat></sceneCornerCoord>
  <sceneCornerCoord><lon>12.1</lon><lat>51.4</lat></sceneCornerCoord>
  <sceneCornerCoord><lon>12.1</lon><lat>51.6</lat></sceneCornerCoord>
  <sceneCornerCoord><lon>11.9</lon><lat>51.6</lat></sceneCornerCoord>
</level1Product>
"""


def test_resolve_source_product_wkt_reads_tsx_wrapper_zip(tmp_path: Path) -> None:
    product_dir = tmp_path / "TSX_OPER_SAR_HS_EEC_20071130T165208_N51-485_E011-982_0000_v0104"
    product_dir.mkdir()
    (product_dir / "TSX_OPER_SAR_HS_EEC_20071130T165208_N51-485_E011-982_0000.MD.XML").write_text(
        "<metadata><productType>SAR_HS_EEC</productType></metadata>",
        encoding="utf-8",
    )
    with zipfile.ZipFile(product_dir / "TSX_OPER_SAR_HS_EEC_20071130T165208_N51-485_E011-982_0000.ZIP", "w") as archive:
        archive.writestr("dims/TSX-1.SAR.L1B/product/TSX1_SAR__EEC_TEST.xml", _tsx_product_xml())

    assert resolve_source_product_wkt(product_dir, "TSX") == (
        "POLYGON((11.9 51.4, 12.1 51.4, 12.1 51.6, 11.9 51.6, 11.9 51.4))"
    )


def test_resolve_tiling_wkt_reads_nisar_bounding_polygon(tmp_path: Path) -> None:
    product = tmp_path / "NISAR_GSLC.h5"
    with h5py.File(product, "w") as h5_file:
        ident = h5_file.require_group("science/LSAR/identification")
        ident.create_dataset(
            "boundingPolygon",
            data=np.bytes_("POLYGON Z ((0 0 10, 1 0 11, 1 1 12, 0 1 13, 0 0 10))"),
        )

    assert resolve_tiling_wkt(None, product, product, "NISAR") == "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"


def test_resolve_tiling_wkt_reads_biomass_raster_footprint(tmp_path: Path) -> None:
    product = tmp_path / "BIOMASS_WRITE.tif"
    transform = from_origin(500000.0, 100.0, 10.0, 10.0)
    with rasterio.open(
        product,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=1,
        dtype="uint8",
        crs=pyproj.CRS.from_epsg(32633).to_wkt(),
        transform=transform,
    ) as dataset:
        dataset.write(np.ones((1, 3, 4), dtype=np.uint8))

    footprint = wkt.loads(resolve_tiling_wkt(None, product, product, "BM"))

    assert footprint.bounds == pytest.approx((15.0, 0.000633, 15.000359, 0.000905), abs=1e-6)


@pytest.mark.parametrize("pipeline_name", ["tsx", "nisar", "biomass"])
def test_pipeline_tiling_metadata_does_not_require_product_wkt(tmp_path: Path, pipeline_name: str) -> None:
    metadata = pipeline_runner._tiling_metadata(
        pipeline_runner.BUILTIN_PIPELINES[pipeline_name],
        None,
        tmp_path / "grid.geojson",
        tmp_path / "cuts",
        "zarr",
    )

    assert metadata["grid_path"] == tmp_path / "grid.geojson"
    assert metadata["cuts_outdir"] == tmp_path / "cuts"
    assert metadata["product_mode"] in {"TSX", "NISAR", "BM"}
    assert "product_wkt" not in metadata


def test_preprocessing_tiling_metadata_does_not_require_product_wkt(tmp_path: Path) -> None:
    metadata = preprocessing._tiling_metadata(
        product_wkt=None,
        grid_path=tmp_path / "grid.geojson",
        cuts_outdir=tmp_path / "cuts",
        product_mode="NISAR",
        gpt_kwargs={"gpt_memory": "16G"},
    )

    assert metadata["grid_path"] == tmp_path / "grid.geojson"
    assert metadata["cuts_outdir"] == tmp_path / "cuts"
    assert metadata["product_mode"] == "NISAR"
    assert metadata["gpt_kwargs"] == {"gpt_memory": "16G"}
    assert "product_wkt" not in metadata
