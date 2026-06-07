from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from sarpyx.utils.worldsar_h5 import (
    CORE_METADATA_KEYS,
    _collect_cut_report_texts,
    _collect_raster_sample_panels,
    _paginate_report_text,
    build_validation_dashboard_rows,
    build_validation_group_summary_rows,
    build_validation_map_layers,
    convert_tile_h5_to_zarr,
    enrich_validation_results_with_h5_structure,
    extract_tile_geometry_from_abstract_metadata,
    merge_worldsar_iw_tiles,
    normalize_expected_tile_geometries,
    resolve_expected_band_names_from_dim_product,
    validate_h5_tile,
    write_h5_validation_report_pdf,
)
from sarpyx.cli import worldsar
from sarpyx.snapflow.h5_quality import summarize_h5_raster_quality
from sarpyx.snapflow.report_manifest import hydrate_tiling_result, write_tiling_manifest
from sarpyx.snapflow.reports import _validate_tile_group
from sarpyx.snapflow.tile_crs import prepare_products_by_epsg
from sarpyx.snapflow.tiling import _validate_tile_result


def _write_tile(
    path: Path,
    *,
    include_extra_array: bool = True,
    include_quality_group: bool = True,
    include_quality_flag: bool = True,
) -> None:
    with h5py.File(path, 'w') as h5_file:
        h5_file.attrs['root_attr'] = np.bytes_(b'root')

        bands_group = h5_file.create_group('bands')
        band = bands_group.create_dataset('Band_A', data=np.arange(4, dtype=np.float32).reshape(2, 2))
        band.attrs['CLASS'] = 'RasterDataNode'
        band.attrs['IMAGE_VERSION'] = '1.0'
        band.attrs['log10_scaled'] = False
        band.attrs['raster_height'] = 2
        band.attrs['raster_width'] = 2
        band.attrs['scaling_factor'] = 1.0
        band.attrs['scaling_offset'] = 0.0
        band.attrs['unit'] = 'linear'

        if include_extra_array:
            geolocation = h5_file.create_group('geolocation')
            geolocation.create_dataset('latitude', data=np.linspace(0.0, 1.0, 4).reshape(2, 2))

        metadata = h5_file.create_group('metadata')
        abstracted = metadata.create_group('Abstracted_Metadata')
        for key in CORE_METADATA_KEYS:
            abstracted.attrs[key] = f'{key}-value'
        if include_quality_flag:
            abstracted.attrs['quality_flag'] = 'ok'

        if include_quality_group:
            quality = metadata.create_group('Quality')
            quality.attrs['state'] = 'good'


def _write_required_band_attrs(dataset) -> None:
    dataset.attrs['CLASS'] = 'RasterDataNode'
    dataset.attrs['IMAGE_VERSION'] = '1.0'
    dataset.attrs['log10_scaled'] = False
    dataset.attrs['raster_height'] = dataset.shape[0]
    dataset.attrs['raster_width'] = dataset.shape[1]
    dataset.attrs['scaling_factor'] = 1.0
    dataset.attrs['scaling_offset'] = 0.0
    dataset.attrs['unit'] = 'linear'


def _write_two_band_tile(path: Path, band_a: np.ndarray, band_b: np.ndarray) -> None:
    with h5py.File(path, 'w') as h5_file:
        bands_group = h5_file.create_group('bands')
        for name, data in {'Band_A': band_a, 'Band_B': band_b}.items():
            dataset = bands_group.create_dataset(name, data=np.asarray(data, dtype=np.float32))
            _write_required_band_attrs(dataset)
        abstracted = h5_file.create_group('metadata').create_group('Abstracted_Metadata')
        for key in CORE_METADATA_KEYS:
            abstracted.attrs[key] = f'{key}-value'


def _write_two_band_dim(path: Path) -> None:
    path.write_text(
        """<Dimap_Document>
  <Image_Interpretation>
    <Spectral_Band_Info><BAND_NAME>Band_A</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Band_B</BAND_NAME></Spectral_Band_Info>
  </Image_Interpretation>
</Dimap_Document>
""",
        encoding='utf-8',
    )


def test_convert_tile_h5_to_zarr_preserves_nested_structure_and_attributes(tmp_path: Path) -> None:
    input_tile = tmp_path / 'tile.h5'
    output_store = tmp_path / 'tile.zarr'
    _write_tile(input_tile)

    converted = convert_tile_h5_to_zarr(input_tile, output_store, overwrite=True)

    assert converted == output_store
    root = zarr.open(output_store.as_posix(), mode='r')
    assert isinstance(root, zarr.Group)
    assert root.attrs['root_attr'] == 'root'
    np.testing.assert_array_equal(root['bands']['Band_A'][:], np.arange(4, dtype=np.float32).reshape(2, 2))
    assert root['bands']['Band_A'].attrs['unit'] == 'linear'
    np.testing.assert_array_equal(root['geolocation']['latitude'][:], np.linspace(0.0, 1.0, 4).reshape(2, 2))
    assert root['metadata']['Abstracted_Metadata'].attrs['MISSION'] == 'MISSION-value'


def _write_merge_tile(path: Path, bands: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as h5_file:
        h5_file.attrs['source'] = path.parent.parent.name
        bands_group = h5_file.create_group('bands')
        for name, data in bands.items():
            dataset = bands_group.create_dataset(name, data=np.asarray(data, dtype=np.float32))
            dataset.attrs['unit'] = 'linear'
        metadata = h5_file.create_group('metadata')
        metadata.create_group('Abstracted_Metadata').attrs['PRODUCT'] = path.parent.name


def test_merge_worldsar_iw_tiles_unions_unique_and_complementary_bands(tmp_path: Path) -> None:
    product = 'S1_TEST_PRODUCT'
    _write_merge_tile(
        tmp_path / 'IW1' / product / '001U_001R.h5',
        {
            'common': np.array([[1, 0], [2, 0]], dtype=np.float32),
            'i_IW1_VV': np.ones((2, 2), dtype=np.float32),
        },
    )
    _write_merge_tile(
        tmp_path / 'IW2' / product / '001U_001R.h5',
        {
            'common': np.array([[0, 3], [0, 4]], dtype=np.float32),
            'i_IW2_VV': np.full((2, 2), 2, dtype=np.float32),
        },
    )
    _write_merge_tile(
        tmp_path / 'IW3' / product / '002U_001R.h5',
        {'i_IW3_VV': np.full((2, 2), 3, dtype=np.float32)},
    )

    summary = merge_worldsar_iw_tiles(tmp_path)

    assert summary['output_dir'] == str(tmp_path / product)
    assert summary['tile_count'] == 2
    assert summary['duplicate_tile_count'] == 1
    assert summary['copied_tiles'] == 1
    assert summary['merged_tiles'] == 1
    with h5py.File(tmp_path / product / '001U_001R.h5', 'r') as merged:
        assert sorted(merged['bands']) == ['common', 'i_IW1_VV', 'i_IW2_VV']
        np.testing.assert_array_equal(
            merged['bands/common'][()],
            np.array([[1, 3], [2, 4]], dtype=np.float32),
        )
    with h5py.File(tmp_path / product / '002U_001R.h5', 'r') as copied:
        assert sorted(copied['bands']) == ['i_IW3_VV']


def test_merge_worldsar_iw_tiles_preserves_conflicting_same_name_bands(tmp_path: Path) -> None:
    product = 'S1_TEST_PRODUCT'
    _write_merge_tile(
        tmp_path / 'IW1' / product / '001U_001R.h5',
        {'Alpha': np.array([[1, 2], [0, 0]], dtype=np.float32)},
    )
    _write_merge_tile(
        tmp_path / 'IW2' / product / '001U_001R.h5',
        {'Alpha': np.array([[9, 2], [3, 0]], dtype=np.float32)},
    )

    summary = merge_worldsar_iw_tiles(tmp_path)

    assert summary['dataset_actions']['conflict_kept_first'] == 1
    with h5py.File(tmp_path / product / '001U_001R.h5', 'r') as merged:
        assert sorted(merged['bands']) == ['Alpha']
        np.testing.assert_array_equal(
            merged['bands/Alpha'][()],
            np.array([[1, 2], [3, 0]], dtype=np.float32),
        )


def test_merge_worldsar_iw_tiles_dry_run_reports_product_without_writes(tmp_path: Path) -> None:
    product = 'S1_TEST_PRODUCT'
    _write_merge_tile(tmp_path / 'IW1' / product / '001U_001R.h5', {'i_IW1_VV': np.ones((2, 2))})

    summary = merge_worldsar_iw_tiles(tmp_path, dry_run=True)

    assert summary['dry_run'] is True
    assert summary['tile_count'] == 1
    assert not (tmp_path / product).exists()


def test_validate_h5_tile_reports_missing_arrays_and_metadata_structure(tmp_path: Path) -> None:
    complete_tile = tmp_path / 'complete.h5'
    missing_tile = tmp_path / 'missing.h5'
    _write_tile(complete_tile)
    _write_tile(
        missing_tile,
        include_extra_array=False,
        include_quality_group=False,
        include_quality_flag=False,
    )

    results = [
        validate_h5_tile(complete_tile, expected_bands=['Band_A']),
        validate_h5_tile(missing_tile, expected_bands=['Band_A']),
    ]
    summary = enrich_validation_results_with_h5_structure(results)

    assert summary['expected_array_paths'] == ['geolocation/latitude']
    assert 'metadata/Quality' in summary['expected_metadata_paths']
    assert 'metadata/Abstracted_Metadata@quality_flag' in summary['expected_metadata_attr_paths']

    complete_result, missing_result = results
    assert complete_result['structure_ok'] is True
    assert missing_result['structure_ok'] is False
    assert missing_result['missing_array_paths'] == ['geolocation/latitude']
    assert missing_result['missing_metadata_paths'] == ['metadata/Quality']
    assert missing_result['missing_metadata_attrs'] == ['metadata/Abstracted_Metadata@quality_flag']
    assert missing_result['status'] == 'failed'


def test_h5_raster_quality_counts_nan_and_all_band_zero_pixels(tmp_path: Path) -> None:
    tile = tmp_path / 'quality.h5'
    _write_two_band_tile(
        tile,
        np.array([[0.0, np.nan], [1.0, 2.0]], dtype=np.float32),
        np.array([[0.0, 3.0], [1.0, np.nan]], dtype=np.float32),
    )

    summary = summarize_h5_raster_quality(tile)

    assert summary['total_pixels'] == 4
    assert summary['zero_all_pixels'] == 1
    assert summary['nan_pixels'] == 2
    assert summary['nodata_pixels'] == 3
    assert summary['valid_pixels'] == 1
    assert summary['raster_data_ok'] is False


def test_validate_tile_group_fails_all_nodata_tile(tmp_path: Path) -> None:
    cuts_dir = tmp_path / 'cuts'
    cuts_dir.mkdir()
    tile = cuts_dir / 'empty.h5'
    _write_two_band_tile(tile, np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32))
    dim = tmp_path / 'product.dim'
    _write_two_band_dim(dim)

    group = _validate_tile_group(cuts_dir, dim)

    result = group['results'][0]
    assert result['status'] == 'failed'
    assert result['raster_data_ok'] is False
    assert result['valid_pixels'] == 0


def test_validate_tile_group_uses_tiling_actual_tiles_filter(tmp_path: Path) -> None:
    cuts_dir = tmp_path / 'cuts'
    cuts_dir.mkdir()
    _write_two_band_tile(cuts_dir / 'current.h5', np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32))
    _write_two_band_tile(cuts_dir / 'stale_other_swath.h5', np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32))
    dim = tmp_path / 'product.dim'
    _write_two_band_dim(dim)

    group = _validate_tile_group(cuts_dir, dim, tiling_result={'actual_tiles': ['current']})

    assert [result['tile'] for result in group['results']] == ['current']


def test_validate_tile_result_skips_and_removes_partial_nodata_tile(tmp_path: Path) -> None:
    tile = tmp_path / 'empty.h5'
    _write_two_band_tile(
        tile,
        np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32),
    )

    result = _validate_tile_result('empty', tile, 'tile cut')

    assert result['status'] == 'partial'
    assert result['valid_pixels'] == 3
    assert not tile.exists()


def test_run_tiling_routes_rectangles_to_matching_epsg_products(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sarpyx.snapflow import tiling as tiling_mod
    tiling_mod = importlib.reload(tiling_mod)

    def rectangle(name: str, epsg: str, x0: float) -> dict:
        props = {'name': name, 'epsg': epsg}
        return {
            'TL': {'properties': props, 'geometry': {'coordinates': [x0, 1.0]}},
            'TR': {'properties': props, 'geometry': {'coordinates': [x0 + 1.0, 1.0]}},
            'BR': {'properties': props, 'geometry': {'coordinates': [x0 + 1.0, 0.0]}},
            'BL': {'properties': props, 'geometry': {'coordinates': [x0, 0.0]}},
        }

    rect_32 = rectangle('tile32', 'EPSG:32632', 0.0)
    rect_33 = rectangle('tile33', 'EPSG:32633', 1.0)
    source = tmp_path / 'source.SAFE'
    intermediate = tmp_path / 'intermediate.dim'
    grid = tmp_path / 'grid.geojson'
    source.mkdir()
    intermediate.write_text('<Dimap_Document />', encoding='utf-8')
    grid.write_text('{}', encoding='utf-8')
    product_32 = tmp_path / 'intermediate_EPSG32632.dim'
    product_33 = tmp_path / 'intermediate_EPSG32633.dim'
    product_32.write_text('32', encoding='utf-8')
    product_33.write_text('33', encoding='utf-8')
    stale_dir = tmp_path / 'cuts' / 'PRODUCT'
    stale_dir.mkdir(parents=True)
    (stale_dir / 'stale_other_swath.h5').write_bytes(b'h5')
    calls = []

    monkeypatch.setattr(tiling_mod, 'select_intersecting_grid_rectangles', lambda *_args, **_kwargs: [rect_32, rect_33])
    monkeypatch.setattr(tiling_mod, 'extract_product_id', lambda _path: 'PRODUCT')
    monkeypatch.setattr(
        tiling_mod,
        'prepare_products_by_epsg',
        lambda _intermediate, epsgs, _gpt_kwargs, terrain_correction=None: {32632: product_32, 32633: product_33},
    )

    def fake_cut(rect, product, cuts_dir, *_args):
        tile = rect['BL']['properties']['name']
        calls.append((tile, Path(product)))
        output = cuts_dir / f'{tile}.h5'
        output.write_bytes(b'h5')
        return {'tile': tile, 'status': 'success', 'output_path': str(output), 'valid_fraction': 1.0, 'nodata_fraction': 0.0}

    monkeypatch.setattr(tiling_mod, '_cut_single_tile', fake_cut)

    result = tiling_mod._run_tiling(
        'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))',
        grid,
        source,
        intermediate,
        tmp_path / 'cuts',
        'S1TOPS',
        None,
        None,
        None,
    )

    assert calls == [('tile32', product_32), ('tile33', product_33)]
    assert result['actual_tiles'] == ['tile32', 'tile33']
    assert result['extra_tiles'] == []
    assert result['crs_groups'] == {'EPSG:32632': 1, 'EPSG:32633': 1}
    report = Path(result['report_path']).read_text(encoding='utf-8')
    assert 'Full-data tile policy' in report
    assert 'Tile selection policy' in report
    assert '- EPSG:32632: 1 candidate intersecting tiles' in report


def test_grid_cell_intersection_selection_keeps_edge_tiles(tmp_path: Path) -> None:
    import json

    from shapely.geometry import Point, Polygon

    from sarpyx.snapflow.tile_selection import _grid_cell_rectangle, select_intersecting_grid_rectangles

    feature = {
        'type': 'Feature',
        'properties': {'name': 'edge-tile', 'row': '461U', 'col': '101R', 'epsg': 'EPSG:32633'},
        'geometry': {'type': 'Point', 'coordinates': [12.0, 41.5]},
    }
    rectangle = _grid_cell_rectangle(feature)
    coords = {corner: rectangle[corner]['geometry']['coordinates'] for corner in ('TL', 'TR', 'BR', 'BL')}
    edge_polygon = Polygon([
        coords['TL'],
        coords['TR'],
        coords['BR'],
        ((coords['BR'][0] + coords['BL'][0]) / 2.0, (coords['BR'][1] + coords['BL'][1]) / 2.0),
        ((coords['TL'][0] + coords['BL'][0]) / 2.0, (coords['TL'][1] + coords['BL'][1]) / 2.0),
    ])
    assert not edge_polygon.contains(Point(*coords['BL']))
    grid = tmp_path / 'grid.geojson'
    grid.write_text(json.dumps({'type': 'FeatureCollection', 'features': [feature]}), encoding='utf-8')

    selected = select_intersecting_grid_rectangles(edge_polygon.wkt, grid)

    assert [item['BL']['properties']['name'] for item in selected] == ['edge-tile']


def test_prepare_products_by_epsg_reruns_terrain_correction_from_pre_tc_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sarpyx.snapflow import tile_crs as tile_crs_mod

    source = tmp_path / 'calibrated.dim'
    intermediate = tmp_path / 'terrain_corrected.dim'
    source.write_text(
        '<Dimap_Document><Raster_Dimensions><NBANDS>0</NBANDS></Raster_Dimensions><Image_Interpretation/><Data_Access/></Dimap_Document>',
        encoding='utf-8',
    )
    intermediate.write_text('intermediate', encoding='utf-8')
    calls = []

    def fake_run_gpt_op(product, outdir, output_format, op_name, **kwargs):
        calls.append((Path(product), Path(outdir), output_format, op_name, kwargs))
        output_path = Path(outdir) / f"{kwargs['output_name']}.dim"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('tc', encoding='utf-8')
        return output_path

    monkeypatch.setattr(tile_crs_mod, 'dim_epsg', lambda _path: 32633)
    monkeypatch.setattr(tile_crs_mod, 'run_gpt_op', fake_run_gpt_op)

    products = prepare_products_by_epsg(
        intermediate,
        [32632, 32633],
        {'gpt_memory': None, 'gpt_parallelism': None, 'gpt_timeout': None},
        terrain_correction={
            'source_product': str(source),
            'params': {
                'map_projection': 'AUTO:42001',
                'pixel_spacing_in_meter': 10.0,
                'output_name': 'ignored',
            },
            'output_dir': str(tmp_path / 'out'),
        },
    )

    assert products[32633] == intermediate
    assert products[32632].name == 'calibrated_TC_EPSG32632.dim'
    assert calls == [
        (
            source,
            tmp_path / 'out' / 'worldsar_tc_epsg',
            'BEAM-DIMAP',
            'TerrainCorrection',
            {
                'map_projection': 'EPSG:32632',
                'pixel_spacing_in_meter': 10.0,
                'output_name': 'calibrated_TC_EPSG32632',
                'gpt_memory': None,
                'gpt_parallelism': None,
                'gpt_timeout': None,
            },
        )
    ]


def test_worldsar_h5_to_zarr_only_mode_creates_sibling_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    input_tile = tmp_path / 'tile_only.h5'
    _write_tile(input_tile)

    monkeypatch.setattr(sys, 'argv', ['worldsar.py', '--input', str(input_tile), '--h5-to-zarr-only'])

    with pytest.raises(SystemExit) as excinfo:
        worldsar.main()

    assert excinfo.value.code == 0

    output = capsys.readouterr().out
    summary = json.loads(output)
    expected_store = input_tile.with_suffix('.zarr')
    assert summary['output'] == str(expected_store)
    assert expected_store.is_dir()


def test_resolve_expected_band_names_prefers_materialized_data_dir(tmp_path: Path) -> None:
    dim_path = tmp_path / 'product.dim'
    data_dir = tmp_path / 'product.data'
    data_dir.mkdir()

    dim_path.write_text(
        """<Dimap_Document>
  <Image_Interpretation>
    <Spectral_Band_Info><BAND_NAME>Alpha</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Anisotropy</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Entropy</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>i_IW3_VH</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>q_IW3_VH</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Intensity_IW3_VH</BAND_NAME></Spectral_Band_Info>
  </Image_Interpretation>
  <Data_Access>
    <Data_File>
      <BAND_INDEX>1</BAND_INDEX>
      <DATA_FILE_PATH href="./product.data/Alpha.hdr" />
    </Data_File>
    <Data_File>
      <BAND_INDEX>2</BAND_INDEX>
      <DATA_FILE_PATH href="./product.data/Anisotropy.hdr" />
    </Data_File>
    <Data_File>
      <BAND_INDEX>3</BAND_INDEX>
      <DATA_FILE_PATH href="./product.data/Entropy.hdr" />
    </Data_File>
    <Data_File>
      <BAND_INDEX>4</BAND_INDEX>
      <DATA_FILE_PATH href="./product.data/i_IW3_VH.hdr" />
    </Data_File>
    <Data_File>
      <BAND_INDEX>5</BAND_INDEX>
      <DATA_FILE_PATH href="./product.data/q_IW3_VH.hdr" />
    </Data_File>
  </Data_Access>
</Dimap_Document>
""",
        encoding='utf-8',
    )

    for name in ('Alpha', 'Anisotropy', 'Entropy', 'i_IW3_VH', 'q_IW3_VH'):
        (data_dir / f'{name}.hdr').write_text('ENVI\n', encoding='utf-8')

    expected_bands = resolve_expected_band_names_from_dim_product(dim_path)

    assert expected_bands == ['Alpha', 'Anisotropy', 'Entropy', 'i_IW3_VH', 'q_IW3_VH']


def test_validate_h5_tile_ignores_virtual_intensity_bands_when_data_dir_is_materialized(tmp_path: Path) -> None:
    dim_path = tmp_path / 'product.dim'
    data_dir = tmp_path / 'product.data'
    data_dir.mkdir()

    dim_path.write_text(
        """<Dimap_Document>
  <Image_Interpretation>
    <Spectral_Band_Info><BAND_NAME>Alpha</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Anisotropy</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Entropy</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>i_IW3_VH</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>q_IW3_VH</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Intensity_IW3_VH</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Intensity_IW3_VV</BAND_NAME></Spectral_Band_Info>
  </Image_Interpretation>
</Dimap_Document>
""",
        encoding='utf-8',
    )

    for name in ('Alpha', 'Anisotropy', 'Entropy', 'i_IW3_VH', 'q_IW3_VH'):
        (data_dir / f'{name}.hdr').write_text('ENVI\n', encoding='utf-8')

    tile_path = tmp_path / 'tile.h5'
    with h5py.File(tile_path, 'w') as h5_file:
        bands_group = h5_file.create_group('bands')
        for band_name in ('Alpha', 'Anisotropy', 'Entropy', 'i_IW3_VH', 'q_IW3_VH'):
            band = bands_group.create_dataset(band_name, data=np.arange(4, dtype=np.float32).reshape(2, 2))
            band.attrs['CLASS'] = 'RasterDataNode'
            band.attrs['IMAGE_VERSION'] = '1.0'
            band.attrs['log10_scaled'] = False
            band.attrs['raster_height'] = 2
            band.attrs['raster_width'] = 2
            band.attrs['scaling_factor'] = 1.0
            band.attrs['scaling_offset'] = 0.0
            band.attrs['unit'] = 'linear'

        metadata = h5_file.create_group('metadata')
        abstracted = metadata.create_group('Abstracted_Metadata')
        for key in CORE_METADATA_KEYS:
            abstracted.attrs[key] = f'{key}-value'

    expected_bands = resolve_expected_band_names_from_dim_product(dim_path)
    result = validate_h5_tile(tile_path, expected_bands=expected_bands)

    assert result['missing_bands'] == []
    assert result['extra_bands'] == []
    assert result['bands_ok'] is True


def test_extract_tile_geometry_from_abstract_metadata_builds_polygon_and_center() -> None:
    geometry = extract_tile_geometry_from_abstract_metadata(
        {
            'first_near_lat': 45.0,
            'first_near_long': 10.0,
            'first_far_lat': 45.0,
            'first_far_long': 10.1,
            'last_far_lat': 44.9,
            'last_far_long': 10.1,
            'last_near_lat': 44.9,
            'last_near_long': 10.0,
            'centre_lat': 44.95,
            'centre_lon': 10.05,
        }
    )

    assert geometry['tile_polygon_coords'] == [
        (10.0, 45.0),
        (10.1, 45.0),
        (10.1, 44.9),
        (10.0, 44.9),
        (10.0, 45.0),
    ]
    assert geometry['tile_center_coords'] == (10.05, 44.95)


def test_extract_tile_geometry_from_abstract_metadata_falls_back_to_center_only() -> None:
    geometry = extract_tile_geometry_from_abstract_metadata(
        {
            'centre_lat': 12.5,
            'centre_lon': 41.9,
        }
    )

    assert geometry['tile_polygon_coords'] is None
    assert geometry['tile_center_coords'] == (41.9, 12.5)


def test_normalize_expected_tile_geometries_preserves_tile_names() -> None:
    rectangles = [
        {
            'TL': {'geometry': {'coordinates': [10.0, 45.0]}},
            'TR': {'geometry': {'coordinates': [10.1, 45.0]}},
            'BR': {'geometry': {'coordinates': [10.1, 44.9]}},
            'BL': {'geometry': {'coordinates': [10.0, 44.9]}, 'properties': {'name': 'tile-a'}},
        },
        {
            'TL': {'geometry': {'coordinates': [10.1, 45.0]}},
            'TR': {'geometry': {'coordinates': [10.2, 45.0]}},
            'BR': {'geometry': {'coordinates': [10.2, 44.9]}},
            'BL': {'geometry': {'coordinates': [10.1, 44.9]}, 'properties': {'name': 'tile-b'}},
        },
    ]

    normalized = normalize_expected_tile_geometries(rectangles)

    assert sorted(normalized) == ['tile-a', 'tile-b']
    assert normalized['tile-a'][0] == (10.0, 45.0)
    assert normalized['tile-a'][-1] == (10.0, 45.0)


def _synthetic_validation_group() -> dict[str, object]:
    expected_tile_geometries = {
        'tile-a': [(10.0, 45.0), (10.1, 45.0), (10.1, 44.9), (10.0, 44.9), (10.0, 45.0)],
        'tile-b': [(10.1, 45.0), (10.2, 45.0), (10.2, 44.9), (10.1, 44.9), (10.1, 45.0)],
        'tile-c': [(10.2, 45.0), (10.3, 45.0), (10.3, 44.9), (10.2, 44.9), (10.2, 45.0)],
        'tile-skip': [(10.3, 45.0), (10.4, 45.0), (10.4, 44.9), (10.3, 44.9), (10.3, 45.0)],
        'tile-outside': [(10.4, 45.0), (10.5, 45.0), (10.5, 44.9), (10.4, 44.9), (10.4, 45.0)],
    }
    expected_deliverable_geometries = {
        tile: coords
        for tile, coords in expected_tile_geometries.items()
        if tile not in {'tile-skip', 'tile-outside'}
    }
    source_wkt = 'POLYGON ((9.95 44.85, 10.35 44.85, 10.35 45.05, 9.95 45.05, 9.95 44.85))'
    return {
        'name': 'IW1',
        'swath': 'IW1',
        'cuts_dir': 'cuts/IW1',
        'intermediate_product': 'intermediate/IW1.dim',
        'expected_bands': ['Band_A', 'Band_B'],
        'expected_array_paths': ['geolocation/latitude'],
        'expected_metadata_paths': ['metadata', 'metadata/Abstracted_Metadata'],
        'expected_metadata_attr_paths': ['metadata/Abstracted_Metadata@quality_flag'],
        'candidate_tiles': ['tile-a', 'tile-b', 'tile-c', 'tile-skip', 'tile-outside'],
        'expected_tiles': ['tile-a', 'tile-b', 'tile-c'],
        'actual_tiles': ['tile-a', 'tile-b', 'tile-extra'],
        'missing_tiles': ['tile-c'],
        'extra_tiles': ['tile-extra'],
        'partial_tiles': ['tile-skip'],
        'skipped_tiles': ['tile-outside'],
        'failed_tiles': ['tile-b'],
        'expected_tile_count': 3,
        'actual_tile_count': 3,
        'source_wkt': source_wkt,
        'report_source_wkt': source_wkt,
        'candidate_tile_geometries': expected_tile_geometries,
        'expected_tile_geometries': expected_deliverable_geometries,
        'results': [
            {
                'tile': 'tile-a',
                'swath': 'IW1',
                'status': 'success',
                'bands_ok': True,
                'metadata_ok': True,
                'band_attrs_ok': True,
                'structure_ok': True,
                'missing_bands': [],
                'extra_bands': [],
                'missing_metadata_section': False,
                'empty_metadata_fields': [],
                'missing_core_metadata_fields': [],
                'empty_core_metadata_fields': [],
                'band_attr_issues': {},
                'shape_summary': [],
                'missing_array_paths': [],
                'missing_metadata_paths': [],
                'missing_metadata_attrs': [],
                'tile_polygon_coords': expected_tile_geometries['tile-a'],
                'tile_center_coords': (10.05, 44.95),
            },
            {
                'tile': 'tile-b',
                'swath': 'IW1',
                'status': 'failed',
                'bands_ok': False,
                'metadata_ok': False,
                'band_attrs_ok': False,
                'structure_ok': False,
                'missing_bands': ['Band_B'],
                'extra_bands': [],
                'missing_metadata_section': False,
                'empty_metadata_fields': ['quality_flag'],
                'missing_core_metadata_fields': [],
                'empty_core_metadata_fields': ['MISSION'],
                'band_attr_issues': {'Band_A': {'missing_attrs': ['unit'], 'empty_attrs': [], 'invalid_shape': False, 'shape': (2, 2)}},
                'shape_summary': ['(2, 2)', '(3, 3)'],
                'missing_array_paths': ['geolocation/latitude'],
                'missing_metadata_paths': ['metadata/Abstracted_Metadata'],
                'missing_metadata_attrs': ['metadata/Abstracted_Metadata@quality_flag'],
                'tile_polygon_coords': None,
                'tile_center_coords': (10.15, 44.95),
            },
            {
                'tile': 'tile-extra',
                'swath': 'IW1',
                'status': 'success',
                'bands_ok': True,
                'metadata_ok': True,
                'band_attrs_ok': True,
                'structure_ok': True,
                'missing_bands': [],
                'extra_bands': [],
                'missing_metadata_section': False,
                'empty_metadata_fields': [],
                'missing_core_metadata_fields': [],
                'empty_core_metadata_fields': [],
                'band_attr_issues': {},
                'shape_summary': [],
                'missing_array_paths': [],
                'missing_metadata_paths': [],
                'missing_metadata_attrs': [],
                'tile_polygon_coords': [(10.32, 45.0), (10.34, 45.0), (10.34, 44.98), (10.32, 44.98), (10.32, 45.0)],
                'tile_center_coords': (10.33, 44.99),
            },
        ],
        'rows': [{'ID': 'tile-a'}, {'ID': 'tile-b'}, {'ID': 'tile-extra'}],
    }


def test_validation_summary_dashboard_and_map_helpers_cover_status_buckets() -> None:
    group = _synthetic_validation_group()

    summary_rows = build_validation_group_summary_rows([group])
    dashboard_rows = build_validation_dashboard_rows(group)
    map_layers = build_validation_map_layers([group])

    assert summary_rows == [{
        'group': 'IW1',
        'candidate': 5,
        'expected': 3,
        'actual': 3,
        'passed': 2,
        'failed': 1,
        'partial': 1,
        'skipped': 1,
        'missing': 1,
        'extra': 1,
        'overall_status': 'FAIL',
    }]
    assert dashboard_rows[0]['check'] == 'band inventory'
    assert dashboard_rows[0]['passed'] == 2
    assert dashboard_rows[-1] == {'check': 'overall', 'passed': 2, 'failed': 1, 'pass_pct': pytest.approx(66.7, abs=0.1)}
    assert [item['tile'] for item in map_layers['passed_polygons']] == ['tile-a']
    assert [item['tile'] for item in map_layers['failed_points']] == ['tile-b']
    assert [item['tile'] for item in map_layers['extra_polygons']] == ['tile-extra']
    assert [item['tile'] for item in map_layers['missing_polygons']] == ['tile-c']
    assert [item['tile'] for item in map_layers['partial_polygons']] == ['tile-skip']
    assert [item['tile'] for item in map_layers['skipped_polygons']] == ['tile-outside']
    assert map_layers['counts'] == {
        'candidate': 5,
        'expected': 3,
        'passed': 1,
        'failed': 1,
        'partial': 1,
        'skipped': 1,
        'missing': 1,
        'extra': 1,
    }
    assert map_layers['tiles_with_center_only_count'] == 1
    assert map_layers['tiles_without_geometry_count'] == 0


def test_write_h5_validation_report_pdf_smoke(tmp_path: Path) -> None:
    report_path = tmp_path / 'validation_report.pdf'
    group = _synthetic_validation_group()

    written = write_h5_validation_report_pdf(report_path, 'synthetic-product', [group])

    assert written == report_path
    assert report_path.exists()
    assert report_path.stat().st_size > 0


def test_validation_pdf_samples_raster_bands_from_tile(tmp_path: Path) -> None:
    tile_path = tmp_path / 'tile-a.h5'
    with h5py.File(tile_path, 'w') as h5_file:
        bands = h5_file.create_group('bands')
        for index, name in enumerate(['Sigma0_VV', 'Sigma0_VH', 'i_IW1_VV_SA1', 'Entropy']):
            dataset = bands.create_dataset(name, data=np.full((12, 10), index + 1, dtype=np.float32))
            _write_required_band_attrs(dataset)

    group = _synthetic_validation_group()
    group['results'][0]['output_path'] = str(tile_path)
    sample, panels, error = _collect_raster_sample_panels([group])

    assert error is None
    assert sample is not None
    assert sample['result']['tile'] == 'tile-a'
    assert [panel['name'] for panel in panels][:3] == ['i_IW1_VV_SA1', 'Sigma0_VH', 'Sigma0_VV']

    report_path = tmp_path / 'validation_report.pdf'
    write_h5_validation_report_pdf(report_path, 'synthetic-product', [group])

    assert report_path.exists()
    assert report_path.stat().st_size > 0


def test_validation_pdf_collects_cut_report_text(tmp_path: Path) -> None:
    cut_report = tmp_path / 'cuts_report_SUCCESS.txt'
    cut_report.write_text(
        'WorldSAR tile cutting report\nExpected tiles: 2\n' + '\n'.join(f'line {index}' for index in range(70)),
        encoding='utf-8',
    )
    group = _synthetic_validation_group()
    group['cut_report_path'] = str(cut_report)

    reports = _collect_cut_report_texts([group])
    pages = _paginate_report_text(reports[0][1], max_lines=30)

    assert reports[0][0] == str(cut_report)
    assert reports[0][1][0] == 'WorldSAR tile cutting report'
    assert len(pages) > 1


def test_tiling_manifest_hydrates_regenerated_report_metadata(tmp_path: Path) -> None:
    cut_report = tmp_path / 'product_cuts_report_SUCCESS.txt'
    cut_report.write_text('WorldSAR tile cutting report\nExpected full-data tiles: 1\n', encoding='utf-8')
    tile_coords = [(10.0, 45.0), (10.1, 45.0), (10.1, 44.9), (10.0, 44.9), (10.0, 45.0)]
    tiling_result = {
        'report_path': cut_report,
        'source_wkt': 'POLYGON ((10 45, 10.1 45, 10.1 44.9, 10 44.9, 10 45))',
        'candidate_tiles': ['tile-a', 'tile-b'],
        'expected_tiles': ['tile-a'],
        'actual_tiles': ['tile-a'],
        'candidate_tile_geometries': {'tile-a': tile_coords},
        'expected_tile_geometries': {'tile-a': tile_coords},
    }

    write_tiling_manifest(cut_report, tiling_result)
    hydrated = hydrate_tiling_result({'report_path': cut_report})

    assert hydrated['expected_tiles'] == ['tile-a']
    assert hydrated['candidate_tiles'] == ['tile-a', 'tile-b']
    assert hydrated['source_wkt'].startswith('POLYGON')
    assert hydrated['expected_tile_geometries']['tile-a'][0] == [10.0, 45.0]


def test_cut_report_text_fallback_preserves_summary_counts(tmp_path: Path) -> None:
    cut_report = tmp_path / 'product_cuts_report_SUCCESS.txt'
    cut_report.write_text(
        '\n'.join(
            [
                'WorldSAR tile cutting report',
                'Product WKT: POLYGON ((10 45, 10.1 45, 10.1 44.9, 10 44.9, 10 45))',
                'Expected tiles: 5',
                'Actual tiles on disk: 3',
                'Skipped tiles (outside raster bounds): 1',
                'Skipped tiles (incomplete raster coverage): 1',
                '',
                'Skipped tiles:',
                '- tile-skip-a: outside raster bounds | /tmp/tile-skip-a.h5',
                '- tile-skip-b: incomplete raster coverage | /tmp/tile-skip-b.h5',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    hydrated = hydrate_tiling_result({'report_path': cut_report})
    group = {
        'name': 'IW1',
        'results': [
            {'tile': 'tile-a', 'status': 'success'},
            {'tile': 'tile-b', 'status': 'success'},
            {'tile': 'tile-c', 'status': 'success'},
        ],
        **hydrated,
    }

    summary = build_validation_group_summary_rows([group])[0]
    assert summary['candidate'] == 5
    assert summary['expected'] == 3
    assert summary['actual'] == 3
    assert summary['skipped'] == 2
    assert hydrated['skipped_tiles'] == ['tile-skip-a', 'tile-skip-b']
    assert hydrated['report_source_wkt'].startswith('POLYGON')
