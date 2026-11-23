from pathlib import Path
from types import SimpleNamespace

import pytest
from pyproj import CRS

from sarpyx.snapflow import runtime, terrain, tile_crs


def _dim(folder, name, bands, offset=400_000):
    folder.mkdir(parents=True)
    target = folder / f"{name}.dim"
    data = target.with_suffix(".data")
    data.mkdir()
    infos, files = [], []
    for i, band in enumerate(bands):
        for ext in ("hdr", "img"):
            (data / f"{band}.{ext}").write_text(band)
        infos.append(f"<Spectral_Band_Info><BAND_INDEX>{i}</BAND_INDEX><BAND_NAME>{band}"
                     f"</BAND_NAME><DATA_TYPE>float32</DATA_TYPE></Spectral_Band_Info>")
        files.append(f'<Data_File><BAND_INDEX>{i}</BAND_INDEX>'
                     f'<DATA_FILE_PATH href="{data.name}/{band}.hdr"/></Data_File>')
    target.write_text(f"<Dimap_Document><Raster_Dimensions><NCOLS>10</NCOLS><NROWS>10</NROWS>"
                      f"<NBANDS>{len(bands)}</NBANDS></Raster_Dimensions>"
                      "<Coordinate_Reference_System><WKT>"
                      f"{CRS.from_epsg(32631).to_wkt()}</WKT></Coordinate_Reference_System>"
                      f"<IMAGE_TO_MODEL_TRANSFORM>10,0,0,-10,{offset},4700000</IMAGE_TO_MODEL_TRANSFORM>"
                      f"<Image_Interpretation>{''.join(infos)}</Image_Interpretation>"
                      f"<Data_Access>{''.join(files)}</Data_Access></Dimap_Document>")
    return target


def test_split_correction_uses_same_radar_source_and_materializes_bands(tmp_path):
    source = tmp_path / "radar.dim"
    seen = {}

    def factory(product, outdir, fmt, **budget):
        def tc(output_name, **params):
            seen[outdir.name] = product, fmt, budget, params
            return _dim(outdir, output_name, params["source_bands"])
        return SimpleNamespace(TerrainCorrection=tc)

    bands = ["i_IW2_VV", "q_IW2_VV", "Alpha", "Entropy", "Anisotropy"]
    output = terrain.run_split_terrain_correction(
        source, tmp_path / "out", "out_TC",
        {"source_bands": bands, "map_projection": "EPSG:32631", "output_complex": True},
        {"gpt_parallelism": 6, "gpt_memory": "8G", "gpt_cache_size": "2G"}, create_operator=factory,
    )
    cx, pol = seen["complex"], seen["polarimetry"]
    assert cx[0] == pol[0] == source
    assert cx[2]["gpt_parallelism"] + pol[2]["gpt_parallelism"] == 6
    assert cx[2]["gpt_memory"] == pol[2]["gpt_memory"] == "4096M"
    assert cx[3]["map_projection"] == pol[3]["map_projection"] == "EPSG:32631"
    assert cx[3]["source_bands"] == bands[:2] and cx[3]["output_complex"]
    assert pol[3]["source_bands"] == bands[2:] and not pol[3]["output_complex"]
    assert pol[3]["img_resampling_method"] == "BILINEAR_INTERPOLATION"
    assert set(terrain.materialized_band_names(output)) == set(bands)
    assert set(p.stem for p in output.with_suffix(".data").glob("*.img")) == set(bands)
    assert not list(output.parent.glob(".split-tc-*"))


def test_append_rejects_shifted_grid_even_when_dimensions_match(tmp_path):
    iq = _dim(tmp_path / "complex", "iq", ["i_IW2_VV", "q_IW2_VV"])
    polar = _dim(tmp_path / "scalar", "polar", ["Alpha", "Entropy"], offset=400_010)
    with pytest.raises(RuntimeError, match="TC grids differ"):
        terrain._append_aligned_polarimetry(iq, polar, ["Alpha", "Entropy"])
    assert not (iq.with_suffix(".data") / "Entropy.img").exists()


def test_runtime_replays_split_policy_for_additional_epsg(tmp_path, monkeypatch):
    radar = tmp_path / "radar.dim"
    op = SimpleNamespace(name="safe", prod_path=radar)
    ctx = runtime.PipelineContext(radar, radar, tmp_path, op, {}, {}, {})
    sel = ["i_IW2_VV", "q_IW2_VV", "Alpha", "Entropy"]
    monkeypatch.setattr(runtime, "tops_complex_tc_band_names", lambda _: sel)
    calls = []

    def split(source, outdir, output_name, params, kwargs, **_):
        calls.append((source, Path(outdir), output_name, params))
        return Path(outdir) / f"{output_name}.dim"

    monkeypatch.setattr(terrain, "run_split_terrain_correction", split)
    monkeypatch.setattr(tile_crs, "_ensure_subap_bands_terrain_corrected", lambda *_: None)
    standard = runtime.step_terrain_correction(ctx, ordered_tops_bands=True,
                                               split_polarimetric_resampling=True,
                                               map_projection="AUTO:42001")
    meta = ctx.metadata["terrain_correction_runs"][str(standard)]
    stamped = tile_crs._terrain_correct_to_epsg(standard, 32632, meta, {})
    assert stamped.name == "radar_TC_SPLIT_EPSG32632.dim"
    assert [c[0] for c in calls] == [radar, radar]
    assert [c[3]["map_projection"] for c in calls] == ["AUTO:42001", "EPSG:32632"]
    assert all(c[3]["source_bands"] == sel for c in calls)
