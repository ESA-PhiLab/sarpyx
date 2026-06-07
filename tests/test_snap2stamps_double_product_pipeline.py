import subprocess
from pathlib import Path

import pytest

from sarpyx.pipelines.double_product import snap2stamps
from sarpyx.pipelines.double_product.s1_insar import DEFAULT_DEM_NAME as DEFAULT_INSAR_DEM_NAME
from sarpyx.pipelines.runner import BUILTIN_PIPELINES
from sarpyx.snapflow.dimap import get_data_dir_from_dim
from sarpyx.snapflow.insar import run_insar_pipeline
from sarpyx.snapflow.runtime import PipelineStep
from sarpyx.snapflow.stamps import run_stamps_prep


def _dimap_product(path: Path, bands: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = get_data_dir_from_dim(path)
    data_dir.mkdir(parents=True, exist_ok=True)
    spectral = "\n".join(
        f"<Spectral_Band_Info><BAND_NAME>{band}</BAND_NAME></Spectral_Band_Info>"
        for band in bands
    )
    data_files = "\n".join(
        f'<Data_File><DATA_FILE_PATH href="{data_dir.name}/{band}.hdr" /></Data_File>'
        for band in bands
    )
    path.write_text(
        f"<Dimap_Document><Image_Interpretation>{spectral}</Image_Interpretation>"
        f"<Data_Access>{data_files}</Data_Access></Dimap_Document>",
        encoding="utf-8",
    )
    return path


def test_builtin_registry_exposes_2stamps_pipeline():
    spec = BUILTIN_PIPELINES["2stamps"]

    assert spec.input_kind == "double"
    assert spec.module is snap2stamps


def test_snap2stamps_recipe_declares_required_topsar_export_sequence():
    recipe = snap2stamps.steps(
        subswath="IW2",
        selected_polarisations=["VV"],
        use_esd=False,
    )

    assert [step.name for step in recipe] == [
        "TopsarSplit",
        "ApplyOrbitFile",
        "TopsarSplit",
        "ApplyOrbitFile",
        "TopsarCoregistration",
        "Interferogram",
        "Deburst",
        "AddElevation",
        "AddStampsLatLonBands",
        "Deburst",
        "ValidateStampsInputs",
        "StampsExport",
        "StampsPrep",
    ]
    assert recipe[0].params["subswath"] == "IW2"
    assert recipe[0].params["selected_polarisations"] == ["VV"]
    assert recipe[4].params["use_esd"] is False
    assert recipe[4].params["dem_name"] == DEFAULT_INSAR_DEM_NAME
    assert recipe[5].params["source_ref"] == "coreg"
    assert recipe[5].params["dem_name"] == DEFAULT_INSAR_DEM_NAME
    assert recipe[5].params["output_elevation"] is True
    assert recipe[5].params["output_lat_lon"] is True
    assert recipe[6].params["source_ref"] == "ifg_raw"
    assert recipe[7].params["source_ref"] == "ifg"
    assert recipe[7].params["dem_name"] == snap2stamps.DEFAULT_DEM_NAME
    assert recipe[8].params["source_ref"] == "ifg_elev"
    assert recipe[9].params["source_ref"] == "coreg"
    assert recipe[-2].params["coreg_ref"] == "coreg_deb"
    assert recipe[-2].params["ifg_ref"] == "ifg_stamps_ready"
    assert recipe[-1].params["method"] == snap2stamps.DEFAULT_STAMPS_PREP_METHOD
    assert recipe[-1].params["da_threshold"] == snap2stamps.DEFAULT_STAMPS_DA_THRESHOLD
    assert not {"TerrainCorrection", "WorldSARTiling"} & {step.name for step in recipe}


def test_snap2stamps_recipe_can_subset_matching_coreg_and_ifg():
    recipe = snap2stamps.steps(
        subset=True,
        polygon_wkt="POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
        validate_inputs=False,
    )

    assert [step.name for step in recipe[-4:]] == ["Subset", "Subset", "StampsExport", "StampsPrep"]
    assert recipe[-4].params["source_ref"] == "ifg_stamps_ready"
    assert recipe[-3].params["source_ref"] == "coreg_deb"
    assert recipe[-2].params["coreg_ref"] == "coreg_subset"
    assert recipe[-2].params["ifg_ref"] == "ifg_subset"
    assert recipe[-1].params["target_folder"] == snap2stamps.DEFAULT_TARGET_FOLDER


def test_declared_snap2stamps_runtime_executes_export_sequence(monkeypatch, tmp_path: Path):
    calls = []

    class FakeGPT:
        def __init__(self, product, outdir, **kwargs):
            self.product = Path(product)
            self.prod_path = Path(product)
            self.outdir = Path(outdir)
            self.outdir.mkdir(parents=True, exist_ok=True)

        def _out(self, label, output_name=None):
            calls.append((label, self.prod_path, self.outdir))
            path = self.outdir / f"{output_name or label}.dim"
            path.write_text(label, encoding="utf-8")
            self.prod_path = path
            return path

        def topsar_split(self, **kwargs):
            return self._out("split")

        def apply_orbit_file(self, **kwargs):
            return self._out("orbit")

        def topsar_coregistration(self, **kwargs):
            calls.append(("coreg_kwargs", kwargs["master_product"], kwargs["slave_product"], kwargs["use_esd"]))
            return self._out("coreg")

        def interferogram(self, **kwargs):
            calls.append(("ifg_kwargs", kwargs))
            return self._out("ifg")

        def deburst(self, **kwargs):
            return self._out("deburst")

        def add_elevation(self, **kwargs):
            calls.append(("add_elevation_kwargs", kwargs))
            return self._out("add_elevation")

        def stamps_lat_lon_bands(self, **kwargs):
            calls.append(("lat_lon_kwargs", kwargs))
            return self._out("lat_lon")

        def stamps_export_pair(self, **kwargs):
            calls.append(("stamps_kwargs", kwargs))
            return self._out("stamps_export", output_name=kwargs["output_name"])

        def last_error_summary(self):
            return "failed"

    monkeypatch.setattr("sarpyx.snapflow.insar.GPT", FakeGPT)
    monkeypatch.setattr(
        "sarpyx.snapflow.insar.run_stamps_prep",
        lambda **kwargs: calls.append(("stamps_prep_kwargs", kwargs)) or {"target_folder": kwargs["target_folder"]},
    )
    master = tmp_path / "master.SAFE"
    slave = tmp_path / "slave.SAFE"
    master.mkdir()
    slave.mkdir()

    result = run_insar_pipeline(
        master,
        slave,
        tmp_path / "out",
        recipe=snap2stamps.steps(use_esd=False, validate_inputs=False),
        gpt_path="/fake/gpt",
    )

    assert result["target_folder"] == tmp_path / "out" / "stamps"
    assert [call[0] for call in calls if not call[0].endswith("_kwargs")] == [
        "split",
        "orbit",
        "split",
        "orbit",
        "coreg",
        "ifg",
        "deburst",
        "add_elevation",
        "lat_lon",
        "deburst",
        "stamps_export",
    ]
    ifg_kwargs = next(call[1] for call in calls if call[0] == "ifg_kwargs")
    assert ifg_kwargs["dem_name"] == DEFAULT_INSAR_DEM_NAME
    assert ifg_kwargs["output_elevation"] is True
    assert ifg_kwargs["output_lat_lon"] is True
    add_elevation_kwargs = next(call[1] for call in calls if call[0] == "add_elevation_kwargs")
    assert add_elevation_kwargs["dem_name"] == snap2stamps.DEFAULT_DEM_NAME
    lat_lon_kwargs = next(call[1] for call in calls if call[0] == "lat_lon_kwargs")
    assert lat_lon_kwargs["output_name"] == "ifg_stamps_ready"
    stamps_kwargs = next(call[1] for call in calls if call[0] == "stamps_kwargs")
    assert Path(stamps_kwargs["coreg_product"]).parent == tmp_path / "out" / "coreg"
    assert Path(stamps_kwargs["ifg_product"]).parent == tmp_path / "out" / "ifg"
    assert stamps_kwargs["target_folder"] == tmp_path / "out" / "stamps"
    prep_kwargs = next(call[1] for call in calls if call[0] == "stamps_prep_kwargs")
    assert prep_kwargs["target_folder"] == tmp_path / "out" / "stamps"
    assert prep_kwargs["master_product"] == master


def test_validate_stamps_inputs_rejects_ifg_without_required_bands(tmp_path: Path):
    coreg = _dimap_product(tmp_path / "coreg.dim", ["i_VV", "q_VV"])
    ifg = _dimap_product(tmp_path / "ifg.dim", ["Phase_ifg_VV", "elevation"])
    recipe = [PipelineStep("ValidateStampsInputs", {"coreg_ref": "master", "ifg_ref": "slave"})]

    with pytest.raises(RuntimeError, match="orthorectified latitude/longitude"):
        run_insar_pipeline(coreg, ifg, tmp_path / "out", recipe=recipe, gpt_path="/fake/gpt")


def test_validate_stamps_inputs_accepts_required_ifg_bands(tmp_path: Path):
    coreg = _dimap_product(tmp_path / "coreg.dim", ["i_VV", "q_VV"])
    ifg = _dimap_product(
        tmp_path / "ifg.dim",
        ["Phase_ifg_VV", "elevation", "orthorectifiedLat", "orthorectifiedLon"],
    )
    recipe = [PipelineStep("ValidateStampsInputs", {"coreg_ref": "master", "ifg_ref": "slave"})]

    result = run_insar_pipeline(coreg, ifg, tmp_path / "out", recipe=recipe, gpt_path="/fake/gpt")

    assert result["coreg"] == coreg
    assert result["ifg"] == ifg


def _stamps_export_folder(path: Path) -> Path:
    for name in ("rslc", "diff0", "geo", "dem"):
        (path / name).mkdir(parents=True, exist_ok=True)
    (path / "rslc" / "20250217.rslc").write_text("rslc", encoding="utf-8")
    (path / "diff0" / "20250217_20250205.diff").write_text("diff", encoding="utf-8")
    (path / "geo" / "20250217.lat").write_text("lat", encoding="utf-8")
    (path / "geo" / "20250217.lon").write_text("lon", encoding="utf-8")
    (path / "dem" / "projected_dem.rslc").write_text("dem", encoding="utf-8")
    return path


def test_run_stamps_prep_builds_mt_prep_snap_command(monkeypatch, tmp_path: Path):
    target = _stamps_export_folder(tmp_path / "stamps")
    calls = []

    monkeypatch.setattr("sarpyx.snapflow.stamps.shutil.which", lambda command: f"/fake/{command}")

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="prepared", stderr="")

    monkeypatch.setattr("sarpyx.snapflow.stamps.subprocess.run", fake_run)

    result = run_stamps_prep(
        target_folder=target,
        master_product=tmp_path / "S1A_SLC_20250217T170613_249406_IW2_VV.SAFE",
        da_threshold=0.35,
        rg_patches=3,
        az_patches=2,
    )

    assert calls[0][0] == [
        "mt_prep_snap",
        "20250217",
        target.resolve().as_posix(),
        "0.35",
        "3",
        "2",
        "50",
        "50",
    ]
    assert calls[0][1]["cwd"] == target.resolve()
    assert result["master_date"] == "20250217"
    assert result["log_path"].read_text(encoding="utf-8").startswith("command: mt_prep_snap")
