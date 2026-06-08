import inspect
from pathlib import Path

from sarpyx.pipelines.double_product import s1_insar
from sarpyx.pipelines.single_product import biomass, csg, nisar, s1_strip, s1_tops, tsx
from sarpyx.snapflow.insar import run_insar_pipeline


def test_sentinel_tops_pipeline_lists_explicit_operations():
    steps = s1_tops.steps()
    assert [step.name for step in steps] == [
        "TopsarSplit",
        "ApplyOrbitFile",
        "Calibration",
        "TopsarDerampDemod",
        "Deburst",
        "do_subaps",
        "polarimetric_decomposition",
        "merge_iq_into_pdec",
        "TerrainCorrection",
        "WorldSARTiling",
    ]
    assert steps[0].params["subswath"] == (s1_tops.DEFAULT_SWATH or "IW*")
    assert steps[0].params["first_burst_index"] == s1_tops.DEFAULT_FIRST_BURST
    assert steps[0].params["last_burst_index"] == s1_tops.DEFAULT_LAST_BURST
    assert steps[5].params["n_decompositions"] == [2]
    assert steps[8].params["source_bands"] is None


def test_sentinel_strip_pipeline_lists_fallback_bandmerge():
    steps = s1_strip.steps(sentinel_subap_decompositions=[5])
    assert [step.name for step in steps] == [
        "ApplyOrbitFile",
        "Calibration",
        "do_subaps",
        "polarimetric_decomposition",
        "merge_iq_into_pdec",
        "TerrainCorrection",
        "WorldSARTiling",
    ]
    assert steps[2].params["n_decompositions"] == [5]
    assert steps[5].params["source_bands"] is None
    assert [step.name for step in steps[4].params["fallback_steps"]] == [
        "update_dim_add_bands_from_data_dir",
        "BandMerge",
    ]


def test_tsx_pipeline_steps_branch_on_resolved_recipe_inputs(tmp_path: Path):
    assert [step.name for step in tsx.steps(geocoded=True, output_file=tmp_path / "out.dim")] == ["Write", "WorldSARTiling"]
    detected_steps = tsx.steps(output_complex=False)
    assert [step.name for step in detected_steps] == ["Calibration", "TerrainCorrection", "WorldSARTiling"]
    assert detected_steps[0].params["output_complex"] is False
    assert detected_steps[1].params["pixel_spacing_in_meter"] == 5.0


def test_simple_pipeline_definition_shapes():
    assert [step.name for step in biomass.steps()] == ["Write", "WorldSARTiling"]
    assert [step.name for step in csg.steps(geocoded=True)] == ["Write", "WorldSARTiling"]
    assert [step.name for step in nisar.steps()] == ["WorldSARTiling"]


def test_pipeline_modules_are_recipe_only():
    for module in (biomass, csg, s1_insar, nisar, s1_strip, s1_tops, tsx):
        public_functions = [
            name
            for name, value in inspect.getmembers(module, inspect.isfunction)
            if value.__module__ == module.__name__ and not name.startswith("_")
        ]
        assert public_functions == ["steps"]


def test_no_duplicate_root_pipeline_directory():
    assert not (Path(__file__).resolve().parents[1] / "pipelines").exists()


def test_insar_pipeline_declares_snapflow_v2_steps():
    recipe = s1_insar.steps(
        subswath="IW2",
        selected_polarisations=["VV"],
        use_esd=False,
        orbit_continue_on_fail=True,
    )
    assert [step.name for step in recipe] == [
        "TopsarSplit",
        "ApplyOrbitFile",
        "TopsarSplit",
        "ApplyOrbitFile",
        "TopsarCoregistration",
        "Deburst",
        "Interferogram",
        "TopoPhaseRemoval",
        "Subset",
        "TerrainCorrection",
        "WorldSARTiling",
    ]
    assert recipe[0].params["subswath"] == "IW2"
    assert recipe[1].params["continue_on_fail"] is True
    assert recipe[3].params["continue_on_fail"] is True
    assert recipe[4].params["use_esd"] is False
    assert recipe[-1].params["intermediate_ref"] == "terrain_corrected"


def test_declared_insar_runtime_executes_recipe(monkeypatch, tmp_path: Path):
    calls = []

    class FakeGPT:
        def __init__(self, product, outdir, **kwargs):
            self.product = Path(product)
            self.outdir = Path(outdir)
            self.outdir.mkdir(parents=True, exist_ok=True)

        def _out(self, label):
            calls.append((label, self.product, self.outdir))
            path = self.outdir / f"{label}.dim"
            path.write_text(label, encoding="utf-8")
            return path

        def topsar_split(self, **kwargs):
            return self._out("split")

        def apply_orbit_file(self, **kwargs):
            return self._out("orbit")

        def topsar_coregistration(self, **kwargs):
            calls.append(("coreg_kwargs", kwargs["master_product"], kwargs["slave_product"], kwargs["use_esd"]))
            return self._out("coreg")

        def deburst(self, **kwargs):
            return self._out("deburst")

        def interferogram(self, **kwargs):
            return self._out("ifg")

        def topo_phase_removal(self, **kwargs):
            return self._out("topo")

        def subset(self, **kwargs):
            return self._out("subset")

        def terrain_correction(self, **kwargs):
            return self._out("tc")

        def last_error_summary(self):
            return "failed"

    monkeypatch.setattr("sarpyx.snapflow.insar.GPT", FakeGPT)
    master = tmp_path / "master.SAFE"
    slave = tmp_path / "slave.SAFE"
    master.mkdir()
    slave.mkdir()

    result = run_insar_pipeline(
        master,
        slave,
        tmp_path / "out",
        recipe=s1_insar.steps(subswath="IW2", selected_polarisations=["VV"], use_esd=False),
        gpt_path="/fake/gpt",
    )

    assert Path(result).name == "tc.dim"
    assert [call[0] for call in calls if call[0] != "coreg_kwargs"] == [
        "split",
        "orbit",
        "split",
        "orbit",
        "coreg",
        "deburst",
        "ifg",
        "topo",
        "subset",
        "tc",
    ]
    assert ("coreg_kwargs", tmp_path / "out" / "master" / "orbit.dim", tmp_path / "out" / "slave" / "orbit.dim", False) in calls
