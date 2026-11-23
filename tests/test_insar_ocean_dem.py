"""Ocean geometry controls without SNAP execution or a pytest dependency."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree.ElementTree import fromstring

from sarpyx.pipelines.double_product import s1_insar
from sarpyx.snapflow.engine import GPT
from sarpyx.snapflow.insar import run_insar_pipeline


class InSAROceanDEMTests(unittest.TestCase):
    def test_generic_recipe_preserves_safe_masking_default(self):
        recipe = s1_insar.steps()
        coreg = next(step for step in recipe if step.name == "TopsarCoregistration")
        self.assertTrue(coreg.params["mask_out_area_without_elevation"])

    def test_checked_ocean_policy_keeps_complete_insar_recipe(self):
        recipe = s1_insar.steps(
            mask_out_area_without_elevation=False, nodata_value_at_sea=False,
        )
        by_name = {step.name: step for step in recipe}
        self.assertFalse(by_name["TopsarCoregistration"].params["mask_out_area_without_elevation"])
        self.assertFalse(by_name["TerrainCorrection"].params["nodata_value_at_sea"])
        self.assertTrue(by_name["TopsarCoregistration"].params["use_esd"])
        self.assertIn("Interferogram", by_name)
        self.assertIn("TopoPhaseRemoval", by_name)
        self.assertIn("WorldSARTiling", by_name)

    def test_coreg_graph_passes_mask_policy_to_snap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master, slave = root / "master.dim", root / "slave.dim"
            master.touch()
            slave.touch()
            # Only graph construction is exercised, not GPT configuration/run.
            gpt = GPT.__new__(GPT)
            gpt.format = "BEAM-DIMAP"
            for masking in (True, False):
                with self.subTest(masking=masking), \
                     patch.object(gpt, "_write_graph", return_value=(root / "graph.xml", root / "out.dim")) as write, \
                     patch.object(gpt, "run_graph", return_value=root / "out.dim"):
                    gpt.topsar_coregistration(
                        master, slave, mask_out_area_without_elevation=masking,
                    )
                    graph = fromstring(write.call_args.kwargs["graph_xml"])
                    value = graph.findtext(
                        "./node[@id='BackGeocoding']/parameters/maskOutAreaWithoutElevation"
                    )
                    self.assertEqual(value, str(masking).lower())
                    self.assertIsNotNone(graph.find("./node[@id='EnhancedSpectralDiversity']"))

    def test_runtime_forwards_ocean_settings_through_full_processing_chain(self):
        calls = []
        class FakeGPT:
            def __init__(self, product, outdir, **kwargs):
                self.outdir = Path(outdir)
                self.outdir.mkdir(parents=True, exist_ok=True)

            def __getattr__(self, name):
                def operator(**params):
                    calls.append((name, params))
                    target = self.outdir / f"{name}.dim"
                    target.touch()
                    return target
                return operator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master, slave = root / "master.SAFE", root / "slave.SAFE"
            master.mkdir()
            slave.mkdir()
            with patch("sarpyx.snapflow.insar.GPT", FakeGPT):
                result = run_insar_pipeline(
                    master, slave, root / "out",
                    recipe=s1_insar.steps(mask_out_area_without_elevation=False,
                                          nodata_value_at_sea=False),
                )
            self.assertEqual(Path(result).name, "terrain_correction.dim")
            by_name = dict(calls)
            self.assertFalse(by_name["topsar_coregistration"]["mask_out_area_without_elevation"])
            self.assertFalse(by_name["terrain_correction"]["nodata_value_at_sea"])
            self.assertIn("interferogram", by_name)
            self.assertIn("topo_phase_removal", by_name)


if __name__ == "__main__":
    unittest.main()
