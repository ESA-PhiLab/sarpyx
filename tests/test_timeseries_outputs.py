"""Tiny, offline unittest fixtures. No GPT, PBS, downloads or real SAR inputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import zarr

from sarpyx.utils.timeseries import (
    find_assets, load_observation, normalized_fields, pair_source_bands, publish_tile,
)
from sarpyx.utils.timeseries_metadata import (
    export_metadata, processed_documents, product_id,
)
from sarpyx.utils.timeseries_membership import MembershipIndex

REFERENCE = "S1A_IW_SLC__1SDV_20200101T000000_20200101T000025_000001"
SECONDARY = "S1A_IW_SLC__1SDV_20200113T000000_20200113T000025_000002"


def native(path, pair=False, offset=0):
    root = zarr.create_group(str(path), zarr_format=3)
    root.attrs.update({
        "tile": "123U_456R", "pass_direction": "ASC", "mission": "S1A",
        "transform": [10., 0., 500000. + offset, 0., -10., 4600000.],
        "epsg": 32633, "polarizations": ["VV"],
        "acquisition_time": "2020-01-01T00:00:00Z",
        "subap_feature_bands": [] if pair else ["branch_feature_VV"],
    })
    bands = root.create_group("bands")
    values = {
        "i_VV": np.ones((3, 4), dtype=np.float32),
        "q_VV": np.full((3, 4), 2, dtype=np.float32),
        "branch_feature_VV": np.full((3, 4), .7, dtype=np.float32),
        "elevation": np.full((3, 4), 100, dtype=np.float32),
        "localIncidenceAngle": np.full((3, 4), 30, dtype=np.float32),
        "Intensity_VV": np.full((3, 4), 5, dtype=np.float32),
    }
    if pair:
        values = {
            "i_ifg_VV_01Jan2020_13Jan2020": values["i_VV"],
            "q_ifg_VV_01Jan2020_13Jan2020": values["q_VV"],
            "coh_VV_01Jan2020_13Jan2020": np.zeros((3, 4), dtype=np.float32),
            "Phase_ifg_VV_01Jan2020_13Jan2020": np.ones((3, 4), dtype=np.float32),
            # Even if a source operator propagates these, never publish them.
            "i_VV": values["i_VV"], "elevation": values["elevation"],
        }
    for name, value in values.items():
        bands.create_array(name, data=value)
    return root


def snap_topo_names(swath="IW1", dates="19Jan2025_12Feb2025"):
    """Declared band layout observed in the five failed HPC TOPO products."""
    return {
        name for pol in ("VH", "VV") for name in (
            f"i_ifg_{pol}_{dates}", f"q_ifg_{pol}_{dates}",
            f"Phase_ifg_srd_{pol}_{dates}", f"coh_{swath}_{pol}_{dates}",
            f"Intensity_ifg_{pol}_{dates}_ifg_srd_{pol}_{dates}",
        )
    }


class TimeseriesTests(unittest.TestCase):
    def test_pair_child_paths_use_parent_output_once(self):
        script = Path(__file__).resolve().parents[2] / "scripts/worldsar_process_timeseries_pair.sh"
        text = script.read_text()
        start = text.index('    INSAR_SUBSWATH="$swath" REQUIRED_CELL=""')
        end = text.index('      bash "$SCRIPT_DIR/worldsar_process_timeseries_pair.sh"', start)
        assignments = text[start:end]
        # Execute the actual production assignments: Bash evaluates these left
        # to right, so referencing PAIR_OUTPUT after assigning it doubles IW1.
        result = subprocess.run(
            ["bash", "-c", 'PAIR_OUTPUT=/parent; WORK_OUTPUT=/parent; swath=IW1; '
             'staging=/stage; PAIR_LOG=/log/run.log; ' + assignments +
             """      bash -c 'printf "%s\\n%s\\n" "$PAIR_OUTPUT" "$PAIR_TILES_DIR"'"""],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout.splitlines(), ["/parent/IW1", "/parent/IW1/tiles"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / "database.parquet"
        self.pairs = self.root / "pairs.parquet"
        rows = []
        for series in ("631", "962"):
            for identity, day in ((REFERENCE, 1), (SECONDARY, 13)):
                rows.append({
                    "major_tom_id": "123U_456R", "modality_id": f"S1_{series}",
                    "modality": "S1", "acquisition_mode": "IW",
                    "product_name": identity + ".SAFE",
                    "datetime": datetime(2020, 1, day, tzinfo=timezone.utc),
                })
        pq.write_table(pa.Table.from_pylist(rows), self.database)
        pq.write_table(pa.Table.from_pylist([{
            "series_id": "631", "major_tom_id": "123U_456R",
            "modality_id": "S1_631", "modality": "S1", "acquisition_mode": "IW",
            "pair_name": "source-pair-name",
            "master_product_name": REFERENCE + ".SAFE",
            "slave_product_name": SECONDARY + ".SAFE",
            "master_datetime": datetime(2020, 1, 1, tzinfo=timezone.utc),
            "slave_datetime": datetime(2020, 1, 13, tzinfo=timezone.utc),
            "relative_path": "sar/S1/631/paired_products/source-pair-name/123U_456R.h5",
        }]), self.pairs)

    def test_pair_band_selection(self):
        names = {"i_ifg_IW1_VV_a_b", "q_ifg_IW1_VV_a_b", "coh_IW1_VV_a_b",
                 "Phase_ifg_IW1_VV_a_b", "i_VV", "elevation"}
        self.assertEqual(set(pair_source_bands(names)), names - {"Phase_ifg_IW1_VV_a_b", "i_VV", "elevation"})
        with self.assertRaises(ValueError):
            pair_source_bands({"i_ifg_VV_a", "coh_VV_a"})

    def test_snap_virtual_srd_phase_is_not_an_extra_observation(self):
        for swath in ("IW1", "IW2", "IW3"):
            with self.subTest(swath=swath):
                names = snap_topo_names(swath)
                selected = pair_source_bands(names)
                self.assertEqual(len(selected), 6)
                self.assertFalse(any(name.lower().startswith(("phase", "intensity")) for name in selected))
                self.assertEqual(sum(name.startswith("coh_") for name in selected), 2)

    def test_pair_matching_still_rejects_wrong_dates_and_ambiguous_bands(self):
        cases = [
            {"i_ifg_VV_a_b", "q_ifg_VV_a_b", "Phase_ifg_srd_VV_a_b", "coh_IW1_VV_a_c"},
            {"i_ifg_VV_a_b", "Phase_ifg_srd_VV_a_b", "coh_IW1_VV_a_b"},
            {"i_ifg_VV_a_b", "q_ifg_VV_a_b", "coh_IW1_VV_a_b", "coherence_VV_a_b"},
        ]
        for names in cases:
            with self.subTest(names=names), self.assertRaises(ValueError):
                pair_source_bands(names)

    def _validator(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / "validate_worldsar_pair.py"
        spec = importlib.util.spec_from_file_location("worldsar_pair_validation_fixture", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_validator_accepts_snap_srd_alias_and_checks_real_iq_signal(self):
        validator = self._validator()
        names = snap_topo_names()
        validator.require_pair_bands(names, "synthetic SNAP TOPO")
        root = zarr.create_group(str(self.root / "srd.zarr"), zarr_format=3)
        group = root.create_group("bands")
        for name in names:
            if name.startswith("Intensity"):
                continue
            data = np.full((3, 4), .5, dtype=np.float32)
            group.create_array(name, data=data)
        validator.validate_pair_signal(group, self.root / "srd.zarr")
        fields, _, _ = normalized_fields(root, "pair")
        self.assertEqual(fields["interferogram"][0].shape, (1, 2, 3, 4))
        for name in group.keys():
            if name.startswith(("i_ifg_", "q_ifg_")):
                group[name][:] = 0
        # Positive virtual phase/coherence must not hide empty actual I/Q.
        with self.assertRaises(SystemExit):
            validator.validate_pair_signal(group, self.root / "srd.zarr")

    def test_phase_only_srd_alias_with_swath_coherence(self):
        root = zarr.create_group(str(self.root / "phase.zarr"), zarr_format=3)
        bands = root.create_group("bands")
        bands.create_array("Phase_ifg_srd_VV_a_b", data=np.ones((2, 2), dtype=np.float32))
        bands.create_array("coh_IW2_VV_a_b", data=np.full((2, 2), .4, dtype=np.float32))
        self._validator().require_pair_bands(set(bands.keys()), "phase-only")
        fields, _, _ = normalized_fields(root, "pair")
        self.assertIn("wrapped_phase", fields)

    def test_pair_contract_and_zero_coherence(self):
        group = native(self.root / "pair.zarr", pair=True)
        fields, elevation, shape = normalized_fields(group, "pair")
        self.assertEqual(fields["interferogram"][0].shape, (1, 1, 3, 4))
        np.testing.assert_array_equal(fields["interferogram"][0], 1 + 2j)
        self.assertTrue(fields["finite_support"][0].all())
        self.assertNotIn("wrapped_phase", fields)
        self.assertNotIn("gslc", fields)
        self.assertIsNone(elevation)

    def test_phase_only_fallback(self):
        group = native(self.root / "pair.zarr", pair=True)
        for name in list(group["bands"].keys()):
            if name.startswith(("i_ifg_", "q_ifg_")):
                del group["bands"][name]
        fields, _, _ = normalized_fields(group, "pair")
        self.assertIn("wrapped_phase", fields)
        self.assertNotIn("interferogram", fields)

    def test_runtime_resolves_pair_bands_before_tc(self):
        from sarpyx.snapflow.insar import InSARContext, run_insar_step
        from sarpyx.snapflow.runtime import PipelineStep
        ctx = InSARContext(master=Path("a"), slave=Path("b"), outdir=self.root, gpt_path="unused")
        ctx.saved["topo"] = Path("topo.dim")
        selected = {"i_ifg_VV_a", "q_ifg_VV_a", "coh_VV_a"}
        with patch("sarpyx.snapflow.insar._dimap_bands", return_value=selected | {"i_VV"}), \
             patch("sarpyx.snapflow.insar._gpt") as factory:
            factory.return_value.terrain_correction.return_value = str(self.root / "TC.dim")
            run_insar_step(ctx, PipelineStep("TerrainCorrection", {
                "source_ref": "topo", "pair_bands_only": True, "save_dem": False,
            }))
            params = factory.return_value.terrain_correction.call_args.kwargs
            self.assertEqual(set(params["source_bands"]), selected)
            self.assertNotIn("pair_bands_only", params)

    def test_tops_swath_paths_are_preserved_only_in_timeseries_mode(self):
        from sarpyx.snapflow import tiling_runtime
        ctx = SimpleNamespace(
            original_product=Path("input.SAFE"), gpt_kwargs={},
            metadata={"grid_path": self.root / "grid.geojson",
                      "cuts_outdir": self.root / "tiles", "product_mode": "S1TOPS",
                      "product_name": "input.SAFE", "preserve_swath_tiles": True},
        )
        with patch.object(tiling_runtime.config, "tiling", True), \
             patch.object(tiling_runtime, "_resolve_tiling_wkt", return_value="fixture"), \
             patch.object(tiling_runtime, "_write_tiling_manifest"), \
             patch.object(tiling_runtime, "validate_worldsar_zarr_tile_group",
                          return_value={"rows": [], "results": []}), \
             patch.object(tiling_runtime, "_run_tiling", return_value={
                 "name": "fixture", "report_path": self.root / "report.json",
                 "cuts_dir": self.root / "tiles", "cut_failed": False}) as cut:
            tiling_runtime.run_tiling_step(ctx, self.root / "TC.dim", swath="IW2", collect=True)
            self.assertEqual(cut.call_args.args[4], self.root / "tiles" / "input.SAFE" / "IW2")
            ctx.metadata["preserve_swath_tiles"] = False
            tiling_runtime.run_tiling_step(ctx, self.root / "TC.dim", swath="IW2", collect=True)
            self.assertEqual(cut.call_args.args[4], self.root / "tiles" / "input.SAFE")

    def test_hook_keeps_requested_scalars_and_feature_attributes(self):
        from sarpyx.hooks.subap_features import SubapFeatureConfig
        from sarpyx.hooks.worldsar import make_worldsar_zarr_tile_hook
        from sarpyx.snapflow.tile_writers import TilePayload
        payload = TilePayload(
            tile_name="cell", output_path=self.root / "cell.zarr",
            arrays={"feature_VV": np.ones((2, 2))},
            abstract_attrs={"PASS": "ASCENDING", "radar_frequency": 5405,
                            "range_bandwidth": 50},
            band_attrs={"feature_VV": {"unit": "1", "new_branch_definition": "keep-me"}},
        )
        hook = make_worldsar_zarr_tile_hook(
            "input.SAFE", product_mode="S1TOPS", subap_features=SubapFeatureConfig(enabled=False))
        result = hook(payload)
        self.assertEqual(result.root_attrs["source_metadata"]["range_bandwidth"], 50)
        self.assertEqual(result.band_attrs["feature_VV"]["new_branch_definition"], "keep-me")

    def test_acquisition_features_unchanged(self):
        fields, elevation, shape = normalized_fields(native(self.root / "a.zarr"), "acquisition")
        self.assertEqual(fields["gslc"][0].dtype, np.complex64)
        self.assertEqual(fields["features/subaperture/branch_feature_VV"][0].shape, (1, 3, 4))
        self.assertIn("geometry/local_incidence_angle", fields)
        group = zarr.open_group(str(self.root / "a.zarr"), mode="a")
        group["bands"].create_array("Entropy", data=np.ones((3, 4), dtype=np.float32))
        fields, _, _ = normalized_fields(group, "acquisition")
        self.assertIn("features/polarimetry/entropy", fields)
        self.assertNotIn("entropy", fields)
        self.assertNotIn("Intensity_VV", fields)
        self.assertEqual(elevation[0].shape, (3, 4))

    def test_unknown_bands_fail_instead_of_silent_loss(self):
        group = native(self.root / "a.zarr")
        group["bands"].create_array("new_science_field", data=np.ones((3, 4)))
        with self.assertRaisesRegex(ValueError, "Unclassified"):
            normalized_fields(group, "acquisition")

    def test_raw_looks_are_not_accepted_as_features(self):
        group = native(self.root / "a.zarr")
        group["bands"].create_array("i_VV_SA1", data=np.ones((3, 4)))
        with self.assertRaises(ValueError):
            normalized_fields(group, "acquisition")

    def _publish(self, path, identity=REFERENCE, metadata_refs=None):
        return publish_tile(
            path, self.root / "dataset", kind="acquisition", identity=identity,
            swath="IW1", grid_id="grid-test", acquisition_ids=[identity],
            metadata_refs=metadata_refs or {}, processing={"dem": {"name": "test"}},
        )

    def test_roundtrip_reuse_and_no_alignment_changes(self):
        first = self.root / "a.zarr"
        native(first, offset=3.25)
        asset = self._publish(first)
        second = self._publish(first)
        self.assertEqual(asset, second)
        loaded = load_observation(self.root / "dataset", asset["uri"])
        group = loaded["data"]
        self.assertEqual(group.attrs["transform"][2], 500003.25)
        self.assertEqual(group["gslc"].metadata.dimension_names, ("time", "pol", "y", "x"))
        self.assertEqual(loaded["terrain"]["elevation"].shape, (3, 4))
        np.testing.assert_array_equal(group["x"][:], [500008.25, 500018.25, 500028.25, 500038.25])
        other = self.root / "b.zarr"
        native(other, offset=4.25)
        self._publish(other, SECONDARY)
        # Different sampling must not be falsely deduplicated or resampled.
        self.assertEqual(len(list((self.root / "dataset").glob("data/static/*/*.zarr"))), 2)

    def test_static_deduplicated_across_acquisitions(self):
        a, b = self.root / "a.zarr", self.root / "b.zarr"
        native(a)
        native(b)
        self._publish(a)
        self._publish(b, SECONDARY)
        self.assertEqual(len(list((self.root / "dataset").glob("data/static/*/*.zarr"))), 1)

    def test_metadata_units_models_and_applied_orbits(self):
        root = ET.fromstring("""<MDElem name="Abstracted_Metadata">
          <MDATTR name="radar_frequency" unit="MHz">5405</MDATTR>
          <MDATTR name="range_bandwidth" unit="MHz">50</MDATTR>
          <MDElem name="orbit_state_vectors"><MDElem name="orbit_vector1">
            <MDATTR name="orbit_vector_time">2020-01-01T00:00:00Z</MDATTR>
            <MDATTR name="x_pos">7000000</MDATTR><MDATTR name="y_pos">0</MDATTR>
            <MDATTR name="z_pos">0</MDATTR><MDATTR name="x_vel">0</MDATTR>
            <MDATTR name="y_vel">7500</MDATTR><MDATTR name="z_vel">0</MDATTR>
          </MDElem></MDElem>
          <MDElem name="Doppler_Centroid_Coefficients">
            <MDATTR name="coefficient">1.25</MDATTR>
          </MDElem>
        </MDElem>""")
        metadata = export_metadata(self.root, REFERENCE, [("IW1/final.dim", root)])
        table = pq.read_table(metadata / "summary.parquet").to_pylist()[0]
        self.assertEqual(table["radar_frequency_hz"], 5.405e9)
        self.assertEqual(table["range_bandwidth_hz"], 50e6)
        orbit = pq.read_table(metadata / "orbit_states.parquet").to_pylist()[0]
        self.assertEqual(orbit["orbit_kind"], "applied_snap")
        self.assertEqual(orbit["vy_m_s"], 7500)
        self.assertTrue(any("Doppler" in row["path"] for row in pq.read_table(metadata / "fields.parquet").to_pylist()))
        self.assertEqual(metadata, export_metadata(self.root, REFERENCE, [("IW1/final.dim", root)]))

    def test_processed_metadata_in_isolated_tops_directory(self):
        dim = self.root / "worldsar_tmp_fixture" / "IW1" / "final_TC.dim"
        dim.parent.mkdir(parents=True)
        dim.write_text("""<Dimap_Document><Dataset_Sources>
          <MDElem name="Abstracted_Metadata"><MDATTR name="radar_frequency">5405</MDATTR></MDElem>
          <MDElem name="Original_Product_Metadata"><MDATTR name="unused">original</MDATTR></MDElem>
          </Dataset_Sources></Dimap_Document>""")
        documents = list(processed_documents(self.root))
        self.assertTrue(any(source.endswith("final_TC.dim") for source, _ in documents))
        self.assertTrue(any(node.get("name") == "Abstracted_Metadata" for _, node in documents))
        self.assertFalse(any(node.get("name") == "Original_Product_Metadata" for _, node in documents))
        self.assertFalse(any("worldsar_tmp_" in source for source, _ in documents))

    def test_source_annotation_and_doppler(self):
        root = ET.fromstring("""<product>
          <radarFrequency>5405000000</radarFrequency>
          <rangeProcessing><processingBandwidth>50000000</processingBandwidth></rangeProcessing>
          <azimuthProcessing><processingBandwidth>1200</processingBandwidth></azimuthProcessing>
          <dcEstimate><azimuthTime>2020-01-01T00:00:00Z</azimuthTime>
            <t0>0.005</t0><dataDcPolynomial>10 20 30</dataDcPolynomial>
          </dcEstimate></product>""")
        metadata = export_metadata(self.root, REFERENCE, [("s1-iw1-vv.xml", root)])
        summary = pq.read_table(metadata / "summary.parquet").to_pylist()[0]
        self.assertEqual(summary["range_bandwidth_hz"], 50e6)
        self.assertEqual(summary["azimuth_bandwidth_hz"], 1200)
        row = pq.read_table(metadata / "doppler.parquet").to_pylist()[0]
        self.assertEqual(row["coefficients"], [10, 20, 30])

    def test_sampled_look_vectors(self):
        root = ET.fromstring("""<product><orbitList>
          <orbit><time>2020-01-01T00:00:00Z</time><position><x>7000000</x><y>0</y><z>0</z></position></orbit>
          <orbit><time>2020-01-01T00:00:10Z</time><position><x>7000000</x><y>100</y><z>0</z></position></orbit>
          </orbitList><geolocationGridPoint>
            <azimuthTime>2020-01-01T00:00:05Z</azimuthTime>
            <latitude>0</latitude><longitude>0</longitude><height>0</height>
            <line>1</line><pixel>2</pixel>
          </geolocationGridPoint></product>""")
        metadata = export_metadata(self.root, REFERENCE, [("iw1.xml", root)])
        look = pq.read_table(metadata / "look_vectors.parquet").to_pylist()[0]
        self.assertGreater(look["up"], .99)
        self.assertAlmostEqual(sum(look[key] ** 2 for key in ("east", "north", "up")), 1)
        self.assertEqual(look["orbit_kind"], "source_annotation")

    def test_product_id(self):
        self.assertEqual(product_id("a.SAFE.zip"), "a")
        with self.assertRaises(ValueError):
            product_id("bad name.SAFE")

    def test_timeseries_recipe_does_not_export_ancillaries(self):
        from sarpyx.pipelines.double_product.s1_insar import steps
        legacy = next(step for step in steps() if step.name == "TerrainCorrection")
        current = next(step for step in steps(timeseries_output=True) if step.name == "TerrainCorrection")
        self.assertNotIn("save_dem", legacy.params)
        self.assertFalse(current.params["save_dem"])
        self.assertFalse(current.params["save_local_incidence_angle"])
        self.assertTrue(current.params["pair_bands_only"])

    def test_publisher_end_to_end_tiny_fixture(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / "publish_worldsar_timeseries.py"
        spec = importlib.util.spec_from_file_location("worldsar_publish_fixture", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        product = self.root / f"{REFERENCE}.SAFE"
        (product / "annotation").mkdir(parents=True)
        (product / "annotation" / "s1-iw1-vv.xml").write_text(
            "<product><radarFrequency>5405000000</radarFrequency></product>")
        run = self.root / "run"
        tile = run / "tiles" / product.name / "IW1" / "cell.zarr"
        native(tile)
        outside = tile.parent / "outside.zarr"
        native(outside).attrs["tile"] = "outside-cell"
        dim = run / "worldsar_tmp_fixture" / "IW1" / "final_TC.dim"
        dim.parent.mkdir(parents=True)
        dim.write_text("""<Dimap_Document><Dataset_Sources>
          <MDElem name="Abstracted_Metadata"><MDATTR name="radar_frequency" unit="MHz">5405</MDATTR></MDElem>
          </Dataset_Sources></Dimap_Document>""")
        dim.with_suffix(".data").mkdir()
        (dim.with_suffix(".data") / "generated.img").write_bytes(b"fixture")
        (run / "keep.log").write_text("diagnostics")
        grid = self.root / "grid.geojson"
        grid.write_text("{}")
        args = argparse.Namespace(
            database=self.database, pairs=self.pairs,
            dataset=self.root / "dataset", run_dir=run, tiles=run / "tiles",
            product=[product], kind="acquisition", grid=grid, swath=None,
            use_esd=True, run_id="fixture", receipt=run / "receipt.json", discard_intermediates=False,
        )
        manifest = module.publish(args)
        self.assertEqual(manifest["assets"][0]["swath"], "IW1")
        self.assertFalse(tile.exists())
        self.assertFalse(outside.exists())
        self.assertEqual(manifest["skipped_unassigned_cells"], ["outside-cell"])
        self.assertTrue((args.dataset / manifest["assets"][0]["uri"]).exists())
        self.assertEqual(len(list(args.dataset.rglob("assets.parquet"))), 1)
        self.assertEqual(len(find_assets(args.dataset, "123U_456R", kind="acquisition", swath="IW1")), 1)
        self.assertEqual(find_assets(args.dataset, "123U_456R", kind="pair"), [])
        for series in ("631", "962"):
            grouped = find_assets(args.dataset, "123U_456R", series_id=series)
            self.assertEqual(len(grouped), 1)
            self.assertEqual(grouped[0]["uri"], manifest["assets"][0]["uri"])
        metadata = load_observation(args.dataset, manifest["assets"][0]["uri"])["metadata"]
        self.assertIn("summary", metadata["acquisition"])
        self.assertIn("acquisition_processing", metadata)
        # Re-running produces only another receipt, not another scientific copy.
        dim.parent.parent.rename(run / "worldsar_tmp_another_attempt")
        dim = run / "worldsar_tmp_another_attempt" / "IW1" / "final_TC.dim"
        native(tile)
        args.receipt = run / "receipt-again.json"
        args.run_id = "fixture-again"
        args.discard_intermediates = True
        module.publish(args)
        self.assertEqual(len(list(args.dataset.glob("data/acquisitions/*/IW1/*/*.zarr"))), 1)
        self.assertEqual(len(list(args.dataset.rglob("assets.parquet"))), 1)
        self.assertFalse(dim.exists())
        self.assertFalse(dim.with_suffix(".data").exists())
        self.assertTrue((run / "keep.log").exists())

    def test_failed_normalization_keeps_native_input(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / "publish_worldsar_timeseries.py"
        spec = importlib.util.spec_from_file_location("worldsar_publish_failure_fixture", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        product = self.root / f"{REFERENCE}.SAFE"
        (product / "annotation").mkdir(parents=True)
        (product / "annotation" / "iw1.xml").write_text("<product><radarFrequency>1</radarFrequency></product>")
        run = self.root / "run"
        tile = run / "tiles" / "IW1" / "a.zarr"
        group = native(tile)
        group["bands"].create_array("unexpected", data=np.ones((3, 4)))
        grid = self.root / "grid.geojson"
        grid.write_text("{}")
        args = argparse.Namespace(
            database=self.database, pairs=self.pairs,
            dataset=self.root / "dataset", run_dir=run, tiles=run / "tiles",
            product=[product], kind="acquisition", grid=grid, swath=None,
            use_esd=True, run_id="fixture", receipt=run / "receipt.json",
        )
        with self.assertRaisesRegex(ValueError, "Unclassified"):
            module.publish(args)
        self.assertTrue(tile.exists())
        self.assertFalse(args.receipt.exists())

    def test_pair_publisher_end_to_end(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / "publish_worldsar_timeseries.py"
        spec = importlib.util.spec_from_file_location("worldsar_pair_fixture", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        products = []
        for identity in (REFERENCE, SECONDARY):
            product = self.root / f"{identity}.SAFE"
            (product / "annotation").mkdir(parents=True)
            (product / "annotation" / "iw1.xml").write_text(
                "<product><radarFrequency>5405000000</radarFrequency></product>")
            products.append(product)
        run = self.root / "run"
        tile = run / "tiles" / "cell.zarr"
        native(tile, pair=True)
        dim = run / "pair" / "final_TOPO_TC.dim"
        dim.parent.mkdir(parents=True)
        dim.write_text("""<Dimap_Document><Dataset_Sources>
          <MDElem name="Abstracted_Metadata"><MDElem name="Baselines">
          <MDATTR name="Perp Baseline" unit="m">123.5</MDATTR>
          </MDElem></MDElem></Dataset_Sources></Dimap_Document>""")
        grid = self.root / "grid.geojson"
        grid.write_text("{}")
        args = argparse.Namespace(
            database=self.database, pairs=self.pairs,
            dataset=self.root / "dataset", run_dir=run, tiles=run / "tiles",
            product=products, kind="pair", grid=grid, swath="IW1",
            use_esd=True, run_id="fixture", receipt=run / "receipt.json",
        )
        manifest = module.publish(args)
        loaded = load_observation(args.dataset, manifest["assets"][0]["uri"])
        self.assertEqual(loaded["data"]["temporal_baseline"][0], 12 * 86400)
        self.assertTrue(manifest["assets"][0]["uri"].startswith(
            f"data/pairs/{REFERENCE}__{SECONDARY}/IW1/123U_456R/"))
        self.assertNotIn("gslc", loaded["data"])
        baseline = loaded["metadata"]["pair_processing"]["baselines"].to_pylist()[0]
        self.assertEqual(baseline["numeric_value"], 123.5)
        self.assertEqual(baseline["unit"], "m")
        self.assertEqual(len(manifest["series_membership_refs"]), 2)
        explicit = find_assets(args.dataset, "123U_456R", kind="pair", series_id="631")
        inferred = find_assets(args.dataset, "123U_456R", kind="pair", series_id="962")
        self.assertEqual(explicit[0]["pair_name"], "source-pair-name")
        self.assertEqual(inferred[0]["membership_source"], "database.parquet:endpoint_intersection")
        self.assertEqual(explicit[0]["uri"], inferred[0]["uri"])

    def test_membership_is_per_cell_and_pair_requires_both_endpoints(self):
        table = pq.read_table(self.database)
        row = table.to_pylist()[0]
        row["major_tom_id"] = "other-cell"
        pq.write_table(pa.concat_tables([table, pa.Table.from_pylist([row], schema=table.schema)]), self.database)
        acquisition = MembershipIndex(self.database, self.pairs, [REFERENCE])
        pair = MembershipIndex(self.database, self.pairs, [REFERENCE, SECONDARY])
        self.assertEqual(len(acquisition.for_cell("other-cell")), 1)
        self.assertEqual(pair.for_cell("other-cell"), [])
        self.assertEqual(acquisition.for_cell("unassigned-cell"), [])


if __name__ == "__main__":
    unittest.main()
