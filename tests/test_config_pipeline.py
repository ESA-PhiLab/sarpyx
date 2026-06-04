from pathlib import Path

import pytest
import yaml

from sarpyx.snapflow.config_pipeline import (
    ConfigPipelineError,
    ConfigPipelineRunner,
    list_config_pipelines,
    validate_pipeline_config,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")
    return path


def test_validate_rejects_pipeline_cycles():
    config = {
        "version": 1,
        "pipelines": {
            "a": {"steps": [{"id": "call_b", "use": "b"}]},
            "b": {"steps": [{"id": "call_a", "use": "a"}]},
        },
    }

    with pytest.raises(ConfigPipelineError, match="cycle"):
        validate_pipeline_config(config)


def test_validate_rejects_unknown_default_pipeline():
    config = {
        "version": 1,
        "default_pipeline": "missing",
        "pipelines": {
            "preprocess": {"steps": [{"id": "calibration", "op": "Calibration"}]},
        },
    }

    with pytest.raises(ConfigPipelineError, match="default_pipeline"):
        validate_pipeline_config(config)


def test_runner_uses_default_pipeline_when_pipeline_omitted(tmp_path: Path):
    product = _touch(tmp_path / "input.dim")
    config = {
        "version": 1,
        "default_pipeline": "beta",
        "pipelines": {
            "alpha": {
                "inputs": {"product": None},
                "steps": [{"id": "alpha_cal", "op": "Calibration", "source": "product"}],
            },
            "beta": {
                "inputs": {"product": None},
                "steps": [{"id": "beta_cal", "op": "Calibration", "source": "product"}],
            },
        },
    }

    result = ConfigPipelineRunner(config, outdir=tmp_path / "out", dry_run=True).run(
        inputs={"product": product},
    )

    assert result.pipeline == "beta"
    assert result.records[0].step_id == "beta_cal"


def test_dry_run_plans_named_nested_pipeline(tmp_path: Path):
    master = _touch(tmp_path / "master.dim")
    slave = _touch(tmp_path / "slave.dim")
    config = {
        "version": 1,
        "defaults": {"format": "BEAM-DIMAP"},
        "pipelines": {
            "preprocess": {
                "inputs": {"product": None},
                "steps": [
                    {
                        "id": "calibration",
                        "op": "Calibration",
                        "source": "product",
                        "params": {"outputSigmaBand": True},
                    }
                ],
            },
            "pair": {
                "inputs": {"master": None, "slave": None},
                "steps": [
                    {"id": "master_pre", "use": "preprocess", "inputs": {"product": "master"}},
                    {"id": "slave_pre", "use": "preprocess", "inputs": {"product": "slave"}},
                    {
                        "id": "coreg",
                        "method": "topsar_coregistration",
                        "sources": {"master_product": "master_pre", "slave_product": "slave_pre"},
                    },
                ],
            },
        },
    }

    result = ConfigPipelineRunner(config, outdir=tmp_path / "out", dry_run=True).run(
        pipeline="pair",
        inputs={"master": master, "slave": slave},
    )

    assert result.output.endswith("coreg/coreg.dim")
    assert [record.step_id for record in result.records] == [
        "calibration",
        "master_pre",
        "calibration",
        "slave_pre",
        "coreg",
    ]
    assert result.records[-1].source_refs == {
        "master_product": str(tmp_path / "out" / "pair" / "master_pre" / "calibration" / "calibration.dim"),
        "slave_product": str(tmp_path / "out" / "pair" / "slave_pre" / "calibration" / "calibration.dim"),
    }


def test_operator_graph_rendering_uses_snap_operator_names(tmp_path: Path):
    product = _touch(tmp_path / "input.dim")
    config = {
        "version": 1,
        "pipelines": {
            "preprocess": {
                "inputs": {"product": None},
                "steps": [
                    {
                        "id": "terrain",
                        "op": "Terrain-Correction",
                        "source": "product",
                        "params": {"demName": "SRTM 3Sec", "saveSelectedSourceBand": False},
                    }
                ],
            }
        },
    }
    runner = ConfigPipelineRunner(config, outdir=tmp_path / "out", dry_run=True)
    result = runner.run(pipeline="preprocess", inputs={"product": product})

    assert result.records[0].graph.endswith("terrain_graph.xml")
    assert result.records[0].output.endswith("terrain.dim")


def test_resume_requires_matching_manifest(tmp_path: Path):
    product = _touch(tmp_path / "input.dim")
    existing = tmp_path / "out" / "preprocess" / "cal" / "cal.dim"
    _touch(existing)
    config = {
        "version": 1,
        "defaults": {"resume": True},
        "pipelines": {
            "preprocess": {
                "inputs": {"product": None},
                "steps": [{"id": "cal", "op": "Calibration", "source": "product"}],
            }
        },
    }

    with pytest.raises(ConfigPipelineError, match="resume manifest"):
        ConfigPipelineRunner(config, outdir=tmp_path / "out").run(
            pipeline="preprocess",
            inputs={"product": product},
        )


def test_list_config_pipelines_is_sorted():
    assert list_config_pipelines({"pipelines": {"z": {}, "a": {}}}) == ["a", "z"]


def test_yaml_example_shape_parses():
    loaded = yaml.safe_load(
        """
        version: 1
        pipelines:
          preprocess:
            inputs:
              product: null
            steps:
              - id: calibration
                op: Calibration
        """
    )

    validate_pipeline_config(loaded)
