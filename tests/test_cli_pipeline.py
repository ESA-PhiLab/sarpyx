import os
import subprocess
import sys
from pathlib import Path

from sarpyx.cli.main import create_parser as create_main_parser
from sarpyx.cli import pipeline as pipeline_cli
from sarpyx.cli.pipeline import create_parser, parse_params, run
from sarpyx.pipelines import runner as pipeline_runner


def test_parse_params_coerces_json_values():
    assert parse_params(["subswath=IW2", "use_esd=false", "selected_polarisations=[\"VV\"]", "count=2"]) == {
        "subswath": "IW2",
        "use_esd": False,
        "selected_polarisations": ["VV"],
        "count": 2,
    }


def test_parse_params_accepts_python_style_bool_values():
    assert parse_params(["use_esd=False", "orbit_continue_on_fail=True"]) == {
        "use_esd": False,
        "orbit_continue_on_fail": True,
    }


def test_top_level_parser_exposes_pipeline_command():
    args = create_main_parser().parse_args(["pipeline", "--list"])

    assert args.command == "pipeline"
    assert args.list_pipelines is True


def test_top_level_cli_help_does_not_import_worldsar_runtime():
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}{os.pathsep}{env.get('PYTHONPATH', '')}"
    script = """
import importlib.abc
import sys

class BlockHeavyRuntime(importlib.abc.MetaPathFinder):
    blocked = {
        "sarpyx.snapflow.runner",
        "sarpyx.hooks.subap_features",
        "sarpyx.hooks.worldsar",
    }

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.blocked or fullname.startswith("scipy"):
            raise RuntimeError(f"blocked heavy import: {fullname}")
        return None

sys.meta_path.insert(0, BlockHeavyRuntime())
from sarpyx.cli.main import create_parser

help_text = create_parser().format_help()
assert "worldsar" in help_text
assert "pipeline" in help_text
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_pipeline_cli_dispatches_builtin_double_product(monkeypatch, tmp_path: Path):
    master = tmp_path / "master.SAFE"
    slave = tmp_path / "slave.SAFE"
    master.mkdir()
    slave.mkdir()
    captured = {}

    def fake_run_insar_pipeline(master_path, slave_path, outdir, **kwargs):
        captured["master"] = Path(master_path)
        captured["slave"] = Path(slave_path)
        captured["outdir"] = Path(outdir)
        captured["gpt_path"] = kwargs["gpt_path"]
        captured["cache_size"] = kwargs["cache_size"]
        captured["recipe"] = kwargs["recipe"]
        captured["metadata"] = kwargs["metadata"]
        return Path(outdir) / "pair" / "tc.dim"

    monkeypatch.setattr(pipeline_runner, "run_insar_pipeline", fake_run_insar_pipeline)
    args = create_parser().parse_args(
        [
            "s1_insar",
            "--master",
            str(master),
            "--slave",
            str(slave),
            "--output",
            str(tmp_path / "out"),
            "--product-wkt",
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "--grid-path",
            str(tmp_path / "grid.geojson"),
            "--gpt-path",
            sys.executable,
            "--param",
            "subswath=IW2",
            "--param",
            "use_esd=false",
        ]
    )

    assert run(args) == 0
    assert captured["master"] == master
    assert captured["slave"] == slave
    assert captured["outdir"] == tmp_path / "out"
    assert captured["gpt_path"] == sys.executable
    assert captured["cache_size"] == "8G"
    assert captured["recipe"][0].params["subswath"] == "IW2"
    assert captured["recipe"][4].params["use_esd"] is False
    assert captured["recipe"][-1].name == "WorldSARTiling"
    assert captured["metadata"]["cuts_outdir"] == tmp_path / "out" / "tiles"
    assert captured["metadata"]["grid_path"] == tmp_path / "grid.geojson"
    assert captured["metadata"]["tile_writer"] == "zarr"


def test_pipeline_cli_prints_warnings_at_end(monkeypatch, tmp_path: Path, capsys):
    master = tmp_path / "master.SAFE"
    slave = tmp_path / "slave.SAFE"
    master.mkdir()
    slave.mkdir()

    monkeypatch.setattr(pipeline_cli, "run_pipeline_target", lambda *_, **__: tmp_path / "out" / "pair" / "tc.dim")
    args = create_parser().parse_args(
        [
            "s1_insar",
            "--master",
            str(master),
            "--slave",
            str(slave),
            "--output",
            str(tmp_path / "out"),
            "--cuts-outdir",
            str(tmp_path / "cuts"),
            "--gpt-path",
            sys.executable,
        ]
    )

    assert run(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Pipeline result:")
    assert lines[1] == "Pipeline warnings:"
    assert "WARNING: --grid-path was not supplied; using the default grid when cuts are requested." in lines
    assert "WARNING: --product-wkt was not supplied; sarpyx will derive the tiling footprint from the processed raster when possible." in lines


def test_insar_pipeline_with_grid_does_not_require_product_wkt(monkeypatch, tmp_path: Path):
    master = tmp_path / "master.SAFE"
    slave = tmp_path / "slave.SAFE"
    grid = tmp_path / "grid.geojson"
    master.mkdir()
    slave.mkdir()
    grid.write_text("{}", encoding="utf-8")
    captured = {}

    def fake_run_insar_pipeline(master_path, slave_path, outdir, **kwargs):
        captured["master"] = Path(master_path)
        captured["slave"] = Path(slave_path)
        captured["recipe"] = kwargs["recipe"]
        captured["metadata"] = kwargs["metadata"]
        return Path(outdir) / "pair" / "tc.dim"

    monkeypatch.setattr(pipeline_runner, "run_insar_pipeline", fake_run_insar_pipeline)

    result = pipeline_runner.run_pipeline_target(
        "s1_insar",
        master=master,
        slave=slave,
        output_dir=tmp_path / "out",
        gpt_path=sys.executable,
        grid_path=grid,
        cuts_outdir=tmp_path / "cuts",
    )

    assert result == tmp_path / "out" / "pair" / "tc.dim"
    assert captured["master"] == master
    assert captured["slave"] == slave
    assert captured["recipe"][0].params["subswath"] is None
    assert captured["recipe"][0].params["selected_polarisations"] is None
    assert "product_wkt" not in captured["metadata"]
    assert captured["metadata"]["product_mode"] == "S1INSAR"
    assert captured["metadata"]["grid_path"] == grid
    assert captured["metadata"]["cuts_outdir"] == tmp_path / "cuts"
    assert captured["metadata"]["tile_writer"] == "zarr"


def test_insar_pipeline_with_cuts_outdir_uses_default_grid(monkeypatch, tmp_path: Path):
    master = tmp_path / "master.SAFE"
    slave = tmp_path / "slave.SAFE"
    default_grid = tmp_path / "grid" / "grid_10km.geojson"
    master.mkdir()
    slave.mkdir()
    default_grid.parent.mkdir()
    default_grid.write_text("{}", encoding="utf-8")
    captured = {}

    def fake_run_insar_pipeline(master_path, slave_path, outdir, **kwargs):
        captured["metadata"] = kwargs["metadata"]
        return Path(outdir) / "pair" / "tc.dim"

    monkeypatch.setattr(pipeline_runner, "run_insar_pipeline", fake_run_insar_pipeline)
    monkeypatch.setattr(pipeline_runner.config, "BASE_PATH", str(tmp_path))

    result = pipeline_runner.run_pipeline_target(
        "s1_insar",
        master=master,
        slave=slave,
        output_dir=tmp_path / "out",
        gpt_path=sys.executable,
        cuts_outdir=tmp_path / "cuts",
    )

    assert result == tmp_path / "out" / "pair" / "tc.dim"
    assert captured["metadata"]["grid_path"] == default_grid
    assert captured["metadata"]["cuts_outdir"] == tmp_path / "cuts"


def test_insar_pair_script_uses_automatic_polarisation_selection_by_default():
    script = Path("scripts/insar_pair.sh").read_text(encoding="utf-8")

    assert "selected_polarisations" not in script


def test_pipeline_runner_loads_external_single_product_file(tmp_path: Path):
    pipeline_file = tmp_path / "custom_pipeline.py"
    pipeline_file.write_text(
        "\n".join(
            [
                "INPUT_KIND = 'single'",
                "OUTPUT_FORMAT = 'BEAM-DIMAP'",
                "def steps(label='ok'):",
                "    return []",
            ]
        ),
        encoding="utf-8",
    )
    product = tmp_path / "product.dim"
    product.write_text("<Dimap_Document/>", encoding="utf-8")

    result = pipeline_runner.run_pipeline_target(
        str(pipeline_file),
        input_path=product,
        output_dir=tmp_path / "out",
        gpt_path=sys.executable,
        params={"label": "custom"},
    )

    assert result == product
