import sys
from pathlib import Path

from sarpyx.cli.main import create_parser as create_main_parser
from sarpyx.cli.pipeline import create_parser, parse_params, run
from sarpyx.pipelines import runner as pipeline_runner


def test_parse_params_coerces_json_values():
    assert parse_params(["subswath=IW2", "use_esd=false", "selected_polarisations=[\"VV\"]", "count=2"]) == {
        "subswath": "IW2",
        "use_esd": False,
        "selected_polarisations": ["VV"],
        "count": 2,
    }


def test_top_level_parser_exposes_pipeline_command():
    args = create_main_parser().parse_args(["pipeline", "--list"])

    assert args.command == "pipeline"
    assert args.list_pipelines is True


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
