from pathlib import Path

import pytest

from sarpyx.cli.pipeline import create_parser, main


def test_create_parser_parses_run_arguments(tmp_path: Path):
    args = create_parser().parse_args(
        [
            "run",
            str(tmp_path / "pipeline.yaml"),
            "--pipeline",
            "pair",
            "--master",
            "/tmp/a.zip",
            "--slave",
            "/tmp/b.zip",
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )

    assert args.command == "run"
    assert args.pipeline == "pair"
    assert args.master == Path("/tmp/a.zip")
    assert args.slave == Path("/tmp/b.zip")
    assert args.outdir == tmp_path / "out"
    assert args.dry_run is True


def test_validate_command_reports_valid_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        """
version: 1
pipelines:
  preprocess:
    inputs:
      product: /tmp/product.dim
    steps:
      - id: calibration
        op: Calibration
""",
        encoding="utf-8",
    )

    assert main(["validate", str(config)]) == 0
    assert "valid:" in capsys.readouterr().out


def test_list_command_prints_pipeline_names(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        """
version: 1
pipelines:
  zeta:
    steps:
      - id: z
        op: Calibration
  alpha:
    steps:
      - id: a
        op: Calibration
""",
        encoding="utf-8",
    )

    assert main(["list", str(config)]) == 0
    assert capsys.readouterr().out.splitlines() == ["alpha", "zeta"]


def test_run_command_uses_default_pipeline(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    product = tmp_path / "input.dim"
    product.write_text("placeholder", encoding="utf-8")
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        """
version: 1
default_pipeline: beta
pipelines:
  alpha:
    inputs:
      product: null
    steps:
      - id: alpha_cal
        op: Calibration
        source: product
  beta:
    inputs:
      product: null
    steps:
      - id: beta_cal
        op: Calibration
        source: product
""",
        encoding="utf-8",
    )

    assert main(["run", str(config), "--set-input", f"product={product}", "--dry-run"]) == 0
    assert "planned: beta.beta_cal" in capsys.readouterr().out


def test_direct_run_binds_master_slave_and_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        """
version: 1
default_pipeline: pair
pipelines:
  pair:
    inputs:
      master: null
      slave: null
    steps:
      - id: coreg
        op: Back-Geocoding
        sources:
          master_product: master
          slave_product: slave
""",
        encoding="utf-8",
    )

    outdir = tmp_path / "out"
    assert (
        main(
            [
                str(config),
                "--master",
                "/tmp/master.SAFE",
                "--slave",
                "/tmp/slave.SAFE",
                "--output",
                str(outdir),
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "planned: pair.coreg" in output
    assert str(outdir / "pair" / "coreg" / "coreg.dim") in output


def test_run_command_accepts_master_slave_shortcuts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        """
version: 1
pipelines:
  pair:
    inputs:
      master: null
      slave: null
    steps:
      - id: coreg
        op: Back-Geocoding
        sources:
          master_product: master
          slave_product: slave
""",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "run",
                str(config),
                "--master",
                "/tmp/master.SAFE",
                "--slave",
                "/tmp/slave.SAFE",
                "--output",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert "planned: pair.coreg" in capsys.readouterr().out


def test_set_input_requires_name_value(tmp_path: Path):
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        """
version: 1
pipelines:
  preprocess:
    steps:
      - id: calibration
        op: Calibration
""",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        main(["run", str(config), "--set-input", "broken", "--dry-run"])

    assert excinfo.value.code == 2
