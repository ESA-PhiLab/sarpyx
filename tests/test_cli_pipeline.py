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
            "--set-input",
            "master=/tmp/a.zip",
            "--set-input",
            "slave=/tmp/b.zip",
            "--outdir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )

    assert args.command == "run"
    assert args.pipeline == "pair"
    assert args.inputs == ["master=/tmp/a.zip", "slave=/tmp/b.zip"]
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
