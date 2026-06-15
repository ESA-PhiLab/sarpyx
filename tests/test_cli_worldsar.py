import sys
from pathlib import Path

import pytest

from sarpyx.cli.worldsar import create_parser, main


WORLDSAR_DEFAULT_FIELDS = (
    'output_dir',
    'cuts_outdir',
    'gpt_path',
    'grid_path',
    'db_dir',
    'gpt_memory',
    'gpt_cache_size',
    'gpt_parallelism',
    'gpt_timeout',
    'lock_timeout',
    'snap_userdir',
    'orbit_type',
)


def test_create_parser_formats_help() -> None:
    parser = create_parser()
    help_text = parser.format_help()

    assert '--input' in help_text
    assert '--orbit-continue-on-fail' in help_text


def test_create_parser_parses_worldsar_arguments() -> None:
    parser = create_parser()

    args = parser.parse_args(
        [
            '--input',
            '/tmp/product.SAFE',
            '--output',
            '/tmp/output',
            '--cuts-outdir',
            '/tmp/cuts',
            '--db-dir',
            '/tmp/db',
            '--h5-to-zarr-only',
            '--zarr-chunk-size',
            '64',
            '64',
            '--sentinel-subap-feature-window-size',
            '7',
            '--orbit-continue-on-fail',
            '--lock-timeout',
            '15',
        ]
    )

    assert args.product_path == '/tmp/product.SAFE'
    assert args.output_dir == '/tmp/output'
    assert args.cuts_outdir == '/tmp/cuts'
    assert args.db_dir == '/tmp/db'
    assert args.h5_to_zarr_only is True
    assert args.zarr_chunk_size == [64, 64]
    assert args.sentinel_subap_feature_window_size == 7
    assert args.orbit_continue_on_fail is True
    assert args.lock_timeout == 15


def test_create_parser_defaults_are_stable() -> None:
    args = create_parser().parse_args(['--input', '/tmp/product.SAFE'])

    for field in WORLDSAR_DEFAULT_FIELDS:
        assert '/shared/home/vmarsocci' not in str(getattr(args, field))
    assert args.gpt_memory == '16G'
    assert args.gpt_cache_size == '8G'
    assert args.gpt_parallelism == 6
    assert args.tile_writer == "zarr"


def test_validate_runtime_args_rejects_even_subap_feature_window() -> None:
    from sarpyx.snapflow import config

    args = create_parser().parse_args(["--input", "/tmp/product.SAFE", "--sentinel-subap-feature-window-size", "4"])

    with pytest.raises(ValueError, match="sentinel-subap-feature-window-size"):
        config.validate_runtime_args(args)


def test_validate_runtime_args_rejects_negative_lock_timeout() -> None:
    from sarpyx.snapflow import config

    args = create_parser().parse_args(["--input", "/tmp/product.SAFE", "--lock-timeout", "-1"])

    with pytest.raises(ValueError, match="lock-timeout"):
        config.validate_runtime_args(args)


def test_main_dispatches_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from sarpyx.cli import worldsar as worldsar_module

    calls = []

    monkeypatch.setattr(worldsar_module, 'run', lambda args: calls.append(args) or 7)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'sarpyx',
            '--input',
            '/tmp/product.SAFE',
            '--output',
            '/tmp/output',
            '--gpt-path',
            '/tmp/gpt',
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 7
    assert len(calls) == 1
    assert calls[0].product_path == '/tmp/product.SAFE'
    assert calls[0].output_dir == '/tmp/output'
    assert calls[0].gpt_path == '/tmp/gpt'


def test_run_defaults_outputs_next_to_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from sarpyx.snapflow import config
    from sarpyx.snapflow import runner

    product = tmp_path / "product.SAFE"
    product.mkdir()
    captured: dict[str, object] = {}

    def fake_pipeline(product_path, output_dir, **kwargs):
        captured["product_path"] = Path(product_path)
        captured["output_dir"] = Path(output_dir)
        captured["cuts_outdir"] = Path(kwargs["cuts_outdir"])
        captured["grid_path"] = Path(kwargs["grid_path"])
        captured["report_outdir"] = Path(kwargs["report_outdir"])
        captured["product_name"] = kwargs["product_name"]
        captured["tile_writer"] = kwargs["tile_writer"]
        captured["pre_write_hook"] = kwargs["pre_write_hook"]
        captured["keep_intermediate"] = kwargs["keep_intermediate"]
        (Path(output_dir) / "keep.txt").write_text("unrelated", encoding="utf-8")
        out = Path(output_dir) / "tc.dim"
        out.write_text("tc", encoding="utf-8")
        return out

    monkeypatch.setattr(runner, "infer_product_mode", lambda _path: "S1TOPS")
    monkeypatch.setattr(runner, "resolve_product_wkt", lambda *_args: "POLYGON EMPTY")
    monkeypatch.setitem(runner.ROUTER, "S1TOPS", fake_pipeline)
    monkeypatch.setattr(config, "GRID_PATH", None)
    monkeypatch.setattr(config, "CUTS_OUTDIR", None)
    monkeypatch.setattr(config, "DB_DIR", None)
    monkeypatch.setattr(config, "db_indexing", True)
    monkeypatch.setattr(config, "ensure_grid_file", lambda grid_path, _base_path: grid_path)

    args = create_parser().parse_args(["--input", str(product)])

    assert runner.run(args) == 0
    assert captured["output_dir"] == product.parent / "output"
    assert captured["cuts_outdir"] == product.parent / "output" / "tiles"
    assert captured["report_outdir"] == product.parent / "output" / "pdfs"
    assert captured["product_name"] == "product.SAFE"
    assert captured["tile_writer"] == "zarr"
    assert callable(captured["pre_write_hook"])
    assert captured["pre_write_hook"].subap_features.enabled is True
    assert captured["pre_write_hook"].subap_features.window_size == 5
    assert captured["keep_intermediate"] is False
    assert (product.parent / "output" / "db").is_dir()
    assert (product.parent / "output" / "pdfs").is_dir()
    assert not (product.parent / "output" / "tc.dim").exists()
    assert (product.parent / "output" / "keep.txt").exists()
    assert config.DB_DIR == str(product.parent / "output" / "db")


def test_run_fails_fast_when_product_lock_is_held(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fcntl

    from sarpyx.snapflow import config
    from sarpyx.snapflow import runner
    from sarpyx.snapflow.locks import worldsar_product_lock_path

    product = tmp_path / "product.SAFE"
    product.mkdir()
    output_dir = tmp_path / "output"
    lock_path = worldsar_product_lock_path(product, output_dir)
    lock_path.parent.mkdir(parents=True)
    handle = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(config, "CUTS_OUTDIR", None)
    monkeypatch.setattr(config, "DB_DIR", None)
    monkeypatch.setattr(config, "db_indexing", True)
    args = create_parser().parse_args(["--input", str(product), "--output", str(output_dir)])

    try:
        with pytest.raises(RuntimeError, match="WorldSAR product lock is held"):
            runner.run(args)
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
