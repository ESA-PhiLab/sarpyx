import os
import sys
from pathlib import Path

import pytest

from sarpyx.cli import worldsar
from sarpyx.cli.worldsar import _resolve_snap_userdir, _validate_runtime_args, create_parser, main


WORLDSAR_DEFAULT_FIELDS = (
    'output_dir',
    'cuts_outdir',
    'gpt_path',
    'grid_path',
    'db_dir',
    'gpt_memory',
    'gpt_parallelism',
    'gpt_timeout',
    'snap_userdir',
    'orbit_type',
)


def test_create_parser_formats_help() -> None:
    parser = create_parser()
    help_text = parser.format_help()

    assert '--input' in help_text
    assert '--orbit-continue-on-fail' in help_text
    assert '--sentinel-subaps' in help_text


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
            '--orbit-continue-on-fail',
            '--sentinel-subaps',
            '4',
        ]
    )

    assert args.product_path == '/tmp/product.SAFE'
    assert args.output_dir == '/tmp/output'
    assert args.cuts_outdir == '/tmp/cuts'
    assert args.db_dir == '/tmp/db'
    assert args.h5_to_zarr_only is True
    assert args.zarr_chunk_size == [64, 64]
    assert args.orbit_continue_on_fail is True
    assert args.sentinel_subaps == 4


def test_create_parser_defaults_are_stable() -> None:
    args = create_parser().parse_args(['--input', '/tmp/product.SAFE'])

    for field in WORLDSAR_DEFAULT_FIELDS:
        assert '/shared/home/vmarsocci' not in str(getattr(args, field))
    assert args.sentinel_subaps is None


def test_validate_runtime_args_rejects_invalid_sentinel_subaps() -> None:
    args = create_parser().parse_args(
        [
            '--input',
            '/tmp/product.SAFE',
            '--sentinel-subaps',
            '1',
        ]
    )

    with pytest.raises(ValueError, match='--sentinel-subaps must be >= 2'):
        _validate_runtime_args(args)


def test_resolve_snap_userdir_skips_unreadable_hdf_native_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_userdir = tmp_path / 'bad-snap'
    native_dir = bad_userdir / 'auxdata' / 'hdf_natives'
    native_dir.mkdir(parents=True)
    loader = native_dir / 'NativeLibraryLoader.jar'
    loader.write_text('not readable by SNAP', encoding='utf-8')
    loader.chmod(0)
    monkeypatch.setenv('TMPDIR', str(tmp_path))

    try:
        resolved = Path(_resolve_snap_userdir(bad_userdir))
    finally:
        loader.chmod(0o600)

    assert resolved != bad_userdir
    assert resolved == tmp_path / f'sarpyx-snap-userdir-{os.getuid()}'


def test_apply_runtime_overrides_resolves_unusable_snap_userdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_userdir = tmp_path / 'bad-snap'
    native_dir = bad_userdir / 'auxdata' / 'hdf_natives'
    native_dir.mkdir(parents=True)
    loader = native_dir / 'NativeLibraryLoader.jar'
    loader.write_text('locked', encoding='utf-8')
    loader.chmod(0)
    fallback = tmp_path / 'tmp'
    fallback.mkdir()
    monkeypatch.setenv('TMPDIR', str(fallback))
    old_snap_userdir = worldsar.SNAP_USERDIR
    old_env = os.environ.get('SNAP_USERDIR')

    try:
        args = create_parser().parse_args(
            [
                '--input',
                '/tmp/product.SAFE',
                '--snap-userdir',
                str(bad_userdir),
            ]
        )

        worldsar._apply_runtime_overrides(args)

        expected = fallback / f'sarpyx-snap-userdir-{os.getuid()}'
        assert Path(worldsar.SNAP_USERDIR) == expected
        assert os.environ['SNAP_USERDIR'] == str(expected)
    finally:
        worldsar.SNAP_USERDIR = old_snap_userdir
        if old_env is None:
            os.environ.pop('SNAP_USERDIR', None)
        else:
            os.environ['SNAP_USERDIR'] = old_env
        loader.chmod(0o600)


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
