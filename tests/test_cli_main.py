import sys
import types

import pytest

from sarpyx.cli.main import create_main_parser


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


def test_create_main_parser_formats_help_with_worldsar_subcommand() -> None:
    parser = create_main_parser()
    help_text = parser.format_help()

    assert 'worldsar' in help_text
    assert '--version' in help_text


def test_create_main_parser_includes_worldsar_subcommand() -> None:
    parser = create_main_parser()

    args = parser.parse_args(
        [
            'worldsar',
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
        ]
    )

    assert args.command == 'worldsar'
    assert args.product_path == '/tmp/product.SAFE'
    assert args.output_dir == '/tmp/output'
    assert args.cuts_outdir == '/tmp/cuts'
    assert args.db_dir == '/tmp/db'
    assert args.h5_to_zarr_only is True
    assert args.zarr_chunk_size == [64, 64]
    assert args.orbit_continue_on_fail is True


def test_worldsar_parser_defaults_match_direct_entrypoint() -> None:
    from sarpyx.cli.worldsar import create_parser as create_worldsar_parser

    top_level_args = create_main_parser().parse_args(['worldsar', '--input', '/tmp/product.SAFE'])
    direct_args = create_worldsar_parser().parse_args(['--input', '/tmp/product.SAFE'])

    for field in WORLDSAR_DEFAULT_FIELDS:
        top_level_value = getattr(top_level_args, field)
        direct_value = getattr(direct_args, field)
        assert direct_value == top_level_value
        assert '/shared/home/vmarsocci' not in str(direct_value)


def test_worldsar_dispatch_uses_top_level_parser_args(monkeypatch: pytest.MonkeyPatch) -> None:
    from sarpyx.cli import main as cli_main

    calls = []
    fake_worldsar = types.ModuleType('sarpyx.cli.worldsar')
    fake_worldsar.run = lambda args: calls.append(args) or 7

    monkeypatch.setitem(sys.modules, 'sarpyx.cli.worldsar', fake_worldsar)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'sarpyx',
            'worldsar',
            '--input',
            '/tmp/product.SAFE',
            '--output',
            '/tmp/output',
            '--gpt-path',
            '/tmp/gpt',
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_main.main()

    assert excinfo.value.code == 7
    assert len(calls) == 1
    assert calls[0].product_path == '/tmp/product.SAFE'
    assert calls[0].output_dir == '/tmp/output'
    assert calls[0].gpt_path == '/tmp/gpt'
