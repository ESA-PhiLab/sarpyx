from sarpyx.cli.main import create_main_parser


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
            '--tops-swaths',
            'IW1',
            '--polarizations',
            'VV',
            '--skip-subaperture',
            '--skip-polarimetric-decomposition',
            '--single-band',
            '--orbit-continue-on-fail',
        ]
    )

    assert args.command == 'worldsar'
    assert args.product_path == '/tmp/product.SAFE'
    assert args.output_dir == '/tmp/output'
    assert args.cuts_outdir == '/tmp/cuts'
    assert args.db_dir == '/tmp/db'
    assert args.tops_swaths == ['IW1']
    assert args.polarizations == ['VV']
    assert args.skip_subaperture is True
    assert args.skip_polarimetric_decomposition is True
    assert args.single_band is True
    assert args.orbit_continue_on_fail is True
