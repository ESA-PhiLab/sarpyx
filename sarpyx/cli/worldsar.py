"""WorldSAR command-line entrypoint and compatibility facade."""

from __future__ import annotations

import sys

from sarpyx.snapflow.compat import (
    _apply_sentinel_orbit_file,
    _sentinel_post_chain,
    merge_iq_into_pdec,
    pipeline_biomass,
    pipeline_nisar,
    pipeline_sentinel,
    pipeline_tsx_csg,
)
from sarpyx.snapflow.gpt import create_gpt_operator as _create_gpt_operator
from sarpyx.snapflow.parser import add_worldsar_arguments, create_parser
from sarpyx.snapflow.product import (
    _resolve_terrasar_product_xml,
    _terrasar_product_variant,
    infer_product_mode,
    resolve_product_wkt as _resolve_product_wkt,
)
from sarpyx.snapflow.runner import run
from sarpyx.snapflow.tiling import (
    _resolve_tiling_wkt,
    _run_tops_swath_tiling,
)


def main(argv=None):
    args = create_parser().parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
