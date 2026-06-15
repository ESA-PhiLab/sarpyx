# CLI Usage Examples

The installed CLI surface is:

- `sarpyx worldsar`
- `sarpyx pipeline`
- `sarpyx-worldsar`
- `sarpyx-pipeline`

Use `uv run` before the command when working from the repository without activating the environment.

## Discover Commands

```bash
uv run sarpyx --help
uv run sarpyx worldsar --help
uv run sarpyx pipeline --help
uv run sarpyx pipeline --list
```

Built-in pipeline recipes:

| Recipe | Inputs |
| --- | --- |
| `s1_tops` | `--input` |
| `s1_strip` | `--input` |
| `tsx` | `--input` |
| `csg` | `--input` |
| `biomass` | `--input` |
| `nisar` | `--input` |
| `s1_insar` | `--master` and `--slave` |

## WorldSAR Single Product

Run the mission-aware WorldSAR workflow when you want product inference, SNAP preprocessing, tiling, and sarpyx outputs in one command.

```bash
uv run sarpyx worldsar \
  --input /data/S1A_IW_SLC_1SDV_PRODUCT.SAFE \
  --output /data/out/worldsar \
  --cuts-outdir /data/out/worldsar/tiles \
  --grid-path /data/grid/grid_10km.geojson \
  --gpt-path /opt/esa-snap/bin/gpt \
  --snap-userdir /data/out/.snap \
  --gpt-memory 16G \
  --gpt-cache-size 8G \
  --gpt-parallelism 6
```

Use the installed compatibility shim the same way:

```bash
sarpyx-worldsar \
  --input /data/product.SAFE \
  --output /data/out/worldsar \
  --grid-path /data/grid/grid_10km.geojson \
  --gpt-path "$GPT_PATH"
```

## Sentinel-1 TOPS Options

Limit preprocessing to one swath or burst range when you need a smaller run.

```bash
uv run sarpyx worldsar \
  --input /data/S1A_IW_SLC_1SDV_PRODUCT.SAFE \
  --output /data/out/iw2 \
  --grid-path /data/grid/grid_10km.geojson \
  --gpt-path "$GPT_PATH" \
  --sentinel-swath IW2 \
  --sentinel-first-burst 3 \
  --sentinel-last-burst 8 \
  --sentinel-subap-decompositions 2 4
```

## Reuse Existing Intermediates

When BEAM-DIMAP intermediates already exist, skip preprocessing and run tiling from them.

```bash
uv run sarpyx worldsar \
  --input /data/product.SAFE \
  --output /data/out/reuse \
  --grid-path /data/grid/grid_10km.geojson \
  --gpt-path "$GPT_PATH" \
  --skip-preprocessing
```

## Convert H5 Tiles to Zarr

Use `--h5-to-zarr-only` for an existing H5 tile.

```bash
uv run sarpyx worldsar \
  --input /data/tiles/tile_001.h5 \
  --output /data/tiles/tile_001.zarr \
  --h5-to-zarr-only \
  --overwrite-zarr \
  --zarr-chunk-size 256 256
```

## Explicit Single-Product Pipeline

Use `sarpyx pipeline` when you want to choose the recipe yourself instead of relying on WorldSAR product inference.

```bash
uv run sarpyx pipeline s1_tops \
  --input /data/product.SAFE \
  --output /data/out/s1_tops \
  --grid-path /data/grid/grid_10km.geojson \
  --cuts-outdir /data/out/s1_tops/tiles \
  --gpt-path "$GPT_PATH" \
  --param sentinel_swath=IW2 \
  --param selected_polarisations='["VV"]'
```

The standalone entry point is equivalent:

```bash
sarpyx-pipeline s1_tops \
  --input /data/product.SAFE \
  --output /data/out/s1_tops \
  --gpt-path "$GPT_PATH"
```

## Sentinel-1 InSAR Pipeline

`s1_insar` is the built-in double-product recipe.

```bash
uv run sarpyx pipeline s1_insar \
  --master /data/master.SAFE \
  --slave /data/slave.SAFE \
  --output /data/out/insar \
  --grid-path /data/grid/grid_10km.geojson \
  --cuts-outdir /data/out/insar/tiles \
  --gpt-path "$GPT_PATH" \
  --param subswath=IW2 \
  --param selected_polarisations='["VV"]' \
  --param use_esd=false
```

`--param NAME=VALUE` parses JSON values when possible. For example, `false` becomes a boolean, `2` becomes an integer, and `'["VV"]'` becomes a list.

## External Pipeline File

An external pipeline file must declare `INPUT_KIND` and a `steps` function.

```python
from sarpyx.snapflow.runtime import PipelineStep

INPUT_KIND = "single"

def steps(subswath=None):
    return [
        PipelineStep(
            "TopsarSplit",
            {"source_ref": "input", "outdir": "split", "subswath": subswath},
            "split",
        )
    ]
```

Run it by path:

```bash
uv run sarpyx pipeline /path/to/my_pipeline.py \
  --input /data/product.SAFE \
  --output /data/out/custom \
  --gpt-path "$GPT_PATH" \
  --param subswath=IW2
```

## Output Checks

After a run, inspect only the declared output roots:

```bash
rg --files /data/out/worldsar /data/out/worldsar/tiles \
  | rg '\.(dim|h5|zarr|tif|tiff|npz|npy|pkl|txt|pdf|json)$'
```

For setup details, see [Installation](installation.md). For more pipeline-specific detail, see [Generic Pipeline CLI](pipeline_cli.md).
