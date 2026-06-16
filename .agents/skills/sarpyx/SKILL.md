---
name: sarpyx
description: Use when asked to run, automate, preflight, validate, or summarize sarpyx SAR processing jobs, including WorldSAR, pipeline recipes, any SNAP GPT operator declared in sarpyx/snapflow/op.py, operator discovery/help, tiling, H5-to-Zarr conversion, mission recipes, validation targets, Docker/SIF workflows, or HPC upload operations.
---

# Sarpyx Processing

Use this skill to autonomously run sarpyx processing after enough concrete inputs are known.
The authoritative CLI surface is:

- `uv run sarpyx worldsar ...`
- `uv run sarpyx pipeline ...`

Treat older examples such as `uv run sarpyx --input ...` as stale until `uv run sarpyx --help` proves otherwise.
For arbitrary SNAP GPT operations, use `scripts/snap_operator.py`; it parses `sarpyx/snapflow/op.py` and validates the requested operator against the live `snap_operators` registry.

## Preflight First

Before heavy processing or operational Makefile targets, run:

```bash
python .agents/skills/sarpyx/scripts/preflight.py --repo /Users/roberto.delprete/Downloads/sarpyx
```

Add known inputs, for example:

```bash
python .agents/skills/sarpyx/scripts/preflight.py \
  --repo /Users/roberto.delprete/Downloads/sarpyx \
  --mode worldsar \
  --input /data/product.SAFE \
  --output /data/out \
  --grid-path /data/grid.geojson \
  --gpt-path /opt/esa-snap/bin/gpt
```

Stop and ask for missing values when preflight reports blockers for input products, output root, grid, SNAP GPT, master/slave products, or required WKT.

## Route Requests

- Use `sarpyx worldsar` for the normal single-product mission-aware workflow.
- Use `sarpyx pipeline` when the user names a recipe, wants product-type inference avoided, or needs master/slave inputs.
- Built-in pipeline recipes are `s1_tops`, `s1_strip`, `tsx`, `csg`, `biomass`, `nisar`, and `s1_insar`.
- `s1_insar` is double-product and needs `--master` and `--slave`; the other built-ins are single-product and need `--input`.
- Use `--param NAME=VALUE` for recipe options; values are parsed as JSON when possible.
- Use `--h5-to-zarr-only` only with `sarpyx worldsar`.
- Use `scripts/snap_operator.py` when the user asks for any SNAP GPT operation/operator from `sarpyx/snapflow/op.py`, including operators not wrapped by `sarpyx pipeline` or `worldsar`.

## Full GPT Operator Coverage

- Do not maintain a copied operator list in this skill. `snap_operator.py --list` is the coverage source because it reads `snap_operators` from `sarpyx/snapflow/op.py`.
- Use exact operator names from `--list`; quote names containing spaces, for example `"Multi-size Mosaic"`.
- Use `--dry-run` to validate registry coverage and command shape without SNAP installed. A dry run can prove the operator is declared and the command can be built; it cannot prove SNAP runtime parameters are sufficient.
- Use `--help-operator --gpt-path /path/to/gpt` before running unfamiliar operators so SNAP reports required `-S` sources and `-P` parameters.
- Use `--source NAME=PATH` for named source inputs, `--input PATH` for the default `source`, repeat `--param NAME=VALUE`, and append unusual GPT options with `--raw-arg`.

## Autonomy Rules

- You may create output directories and run processing after preflight is clean.
- You may run `make validate-*`, Docker, SIF, and HPC upload targets only when the user explicitly asks for that operational class.
- Do not delete existing products, prune Docker, publish packages, push images, or push SIFs unless the user explicitly asks for that exact action.
- Prefer explicit `--output`, `--cuts-outdir`, `--grid-path`, `--gpt-path`, and `--snap-userdir` values for reproducibility.
- For SNAP-heavy jobs, start with conservative `--gpt-memory`, `--gpt-cache-size`, `--gpt-parallelism`, and `--gpt-timeout` values supplied by the user or project defaults.
- For generic SNAP operators, first list or inspect the operator unless the user already provided exact `-S` sources and `-P` parameters.

## Command Shapes

WorldSAR:

```bash
uv run sarpyx worldsar \
  --input /data/product.SAFE \
  --output /data/out \
  --cuts-outdir /data/out/tiles \
  --grid-path /data/grid.geojson \
  --gpt-path /opt/esa-snap/bin/gpt
```

Single-product recipe:

```bash
uv run sarpyx pipeline s1_tops \
  --input /data/product.SAFE \
  --output /data/out \
  --grid-path /data/grid.geojson \
  --gpt-path /opt/esa-snap/bin/gpt
```

Double-product recipe:

```bash
uv run sarpyx pipeline s1_insar \
  --master /data/master.SAFE \
  --slave /data/slave.SAFE \
  --output /data/out \
  --gpt-path /opt/esa-snap/bin/gpt
```

Generic SNAP operator from `sarpyx/snapflow/op.py`:

```bash
python .agents/skills/sarpyx/scripts/snap_operator.py \
  --repo /Users/roberto.delprete/Downloads/sarpyx \
  --operator Calibration \
  --input /data/product.SAFE \
  --target /data/out/product_cal.dim \
  --format BEAM-DIMAP \
  --gpt-path /opt/esa-snap/bin/gpt \
  --param selectedPolarisations=VV
```

For operator discovery or parameters:

```bash
python .agents/skills/sarpyx/scripts/snap_operator.py --repo /Users/roberto.delprete/Downloads/sarpyx --list
python .agents/skills/sarpyx/scripts/snap_operator.py --repo /Users/roberto.delprete/Downloads/sarpyx --operator Terrain-Correction --help-operator --gpt-path /opt/esa-snap/bin/gpt
python .agents/skills/sarpyx/scripts/snap_operator.py --repo /Users/roberto.delprete/Downloads/sarpyx --operator "Multi-size Mosaic" --dry-run
```

Use `--source NAME=PATH` for non-default source parameters, repeat `--param NAME=VALUE` for `-P` parameters, and use `--dry-run` before destructive or expensive runs.

## After Execution

Summarize the actual artifacts, not just command success:

- Final/intermediate `.dim` products.
- Tile files such as `.h5`, `.zarr`, `.tif`, `.npz`, `.npy`, or `.pkl`.
- Cut reports, validation PDFs, manifests, JSON summaries, and logs.
- Any failed checks, missing artifacts, skipped validation, or residual runtime risk.

Use narrow searches rooted at the declared output and cuts directories, for example:

```bash
rg --files /data/out /data/out/tiles | rg '\.(dim|h5|zarr|tif|tiff|npz|npy|pkl|txt|pdf|json)$'
```
