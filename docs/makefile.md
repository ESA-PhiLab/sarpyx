# Makefile Usage

The Makefile is a thin user workflow wrapper. It keeps installation, download,
pipeline execution, and PBS submission in one place while implementation details
live in `scripts/`.

Run `make help` to list the available targets.

## Defaults

| Variable | Default | Used by |
| --- | --- | --- |
| `ENV_PREFIX` | `<repo>/.conda/sarpyx` | `install`, `pipeline`, `pipeline-pair` |
| `ENV_FILE` | `<repo>/environment.yml` | `install` |
| `PYTHON` | `$(ENV_PREFIX)/bin/python` | `pipeline`, `pipeline-pair` |
| `GPT_PATH` | `$(ENV_PREFIX)/opt/esa-snap/bin/gpt` | `pipeline`, `pipeline-pair` |
| `PHIDOWN` | `$(ENV_PREFIX)/bin/phidown` | `download` |
| `OUTPUT_ROOT` | `outputs/make` | `pipeline`, `pipeline-pair` |
| `PIPELINE` | `s1_tops` | `pipeline`, `pipeline-pair` |
| `OUTPUT` | `$(OUTPUT_ROOT)/$(PIPELINE)` | `pipeline`, `pipeline-pair` |
| `CUTS_OUTDIR` | `$(OUTPUT)/tiles` | `pipeline`, `pipeline-pair` |
| `DOWNLOAD_DIR` | `input_data` | `download` |
| `DOWNLOAD_MODE` | `safe` | `download` |
| `PBS_SIZE` | `large` | `pbs` |
| `PBS_NAME` | `sarpyx` | `pbs` |
| `PBS_QUEUE` | `cpu_std` | `pbs` |

## Install

Create or update the local conda environment at `.conda/sarpyx` from
`environment.yml`. This installs SNAP (`snap13`), this checkout in editable
mode, and `phidown` through the `sarpyx[copernicus]` extra.
The install script verifies exact local executables:

- `.conda/sarpyx/opt/esa-snap/bin/gpt`
- `.conda/sarpyx/bin/sarpyx`
- `.conda/sarpyx/bin/phidown`

```bash
make install
```

Override the local environment prefix when needed:

```bash
make install ENV_PREFIX=$PWD/.conda/sarpyx-snap
conda activate $PWD/.conda/sarpyx-snap
```

Use a different environment file when needed:

```bash
make install ENV_FILE=/path/to/environment.yml
```

## Download

Download a product with `phidown`:

```bash
make download PRODUCT_NAME=S1A_IW_SLC__1SDV_20240807T154929_20240807T154956_055109_06B710_F274.SAFE
```

The helper uses `.conda/sarpyx/bin/phidown` by default. It does not resolve
`phidown` from `PATH`.

Use a different output folder or config file:

```bash
make download \
  PRODUCT_NAME=S1A_IW_SLC__1SDV_20240807T154929_20240807T154956_055109_06B710_F274.SAFE \
  DOWNLOAD_DIR=input_data \
  CONFIG_FILE=/path/to/.s5cfg
```

## Single-Product Pipeline

Run a built-in or external single-product recipe:

```bash
make pipeline \
  PIPELINE=s1_tops \
  INPUT=input_data/product.SAFE \
  GRID_PATH=grid/grid_10km.geojson \
  GPT_PATH=/path/to/gpt
```

The default output is `outputs/make/<pipeline>`, and tiles default to
`outputs/make/<pipeline>/tiles`.

The helper runs the repository dispatcher as `python -m sarpyx.cli.main` so it
does not depend on an older `sarpyx` executable that may already be on `PATH`.
By default it uses `.conda/sarpyx/bin/python` and resolves SNAP GPT at
`.conda/sarpyx/opt/esa-snap/bin/gpt`.
Ambient shell variables named `PYTHON`, `GPT_PATH`, or `PHIDOWN` are ignored by
the Makefile so unrelated tools cannot be picked up accidentally. To override a
path, pass it on the make command line:

```bash
make pipeline \
  PYTHON=/path/to/env/bin/python \
  GPT_PATH=/path/to/env/opt/esa-snap/bin/gpt \
  PIPELINE=s1_tops \
  INPUT=input_data/product.SAFE
```

Pass recipe parameters as space-separated `NAME=VALUE` pairs:

```bash
make pipeline \
  PIPELINE=s1_tops \
  INPUT=input_data/product.SAFE \
  PARAMS='first_burst=1 last_burst=1'
```

## Pair Pipeline

Run a double-product recipe such as `s1_insar`:

```bash
make pipeline-pair \
  PIPELINE=s1_insar \
  MASTER=input_data/master.SAFE \
  SLAVE=input_data/slave.SAFE \
  GRID_PATH=grid/grid_10km.geojson \
  GPT_PATH=/path/to/gpt
```

## PBS

Submit any command to a CPU node through `scripts/pbs_caller.sh`:

```bash
make pbs PBS_SIZE=small CMD='python -m sarpyx.cli.main pipeline --list'
make pbs PBS_SIZE=medium CMD='python -m sarpyx.cli.main pipeline s1_tops --input input_data/product.SAFE --output outputs/make/s1_tops'
make pbs PBS_SIZE=large CMD='python -m sarpyx.cli.main pipeline s1_insar --master input_data/master.SAFE --slave input_data/slave.SAFE --output outputs/make/s1_insar'
```

Dry-run the generated PBS script before submitting:

```bash
make pbs PBS_SIZE=small PBS_ARGS=--dry-run CMD='python -m sarpyx.cli.main pipeline --list'
```

Profiles are defined in `scripts/pbs_caller.sh`:

| Size | Walltime | CPUs | Memory |
| --- | --- | --- | --- |
| `small` | `02:00:00` | 32 | `32g` |
| `medium` | `04:00:00` | 96 | `64g` |
| `large` | `06:00:00` | 192 | `128g` |
