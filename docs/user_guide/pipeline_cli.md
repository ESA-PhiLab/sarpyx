# Generic Pipeline CLI

For end-to-end command examples, start with [CLI Usage Examples](cli_examples.md).

`sarpyx pipeline` runs an explicit pipeline recipe. Use it when product-type
inference is not desired, or when a recipe needs named inputs such as a
master/slave pair.

Use `sarpyx worldsar` for the opinionated single-product WorldSAR workflow.
Use `sarpyx pipeline` when you want to choose the recipe yourself.

## Built-In Pipelines

List the built-in recipes:

```bash
sarpyx pipeline --list
```

Run a single-product recipe by name:

```bash
sarpyx pipeline s1_tops \
  --input /data/product.SAFE \
  --output /data/out \
  --gpt-path /opt/miniconda3/envs/sarpyx/opt/esa-snap/bin/gpt \
  --param sentinel_swath=IW2
```

Run the double-product Sentinel-1 InSAR recipe:

```bash
sarpyx pipeline s1_insar \
  --master /data/master.SAFE \
  --slave /data/slave.SAFE \
  --output /data/insar-out \
  --gpt-path /opt/miniconda3/envs/sarpyx/opt/esa-snap/bin/gpt \
  --param subswath=IW2 \
  --param selected_polarisations='["VV"]' \
  --param use_esd=false
```

`--param NAME=VALUE` values are parsed as JSON when possible. Unquoted values
such as `IW2` remain strings; `false`, `2`, and `["VV"]` become a boolean,
integer, and list.

## External Pipeline Files

An external pipeline file must declare its input kind and a `steps` function:

```python
from sarpyx.snapflow.runtime import PipelineStep

INPUT_KIND = "double"

def steps(subswath=None):
    return [
        PipelineStep("TopsarSplit", {"source_ref": "master", "outdir": "master", "subswath": subswath}, "master_split"),
    ]
```

Then run it by path:

```bash
sarpyx pipeline /path/to/my_pipeline.py \
  --master /data/master.SAFE \
  --slave /data/slave.SAFE \
  --output /data/out \
  --param subswath=IW2
```

Supported input kinds are:

- `single`: requires `--input`.
- `double`: requires `--master` and `--slave`.

The generic command uses the same GPT performance defaults as `worldsar`:

```text
--gpt-parallelism 6
--gpt-memory 16G
--gpt-cache-size 8G
```
