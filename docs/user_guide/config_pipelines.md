# Config-Driven SNAP Pipelines

sarpyx provides an alternate YAML-driven CLI for composing SNAP processing
pipelines without changing the existing WorldSAR command:

```bash
uv run sarpyx-pipeline validate pipeline.yaml
uv run sarpyx-pipeline list pipeline.yaml
uv run sarpyx-pipeline run pipeline.yaml --pipeline preprocess --set-input product=/data/input.SAFE --outdir data/output
```

The config format is versioned.  Version `1` supports named pipelines, nested
mini-pipelines, raw SNAP operator steps, and existing `GPT` method steps.

```yaml
version: 1

defaults:
  format: BEAM-DIMAP
  gpt_path: /usr/local/snap/bin/gpt
  memory: 24G
  parallelism: 8
  timeout: 7200
  resume: false
  overwrite: false
  keep_graphs: true

pipelines:
  preprocess:
    inputs:
      product: null
    steps:
      - id: calibration
        op: Calibration
        source: product
        params:
          outputSigmaBand: true

      - id: terrain_correction
        op: Terrain-Correction
        params:
          demName: SRTM 3Sec
          pixelSpacingInMeter: 10.0
```

## Step Types

- `op`: raw SNAP operator name. sarpyx renders a temporary XML graph with
  `Read -> Operator -> Write` and passes `params` as SNAP XML parameters.
- `method`: existing `sarpyx.snapflow.engine.GPT` method name. Use Python
  method names such as `terrain_correction`, `interferogram`, or
  `topsar_coregistration`.
- `use`: call another named pipeline from the same YAML file.

Each step must set exactly one of `op`, `method`, or `use`.

## Pair Inputs

Pair workflows use named inputs and explicit sources.  This keeps master/slave
wiring readable and avoids positional ambiguity.

```yaml
version: 1

pipelines:
  preprocess:
    inputs:
      product: null
    steps:
      - id: calibration
        op: Calibration
        source: product

  insar_pair:
    inputs:
      master: null
      slave: null
    steps:
      - id: master_pre
        use: preprocess
        inputs:
          product: master

      - id: slave_pre
        use: preprocess
        inputs:
          product: slave

      - id: coreg
        method: topsar_coregistration
        sources:
          master_product: master_pre
          slave_product: slave_pre
        params:
          use_esd: true

      - id: interferogram
        method: interferogram
        source: coreg
        params:
          include_coherence: true
```

Run it with:

```bash
uv run sarpyx-pipeline run insar.yaml \
  --pipeline insar_pair \
  --set-input master=/data/master.SAFE \
  --set-input slave=/data/slave.SAFE \
  --outdir data/output/insar
```

## Output Safety

By default, an existing step output is an error.  Use `--resume` to reuse an
existing output only when its manifest fingerprint matches the step config, or
`--overwrite` to replace the expected output and associated BEAM-DIMAP `.data`
directory.

Use `--dry-run --json` to inspect planned steps without creating outputs.
