# Evidence

- `make pipeline` used repo Python after the prior fix but still failed because
  SNAP GPT was not resolved from the intended conda environment.
- Make imports ambient shell variables by default, so global `PYTHON` or
  `GPT_PATH` values could still bypass local tooling.
- The expected SNAP GPT path for the conda SNAP package is
  `<env-prefix>/opt/esa-snap/bin/gpt`.
