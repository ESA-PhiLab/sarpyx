# Evidence

- `environment.yml` existed but installed only `-e .`, so phidown was implicit
  only if added separately.
- `scripts/install_conda.sh` created the environment inline instead of using the
  environment file.
- `snap13=13.0.0` is the repo's current conda SNAP dependency.
