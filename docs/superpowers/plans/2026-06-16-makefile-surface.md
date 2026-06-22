# Makefile Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broad Makefile with a minimal user workflow for conda install, product download, single-product pipeline runs, pair pipeline runs, and PBS command submission.

**Architecture:** The Makefile becomes a thin command router. Bash scripts under `scripts/` own argument validation and command construction, while `docs/makefile.md` documents user-facing variables and examples.

**Tech Stack:** GNU Make, Bash, conda, phidown, `sarpyx worldsar`, `sarpyx pipeline`, PBS `qsub`.

---

### Task 1: Lock The Makefile Contract

**Files:**
- Create: `tests/test_makefile_surface.py`
- Modify: `tests/test_pbs_caller.py`

- [ ] Add tests that `make help` exposes only `install`, `download`, `pipeline`, `pipeline-pair`, and `pbs`.
- [ ] Add a dry-run test that `make pbs CMD='sarpyx pipeline --list' PBS_ARGS=--dry-run` emits a PBS script containing the current CLI command.
- [ ] Update the existing PBS script path to `scripts/pbs_caller.sh` and remove old `uv run sarpyx` expectations.

### Task 2: Replace Makefile

**Files:**
- Modify: `Makefile`

- [ ] Keep only shared defaults, the five user targets, and `help`.
- [ ] Delegate installation, download, and pipeline command construction to scripts.
- [ ] Make output defaults deterministic under `outputs/make`.

### Task 3: Add User Workflow Scripts

**Files:**
- Create: `scripts/install_conda.sh`
- Create: `scripts/download.sh`
- Create: `scripts/run_pipeline.sh`

- [ ] `install_conda.sh` creates a conda environment when missing and installs the checkout editable with the `copernicus` extra.
- [ ] `download.sh` downloads a named product with `phidown`.
- [ ] `run_pipeline.sh` runs either single-product or pair pipelines with current `sarpyx pipeline` syntax.

### Task 4: Document Usage

**Files:**
- Create: `docs/makefile.md`

- [ ] Document each Makefile target, required variables, defaults, and examples.
- [ ] Include PBS examples for `small`, `medium`, and `large`.

### Task 5: Validate

**Files:**
- No new files.

- [ ] Run focused tests for Makefile and PBS behavior.
- [ ] Run `make help`.
- [ ] Report skipped heavyweight processing checks.
