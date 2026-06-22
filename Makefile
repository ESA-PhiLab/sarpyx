# sarpyx Makefile

SHELL := /bin/bash
.DEFAULT_GOAL := help

# User defaults. Override these on the make command line.
ENV_PREFIX ?= $(CURDIR)/.conda/sarpyx
ENV_FILE ?= $(CURDIR)/environment.yml
PYTHON ?= $(ENV_PREFIX)/bin/python
GPT_PATH ?= $(ENV_PREFIX)/opt/esa-snap/bin/gpt
PHIDOWN ?= $(ENV_PREFIX)/bin/phidown
ifeq ($(origin PYTHON),environment)
PYTHON := $(ENV_PREFIX)/bin/python
endif
ifeq ($(origin GPT_PATH),environment)
GPT_PATH := $(ENV_PREFIX)/opt/esa-snap/bin/gpt
endif
ifeq ($(origin PHIDOWN),environment)
PHIDOWN := $(ENV_PREFIX)/bin/phidown
endif
OUTPUT_ROOT ?= outputs/make
PIPELINE ?= s1_tops
OUTPUT ?= $(OUTPUT_ROOT)/$(PIPELINE)
CUTS_OUTDIR ?= $(OUTPUT)/tiles
DOWNLOAD_DIR ?= input_data
DOWNLOAD_MODE ?= safe
PBS_SIZE ?= large
PBS_NAME ?= sarpyx
PBS_QUEUE ?= cpu_std
PBS_ARGS ?=
EXTRA_ARGS ?=
PARAMS ?=

.PHONY: help install download pipeline pipeline-pair pbs

help:
	@awk 'BEGIN {FS = ":.*##"; print "Targets:"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Create/update conda env with SNAP, sarpyx, and phidown
	@ENV_PREFIX="$(ENV_PREFIX)" ENV_FILE="$(ENV_FILE)" bash scripts/install_conda.sh

download: ## Download PRODUCT_NAME with phidown into DOWNLOAD_DIR
	@PRODUCT_NAME="$(PRODUCT_NAME)" DOWNLOAD_DIR="$(DOWNLOAD_DIR)" DOWNLOAD_MODE="$(DOWNLOAD_MODE)" CONFIG_FILE="$(CONFIG_FILE)" ENV_PREFIX="$(ENV_PREFIX)" PHIDOWN="$(PHIDOWN)" bash scripts/download.sh

pipeline: ## Run a single-product pipeline: PIPELINE=name INPUT=path
	@PIPELINE="$(PIPELINE)" \
	INPUT="$(INPUT)" \
	OUTPUT="$(OUTPUT)" \
	CUTS_OUTDIR="$(CUTS_OUTDIR)" \
	GRID_PATH="$(GRID_PATH)" \
	GPT_PATH="$(GPT_PATH)" \
	PYTHON="$(PYTHON)" \
	ENV_PREFIX="$(ENV_PREFIX)" \
	PARAMS="$(PARAMS)" \
	EXTRA_ARGS="$(EXTRA_ARGS)" \
	bash scripts/run_pipeline.sh single

pipeline-pair: ## Run a double-product pipeline: PIPELINE=name MASTER=path SLAVE=path
	@PIPELINE="$(PIPELINE)" \
	MASTER="$(MASTER)" \
	SLAVE="$(SLAVE)" \
	OUTPUT="$(OUTPUT)" \
	CUTS_OUTDIR="$(CUTS_OUTDIR)" \
	GRID_PATH="$(GRID_PATH)" \
	GPT_PATH="$(GPT_PATH)" \
	PYTHON="$(PYTHON)" \
	ENV_PREFIX="$(ENV_PREFIX)" \
	PARAMS="$(PARAMS)" \
	EXTRA_ARGS="$(EXTRA_ARGS)" \
	bash scripts/run_pipeline.sh pair

pbs: ## Submit CMD through PBS: CMD='python -m sarpyx.cli.main pipeline --list'
	@test -n "$(CMD)" || { echo "ERROR: CMD is required, for example CMD='python -m sarpyx.cli.main pipeline --list'."; exit 2; }
	@bash scripts/pbs_caller.sh \
		--size "$(PBS_SIZE)" \
		--name "$(PBS_NAME)" \
		--queue "$(PBS_QUEUE)" \
		$(PBS_ARGS) \
		-- bash -lc "$(CMD)"
