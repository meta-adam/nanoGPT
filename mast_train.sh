#!/usr/bin/env bash

JOB_ONCALL="devai"
JOB_DATA_PROJECT="devai"

torchx run \
    --scheduler=mast_conda \
    --scheduler_args="hpcIdentity=${JOB_DATA_PROJECT},hpcJobOncall=${JOB_ONCALL},workspace_fbpkg_name=metaconda_demo,fbpkg_ids=conda_mast_core:stable,forceSingleRegion=False"  \
    fb.conda.torchrun \
    --h t1 --run_as_root True \
    -- \
    --no-python --nnodes=2 --nproc-per-node=1 \
    /packages/conda_mast_core/run/torchx_run.sh train.py config/train_shakespeare_char_cpu.py \
    --gradient_accumulation_steps=2 --out_dir=/tmp
