#!/bin/bash
# Offline BC training on HUGE-Bench task0. Does not need IsaacLab.

set -e
source /home/ubuntu/miniconda3/bin/activate isaac
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /home/ubuntu

python -m huge_bench.train_bc \
    --batch_size "${BATCH_SIZE:-16}" \
    --grad_accum "${GRAD_ACCUM:-4}" \
    --max_steps "${MAX_STEPS:-20000}" \
    --head_lr 3e-4 \
    --lora_lr 1e-6 \
    --val_every 500 \
    --save_every 2000 \
    --log_dir /home/ubuntu/drone_project/logs/huge_bench \
    "$@"
