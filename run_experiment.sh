#!/bin/bash
# Usage: bash run_experiment.sh <exp_name> <extra_args...>
# Runs a training experiment and captures the final loss

EXP_NAME=${1:?Usage: run_experiment.sh <exp_name> <extra_args...>}
shift
EXTRA_ARGS="$@"

OUTPUT_DIR="results/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    main_jit.py \
    --model JiT-B/16 \
    --img_size 128 --noise_scale 1.0 \
    --batch_size 64 --blr 5e-5 \
    --epochs 10 --warmup_epochs 1 \
    --class_num 10 \
    --output_dir "${OUTPUT_DIR}" \
    --data_path data/imagenette2-320 \
    --num_workers 4 \
    --save_last_freq 100 \
    --log_freq 50 \
    ${EXTRA_ARGS} 2>&1 | tee "${OUTPUT_DIR}/train.log"
