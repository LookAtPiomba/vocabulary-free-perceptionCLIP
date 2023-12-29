#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"

augmentation=$1
template0=$2
template1=$3
infer_mode=$4
save_name=$5

python src/CaSED_zero_shot_inference/eval_acc_self_infer.py --eval_augmentation="$augmentation" \
  --template0="$template0" \
  --template1="$template1" \
  --save_name="$save_name" \
  --infer_mode="$infer_mode" \
  --model=ViT-B/16 \