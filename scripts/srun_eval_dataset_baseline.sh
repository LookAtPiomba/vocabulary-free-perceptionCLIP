#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"

dataset=$1
model=$2
use_simple_template=$3
template=$4
save_name=$5

python src/CaSED_zero_shot_inference/dataset_baseline.py --dataset="$dataset" \
  --save_name="$save_name" \
  --model="$model" \
  --template="$template" \
  --use_simple_template="$use_simple_template" \