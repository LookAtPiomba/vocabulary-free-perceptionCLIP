#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"

dataset=$1
template=$2
model=$3
save_name=$4
metric=$5

python  src/CaSED_zero_shot_inference/perceptionclip_one_step.py  --dataset="$dataset" \
  --template="$template" \
  --save_name="$save_name" \
  --model="$model" \
  --metric="$metric"


