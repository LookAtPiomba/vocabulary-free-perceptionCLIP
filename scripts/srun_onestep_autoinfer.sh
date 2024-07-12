#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"

dataset=$1
template=$2
save_name=$3
metric=$4

python  src/CaSED_zero_shot_inference/perceptionCaSED_one_step.py  --dataset="$dataset" \
  --template="$template" \
  --save_name="$save_name" \
  --metric="$metric"