#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"

dataset=$1
save_name=$2

python  src/CaSED_zero_shot_inference/perceptionCaSED.py  --dataset="$dataset" \
  --save_name="$save_name" \