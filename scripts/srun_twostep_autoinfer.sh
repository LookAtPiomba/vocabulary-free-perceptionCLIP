#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"

dataset=$1
template=$2
convert_text=$3
save_name=$4


python  src/CaSED_zero_shot_inference/perceptionCaSED_two_step.py  --dataset="$dataset" \
  --template="$template" \
  --save_name="$save_name" \
  --convert_text="$convert_text" \