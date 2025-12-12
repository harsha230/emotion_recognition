#!/usr/bin/env bash

set -e


NAMES=(
  "qwen3-vl_8b-instruct_continuous"
  "llava-phi3_3.8b_continuous"
  "llava_7b_continuous"
  "minicpm-v_8b_continuous"
  "qwen3-vl_2b-instruct_continuous"
  "qwen3-vl_4b-instruct_continuous"
)


for NAME in "${NAMES[@]}"; do
  python results_continuous.py \
    --name "${NAME}"
done


echo "All processes completed."
