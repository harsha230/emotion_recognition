#!/usr/bin/env bash

set -e
NAMES=(
  "deepface_method"
  "llava_13b_discrete"
  "minicpm-v_8b_discrete"
  "llava_7b_discrete"
  "llava-phi3_3.8b_discrete"
  "moondream_1.8b_discrete"
  "qwen3-vl_30b-a3b-instruct_discrete"
  "qwen3-vl_8b-instruct_discrete"
  "qwen3-vl_2b-thinking_discrete"
  "qwen3-vl_4b-instruct_discrete"
  "qwen3-vl_2b-instruct_discrete"
  "qwen3-vl_32b-instruct_discrete"
  "llava-llama3_8b_discrete"
)
# NAMES=(
#   "qwen3-vl_4b-instruct_continuous"
#   "qwen3-vl_2b-instruct_continuous"
# )

echo "Starting standardizing..."
for NAME in "${NAMES[@]}"; do
  python standardizing.py \
    --name "${NAME}"
done

echo "Starting metrics extraction..."
for NAME in "${NAMES[@]}"; do
  python metrics_extract.py \
    --name "${NAME}"
done

echo "All processes completed."
