#!/usr/bin/env bash

set -e
NAMES=(
  # "deepface_method"
  # "llava_13b_discrete"
  # "minicpm-v_8b_discrete"
  # "llava_7b_discrete"
  # "llava-phi3_3.8b_discrete"
  # "moondream_1.8b_discrete"
  # "qwen3-vl_30b-a3b-instruct_discrete"
  # "qwen3-vl_8b-instruct_discrete"
  # "qwen3-vl_2b-thinking_discrete"
  # "qwen3-vl_4b-instruct_discrete"
  # "qwen3-vl_2b-instruct_discrete"
  # "qwen3-vl_32b-instruct_discrete"
  # "llava-llama3_8b_discrete"
  # "llava_13b_hierarchical"
  # "llava_13b_hierarchical"
  # "qwen3-vl_8b-instruct_hierarchical"
  # "qwen3-vl_2b-instruct_hierarchical"
  # "llava-llama3_8b_hierarchical"
  # "llava-phi3_3.8b_hierarchical"
  # "llava_7b_hierarchical"
  # "qwen3-vl_4b-instruct_hierarchical"
  "gemma3_12b-it-q4_K_M_hierarchical"
  "gemma3_4b-it-q4_K_M_hierarchical"
  "gemma3_12b-it-q4_K_M_discrete"
  "gemma3_4b-it-q4_K_M_discrete"
  "ministral-3_3b-instruct-2512-q4_K_M_discrete"
)


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
