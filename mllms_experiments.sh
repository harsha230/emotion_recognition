#!/usr/bin/env bash

set -e

# Models to be tested
MODELS=(
  "qwen3-vl:2b-instruct"
  "qwen3-vl:4b-instruct"
  "qwen3-vl:2b-thinking"
  "qwen3-vl:4b-thinking"
)

# Methods / prompts to be tested
METHODS=(
  "discrete"
  "continuous"
)

for METHOD in "${METHODS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Running experiment with model=${MODEL}, method=${METHOD}"

    python mllm_method.py \
      --model "${MODEL}" \
      --method "${METHOD}"
  done
done

echo "All experiments finished."
