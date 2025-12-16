#!/usr/bin/env bash

set -e

# Models to be tested
MODELS=(
  "gemma3:12b-it-q4_K_M"
  "ministral-3:14b-instruct-2512-q4_K_M"
  "gemma3:27b-it-q4_K_M"
)


# Methods / prompts to be tested
METHODS=(
  "discrete"
  "continuous"
  "hierarchical"
)

for METHOD in "${METHODS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    ollama pull "${MODEL}"
    echo "Running experiment with model=${MODEL}, method=${METHOD}"

    python mllm_method.py \
      --model "${MODEL}" \
      --method "${METHOD}"
    ollama rm "${MODEL}"
  done
done

echo "All experiments finished."
