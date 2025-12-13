#!/usr/bin/env bash

set -e

# Models to be tested
MODELS=(
  "llava-llama3:8b"
  "llava:7b"
  "qwen3-vl:8b-instruct"
  "minicpm-v:8b"
  "llava:13b" 
)

# Methods / prompts to be tested
METHODS=(
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
