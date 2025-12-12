import os
from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd  # necessário se formos usar conversão para pandas, mas não precisa

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

names = [
    "deepface_method",
  "llava_13b_discrete",
  "minicpm-v_8b_discrete",
  "llava_7b_discrete",
  "llava-phi3_3.8b_discrete",
  "qwen3-vl_30b-a3b-instruct_discrete",
  "qwen3-vl_8b-instruct_discrete",
  "qwen3-vl_2b-thinking_discrete",
  "qwen3-vl_4b-instruct_discrete",
  "qwen3-vl_2b-instruct_discrete",
  "qwen3-vl_32b-instruct_discrete",
  "llava-llama3_8b_discrete",
  "qwen3-vl_8b-instruct_continuous",
  "llava-phi3_3.8b_continuous",
  "llava_7b_continuous",
  "minicpm-v_8b_continuous",
  "qwen3-vl_2b-instruct_continuous",
  "qwen3-vl_4b-instruct_continuous"
]

datasets_list = []

for n in names:
    ds = load_dataset(
        f"Emotion-Aware-AI-Assistant/{n}_results",
        split="train",
        token=hf_token
    )

    # -------------------------
    # ADD tie_counts COLUMN IF MISSING
    # -------------------------
    if "tie_counts" not in ds.column_names:
        ds = ds.add_column("tie_counts", [""] * ds.num_rows)

    datasets_list.append(ds)
    print(n, ds.num_rows)

dataset_all = concatenate_datasets(datasets_list)

dataset_all.push_to_hub(
    "Emotion-Aware-AI-Assistant/all_models_results",
    token=hf_token
)
