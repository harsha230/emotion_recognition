from llm_models.openai_model import run_openai_model
from llm_models.gemini_model import run_gemini_model
from llm_models.ollama_model import run_ollama_model
import argparse
from utils.prompts import prompt1, prompt2, prompt3_step1, prompt3_step2_neg, prompt3_step2_pos
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv


load_dotenv()
token = os.getenv("HF_TOKEN")

# Prompt1 - Discrete Emotion Classification
# Prompt2 - Continuous Emotion Scoring
# Prompt3 - Hierarchical (Step 1: Valence, Step 2: Specific)
print("Starting emotion recognition script...")
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--method", type=str, required=True)
args = parser.parse_args()


def sanitize_filename(name: str) -> str:
    """
    Replace characters not allowed in file names (like ':' on Windows).
    Example: 'qwen3-vl:2b-instruct' → 'qwen3-vl_2b-instruct'
    """
    return name.replace(":", "_").replace("/", "_").replace("\\", "_")

model = args.model
method = args.method

if method == "discrete":
    prompt = prompt1
elif method == "continuous":
    prompt = prompt2
elif method == "hierarchical":
    primary_prompt = prompt3_step1
else:
    raise ValueError(f"Unknown method: {method}. Expected 'discrete' or 'continuous'.")

os.makedirs("results", exist_ok=True)
safe_model = sanitize_filename(model)
output_path = f"results/{safe_model}_{method}"
# result = recognize_emotion(image_path, prompt, model)

print(f"Loading dataset...")
df = pd.read_csv(f"dataset_path_labels.csv")


print(f"Running {model} on dataset with {method}...")
for index, row in tqdm(df.iterrows(), total=len(df)):
    path = row['image_path'].replace('\\', '/')
    if method == "hierarchical":
        step1_result = run_ollama_model(image_path=path, prompt=prompt3_step1, model_name=model)
        valence = step1_result.get("final_answer", "").strip().lower()
        
        combined_explanation = f"[Step 1: {valence}] {step1_result.get('explanation', '')}"
        result = step1_result 
        final_answer = valence 

        if "negative" in valence:
            step2_result = run_ollama_model(image_path=path, prompt=prompt3_step2_neg, model_name=model)
            final_answer = step2_result.get("final_answer", "N/A")
            combined_explanation += f" | [Step 2: Negative] {step2_result.get('explanation', '')}"
            result = step2_result 
            
        elif "positive" in valence:
            step2_result = run_ollama_model(image_path=path, prompt=prompt3_step2_pos, model_name=model)
            final_answer = step2_result.get("final_answer", "N/A")
            combined_explanation += f" | [Step 2: Positive] {step2_result.get('explanation', '')}"
            result = step2_result 
            
        elif "neutral" in valence:
            final_answer = "neutral"

        result["final_answer"] = final_answer
        result["explanation"] = combined_explanation

    else:
        result = run_ollama_model(image_path=path, prompt=prompt, model_name=model)
    # result = run_ollama_model(image_path=path, prompt=prompt, model_name=model)
    df.loc[index, "label"] = row["emotion"]
    df.loc[index, "image_path"] = row["image_path"]
    df.loc[index, "model_name"] = result.get("model_name", "N/A")
    df.loc[index, "response_time"] = result.get("response_time", "N/A")
    df.loc[index, "explanation"] = result.get("explanation", "N/A")
    df.loc[index, "predicted_emotion"] = result.get("final_answer", "N/A")

    if (index + 1) % 100 == 0:
        df.to_csv(f"{output_path}_backup.csv", index=False)
        print(f"Checkpoint saved with {index + 1} samples at {output_path}.csv")

    if (index + 1) % 500 == 0:
        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(f"Emotion-Aware-AI-Assistant/{safe_model}_{method}", token=token)


df.to_csv(f"{output_path}.csv", index=False)

dataset = Dataset.from_pandas(df)
print(dataset)

dataset.push_to_hub(f"Emotion-Aware-AI-Assistant/{safe_model}_{method}", token=token)