from cv_models.deepface_model import run_deepface_model
import pandas as pd
from tqdm import tqdm
import time
from datasets import Dataset

print("Loading dataset...")
df = pd.read_csv(f"dataset_path_labels.csv")


start_time = time.time()
print("Running DeepFace model on dataset...")
for index, row in tqdm(df.iterrows(), total=len(df)):
    result = run_deepface_model(image_path=row["image_path"])
    df.loc[index, "label"] = row["emotion"]
    df.loc[index, "image_path"] = row["image_path"]
    df.loc[index, "model_name"] = result.get("model_name", "N/A")
    df.loc[index, "response_time"] = result.get("response_time", "N/A")
    df.loc[index, "prediction_continuos"] = result.get("explanation", "N/A")
    df.loc[index, "predicted_emotion"] = result.get("final_answer", "N/A")

df.to_csv(f"deepface_results.csv", index=False)

dataset = Dataset.from_pandas(df)
print(dataset)

token = ""
dataset.push_to_hub(f"Emotion-Aware-AI-Assistant/deepface_method", token=token)