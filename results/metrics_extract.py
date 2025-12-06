from datasets import Dataset
from dotenv import load_dotenv
import os
import pandas as pd
from datasets import load_dataset
import argparse
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
import json
print("Starting metrics extraction script...")
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()
name = args.name

print(f"Extracting metrics for {name}...")
ds = load_dataset(f"Emotion-Aware-AI-Assistant/{name}_standardized", token=hf_token)
df = pd.DataFrame(ds['train'])


EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

INVALID_PREDICTIONS = ["wrong_format", "emotion_refused", "emotion_not_listed"]


df_eval = df[df["label"].isin(EMOTIONS)]

y_true = df_eval["label"]
y_pred = df_eval["predicted_emotion"]


y_pred_clean = [
    pred if pred in EMOTIONS else "INVALID"
    for pred in y_pred
]

accuracy = accuracy_score(y_true, y_pred_clean)
macro_f1 = f1_score(y_true, y_pred_clean, average="macro", labels=EMOTIONS)
macro_precision = precision_score(y_true, y_pred_clean, average="macro", labels=EMOTIONS)
macro_recall = recall_score(y_true, y_pred_clean, average="macro", labels=EMOTIONS)

f1_per_class = f1_score(y_true, y_pred_clean, average=None, labels=EMOTIONS)
precision_per_class = precision_score(y_true, y_pred_clean, average=None, labels=EMOTIONS)
recall_per_class = recall_score(y_true, y_pred_clean, average=None, labels=EMOTIONS)

labels_with_invalid = EMOTIONS + ["INVALID"]
cm = confusion_matrix(y_true, y_pred_clean, labels=labels_with_invalid)

df_confusion = pd.DataFrame(cm, index=labels_with_invalid, columns=labels_with_invalid)
confusion_json = df_confusion.to_json()


invalid_counts = {}

for emotion in EMOTIONS:
    subset = df_eval[df_eval["label"] == emotion]
    invalid_counts[emotion] = {
        reason: int((subset["predicted_emotion"] == reason).sum())
        for reason in INVALID_PREDICTIONS
    }

df_invalid = pd.DataFrame.from_dict(invalid_counts, orient="index")
df_invalid.columns = INVALID_PREDICTIONS

invalid_json = df_invalid.to_json()


data = {
    "accuracy": [accuracy],
    "macro_f1": [macro_f1],
    "macro_precision": [macro_precision],
    "macro_recall": [macro_recall],
    "confusion_matrix_json": [confusion_json],
    "invalid_counts_json": [invalid_json],
}

for emotion, value in zip(EMOTIONS, f1_per_class):
    data[f"f1_{emotion}"] = [value]

for emotion, value in zip(EMOTIONS, precision_per_class):
    data[f"precision_{emotion}"] = [value]

for emotion, value in zip(EMOTIONS, recall_per_class):
    data[f"recall_{emotion}"] = [value]

df_results = pd.DataFrame(data)

df_results['model_name'] = df['model_name'].iloc[0]
df_results['method'] = df['method'].iloc[0]

df_results = df_results[['model_name', 'method', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall',
       'confusion_matrix_json', 'invalid_counts_json', 'f1_anger',
       'f1_disgust', 'f1_fear', 'f1_happiness', 'f1_neutral', 'f1_sadness',
       'f1_surprise', 'precision_anger', 'precision_disgust', 'precision_fear',
       'precision_happiness', 'precision_neutral', 'precision_sadness',
       'precision_surprise', 'recall_anger', 'recall_disgust', 'recall_fear',
       'recall_happiness', 'recall_neutral', 'recall_sadness',
       'recall_surprise']]

wrong_format = len(df[df['predicted_emotion'] == 'wrong_format'])
emotion_refused = len(df[df['predicted_emotion'] == 'emotion_refused'])
wrong_emotion = len(df[df['predicted_emotion'] == 'wrong_emotion'])

total_time = df['response_time'].sum()
total_time_hours = total_time / 3600
mean_time = df['response_time'].mean()
response_time_std = df['response_time'].std()
response_time_mode = df['response_time'].mode()[0]

model_name = df['model_name'].iloc[0]

df_info = pd.DataFrame({
    "wrong_format": [wrong_format],
    "emotion_refused": [emotion_refused],
    "emotion_not_listed": [wrong_emotion],
    "total_time": [total_time],
    "total_time_hours": [total_time_hours],
    "mean_time": [mean_time],
    "response_time_std": [response_time_std],
    "response_time_mode": [response_time_mode]
})
df_results = pd.concat([df_results, df_info], axis=1)

dataset = Dataset.from_pandas(df_results)
dataset.push_to_hub(f"Emotion-Aware-AI-Assistant/{name}_results", token=hf_token)