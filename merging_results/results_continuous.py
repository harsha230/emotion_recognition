from dotenv import load_dotenv
import os
import pandas as pd
from datasets import load_dataset
import re
from datasets import Dataset
import argparse
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

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

print("AAAAAAAAAAAAAAAA")

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()
name = args.name

ds = load_dataset(f"Emotion-Aware-AI-Assistant/{name}", token=hf_token)
df = pd.DataFrame(ds['train'])
df['predicted_emotion'].iloc[0]
import re

def extract_discrete_emotion(text):
    if not isinstance(text, str) or text.strip() == "":
        return "invalid"
    
    pairs = re.findall(r'([a-zA-Z_]+):\s*(\d+)', text)
    
    if not pairs:
        return text
    
    try:
        emotion_dict = {emotion.lower(): int(score) for emotion, score in pairs}
    except:
        return text
    
    max_value = max(emotion_dict.values())
    
    max_emotions = [emo for emo, score in emotion_dict.items() if score == max_value]
    
    if len(max_emotions) > 1:
        return f"tie{len(max_emotions)}"
    
    return max_emotions[0]


df["discrete_emotion"] = df["predicted_emotion"].apply(extract_discrete_emotion)

df["discrete_emotion"].value_counts()
df.columns
df.rename(columns={"predicted_emotion": "predicted_emotion_continuous", "discrete_emotion": "predicted_emotion"}, inplace=True)
df["predicted_emotion"].value_counts()
bigger = df[df['predicted_emotion'].str.len() > 9]
refused = bigger[bigger['predicted_emotion'].str.contains('sorry', case=False) | bigger['predicted_emotion'].str.contains('cannot', case=False)]
wrong = bigger[~bigger.index.isin(refused.index)]

df['predicted_emotion'] = ['emotion_refused' if idx in refused.index else pred for idx, pred in zip(df.index, df['predicted_emotion'])]
df['predicted_emotion'] = ['wrong_format' if idx in wrong.index else pred for idx, pred in zip(df.index, df['predicted_emotion'])]

df["predicted_emotion"].value_counts()
TARGET_EMOTIONS = [ 'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise' ]


import re

def is_tie(label: str) -> bool:
    if not isinstance(label, str):
        return False
    # matches: "tie", "tie2", "tie3", "tie10", etc.
    return bool(re.fullmatch(r"tie\d*", label.lower().strip()))


valid_labels = TARGET_EMOTIONS + ['wrong_format', 'emotion_refused']
def map_emotion_final(label: str):
    if not isinstance(label, str):
        return 'emotion_not_listed'

    label_clean = label.lower().strip()

    # 1. Emoções normais
    if label_clean in TARGET_EMOTIONS:
        return label_clean

    # 2. Categorias válidas adicionais
    if label_clean in ['wrong_format', 'emotion_refused']:
        return label_clean

    # 3. Empates (aceita tie, tie2, tie3, tie7...)
    if is_tie(label_clean):
        return label_clean

    # 4. Caso nada disso encaixe → inválido
    return 'emotion_not_listed'



df["predicted_emotion"] = df["predicted_emotion"].apply(map_emotion_final)

df["predicted_emotion"].value_counts()
df["correct"] = (df["label"] == df["predicted_emotion"]).astype(int)

method_name = name.split("_")[-1]           # "discrete"
model_name = "_".join(name.split("_")[:-1])   # "minicpm-v_8b"

df = df[['model_name', 'image_path', 'label', 'predicted_emotion', 'correct', 'response_time']]
df['method'] = method_name

# Contar todos os empates (tie, tie2, tie3...)
tie_counts = (
    df["predicted_emotion"]
    .str.extract(r"(tie\d*)")[0]      # captura tie, tie2, tie3 etc
    .dropna()
    .value_counts()
    .to_dict()
)

# Converter para JSON
tie_json = json.dumps(tie_counts)
print("Tie counts JSON:", tie_json)


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
weighted_f1 = f1_score(y_true, y_pred_clean, average="weighted", labels=EMOTIONS)
weighted_precision = precision_score(y_true, y_pred_clean, average="weighted", labels=EMOTIONS)
weighted_recall = recall_score(y_true, y_pred_clean, average="weighted", labels=EMOTIONS)


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
    "weighted_f1": [weighted_f1],
    "weighted_precision": [weighted_precision],
    "weighted_recall": [weighted_recall],
    "confusion_matrix_json": [confusion_json],
    "invalid_counts_json": [invalid_json],
    "tie_counts_json": [tie_json]
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

df_results = df_results[['model_name', 'method', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'weighted_f1', 'weighted_precision', 'weighted_recall',
       'confusion_matrix_json', 'invalid_counts_json', 'f1_anger',
       'f1_disgust', 'f1_fear', 'f1_happiness', 'f1_neutral', 'f1_sadness',
       'f1_surprise', 'precision_anger', 'precision_disgust', 'precision_fear',
       'precision_happiness', 'precision_neutral', 'precision_sadness',
       'precision_surprise', 'recall_anger', 'recall_disgust', 'recall_fear',
       'recall_happiness', 'recall_neutral', 'recall_sadness',
       'recall_surprise', 'tie_counts_json']]

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
print(df_results)