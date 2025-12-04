from dotenv import load_dotenv
import os
import pandas as pd
from datasets import load_dataset
import re
from datasets import Dataset
import argparse

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()
name = args.name

ds = load_dataset(f"Emotion-Aware-AI-Assistant/{name}", token=hf_token)
df = pd.DataFrame(ds['train'])

bigger = df[df['predicted_emotion'].str.len() > 9]
refused = bigger[bigger['predicted_emotion'].str.contains('sorry', case=False) | bigger['predicted_emotion'].str.contains('cannot', case=False)]
wrong = bigger[~bigger.index.isin(refused.index)]

df['predicted_emotion'] = ['emotion_refused' if idx in refused.index else pred for idx, pred in zip(df.index, df['predicted_emotion'])]
df['predicted_emotion'] = ['wrong_format' if idx in wrong.index else pred for idx, pred in zip(df.index, df['predicted_emotion'])]



TARGET_EMOTIONS = [
    'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'
]

EMOTION_VARIANTS = {
    'anger':      [r'anger', r'angry'],
    'disgust':    [r'disgust', r'disgusted'],
    'fear':       [r'fear', r'afraid'],
    'happiness':  [r'happy', r'happiness'],
    'neutral':    [r'neutral', r'normal'],
    'sadness':    [r'sad(ness)?'],
    'surprise':   [r'surprise(d)?']
}

COMPILED_PATTERNS = {
    emotion: re.compile(r'|'.join(variants), flags=re.IGNORECASE)
    for emotion, variants in EMOTION_VARIANTS.items()
}

def map_emotion(label: str):
    if not isinstance(label, str):
        return None

    text = label.strip().lower()

    matches = []
    for emotion, pattern in COMPILED_PATTERNS.items():
        if pattern.search(text):
            matches.append(emotion)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        priority = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness',
       'surprise']
        for p in priority:
            if p in matches:
                return p

    for emotion in TARGET_EMOTIONS:
        if emotion in text:
            return emotion

    return label

def normalize_emotions(df, col='predicted_emotion', new_col='normalized_emotion'):
    df[new_col] = df[col].apply(map_emotion)
    return df


df = normalize_emotions(df)
def map_format(label: str):
    if not isinstance(label, str):
        return None

    if '<' in label or '>' in label or 'format' in label or '"' in label or "." in label or "Nan" in label:
        return 'wrong_format'
    
    return label


df['normalized_emotion'] = df['normalized_emotion'].apply(map_format)

valid_labels = TARGET_EMOTIONS + ['wrong_format', 'emotion_refused']

def map_emotion(label: str):
    if not isinstance(label, str):
        return None
    if label in valid_labels:
        return label
    else:
        return 'emotion_not_listed'


df['normalized_emotion'] = df['normalized_emotion'].apply(map_emotion)

df["correct"] = (df["label"] == df["normalized_emotion"]).astype(int)

method_name = name.split("_")[-1]           # "discrete"
model_name = "_".join(name.split("_")[:-1])   # "minicpm-v_8b"

df_results = df[['model_name', 'image_path', 'label', 'normalized_emotion', 'correct', 'response_time']]

df_results.rename(columns={
    'normalized_emotion': 'predicted_emotion'
}, inplace=True)

df_results['method'] = method_name


dataset = Dataset.from_pandas(df_results)
print(dataset)
dataset.push_to_hub(f"Emotion-Aware-AI-Assistant/{name}_standardized", token=hf_token)

