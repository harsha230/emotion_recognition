# Dataset Columns Description – Emotion Recognition Evaluation Results

This document describes each column contained in the **results dataset** generated during the evaluation of multimodal emotion-recognition models. The dataset summarizes global metrics, per-class metrics, timing statistics, and detailed diagnostic information stored in JSON format.

---

## **General Metadata**

### **`model_name`**

The name of the evaluated model (e.g., `minicpm-v:8b`, `qwen3-vl:4b-instruct`).

Used to distinguish performance across different architectures.

### **`method`**

Indicates the experimental method used for inference, such as:

- `"discrete"` → model selects one of the predefined emotion classes
- `"continuous"` → model outputs regression-based emotional scores

---

## **Global Performance Metrics**

### **`accuracy`**

Proportion of samples where the model prediction exactly matched the ground-truth emotion label.

Invalid outputs (e.g., `"wrong_format"`, `"emotion_refused"`) count as incorrect predictions.

### **`macro_f1`**

Average F1-score across all emotion classes, giving equal weight to each class regardless of frequency.

A sensitive indicator of model performance on minority classes.

### **`macro_precision`**

Mean precision across all emotion classes. Precision reflects the proportion of predicted instances of a given class that were correct.

### **`macro_recall`**

Mean recall across all emotion classes. Recall measures how well the model recovers the true occurrences of each class.

---

## **JSON Diagnostic Fields**

### **`confusion_matrix_json`**

A JSON-encoded confusion matrix containing counts of predictions vs. true labels.

The structure includes:

- rows → true labels
- columns → predicted labels
- an additional `"INVALID"` column collects predictions that were not valid emotions (e.g., `"wrong_format"`).

This allows detailed post-analysis such as error patterns, bias toward specific classes, and misclassification hotspots.

### **`invalid_counts_json`**

A JSON object summarizing, for each emotion class, how many predictions failed due to:

- `"wrong_format"` → model returned an unusable output
- `"emotion_refused"` → model refused to classify
- `"emotion_not_listed"` → model produced an emotion not included in the allowed set

This helps quantify the model’s reliability and robustness.

---

## **Per-Class Metrics**

For each class in the set:

`anger, disgust, fear, happiness, neutral, sadness, surprise`

the dataset contains three metrics:

---

### **`f1_<emotion>`**

F1-score for the specific class.

Helps evaluate how well the model distinguishes each individual emotion.

---

### **`precision_<emotion>`**

Precision for the specific class.

Shows how often predictions of that emotion were correct.

---

### **`recall_<emotion>`**

Recall for the specific class.

Measures how well the model retrieves all actual instances of the emotion.

---

## **Timing Metrics**

### **`total_time`**

Total sum of all response times for the model across all evaluated samples (in seconds or converted to hours, depending on your pipeline).

### **`mean_time`**

Average inference time per sample.

### **`response_time_std`**

Standard deviation of response times, indicating stability or variability across samples.

### **`response_time_mode`**

The most frequently occurring response time in the dataset.

If converted to hours, the dataset may also include:

- `total_time_hours`
- `mean_time_hours`
- etc.

---

# **Summary**

This results dataset provides a complete performance overview that includes:

- Global accuracy and macro-averaged metrics
- Class-specific precision, recall, and F1
- JSON diagnostic tools for confusion patterns and invalid predictions
- Timing statistics for performance benchmarking

It is structured to support model comparison, ablation studies, error analysis, and inclusion in research reporting.