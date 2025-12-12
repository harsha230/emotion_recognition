prompt1 = '''You are an expert in Facial Emotion Recognition. You will receive an image of a human face. Carefully analyze the facial expression, focusing on key cues such as eyes, eyebrows, mouth shape, tension in facial muscles, and overall affect.

Your task is to classify the single most dominant emotion expressed in the image. The possible classes are: anger, disgust, fear, happiness, neutral, sadness, surprise

Follow the format strictly:
Explanation: <explanation>
Final Answer: <one_emotion_label>'''


prompt2 = '''You are an expert in Facial Emotion Recognition. You will receive an image of a human face. Carefully analyze the facial expression, focusing on key cues such as eyes, eyebrows, mouth shape, tension in facial muscles, and overall affect.

Your task is to provide a score from 1 to 5 for each of the seven emotions, representing how strongly that emotion appears in the image. Use the following scale:
1 = Not present
2 = Very weak presence
3 = Moderate presence
4 = Strong presence
5 = Very strong or dominant presence

The emotions you must score are: anger, disgust, fear, happiness, neutral, sadness, surprise

Follow the format strictly:
Explanation: <explanation>
Final Answer:
anger: <1–5>
disgust: <1–5>
fear: <1–5>
happiness: <1–5>
neutral: <1–5>
sadness: <1–5>
surprise: <1–5>'''


prompt3_step1 = '''You are an expert in Facial Emotion Recognition. Analyze the facial expression in the image.
Your task is to determine the valence/category of the emotion expressed.

Classify the emotion into exactly one of these three categories:
1. Positive (includes happiness and surprise)
2. Negative (includes anger, fear, disgust, and sadness)
3. Neutral

Follow the format strictly:
Explanation: <explanation>
Final Answer: <Positive/Negative/Neutral>'''

# Step 2a: If Step 1 was Negative
prompt3_step2_neg = '''You are an expert in Facial Emotion Recognition. You have identified that this face expresses a NEGATIVE emotion.
Now, classify the specific negative emotion from the following list:
- anger
- disgust
- fear
- sadness

Follow the format strictly:
Explanation: <explanation>
Final Answer: <one_emotion_label>'''

# Step 2b: If Step 1 was Positive
prompt3_step2_pos = '''You are an expert in Facial Emotion Recognition. You have identified that this face expresses a POSITIVE emotion.
Now, classify the specific emotion from the following list:
- happiness
- surprise

Follow the format strictly:
Explanation: <explanation>
Final Answer: <one_emotion_label>'''
