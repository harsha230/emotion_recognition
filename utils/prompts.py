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