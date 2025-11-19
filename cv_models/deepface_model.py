import time
from deepface import DeepFace
from utils.response_utils import normalize_response

def run_deepface_model(image_path: str, model_name: str = "deepface"):
    start_time = time.time()

    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False
        )

        result = result[0] if isinstance(result, list) else result

        emotions = result.get("emotion", {})
        dominant = result.get("dominant_emotion", "unknown")

        percentage_lines = "\n".join(
            f"{emo}: {score:.2f}%"
            for emo, score in emotions.items()
        )

        explanation = (
            f"DeepFace multiple emotion probabilities:\n"
            f"{percentage_lines}\n\n"
            f"Dominant emotion: {dominant}"
        )

        return normalize_response(
            model_name=model_name,
            explanation=explanation,
            final_answer=dominant,
            response_time=round(time.time() - start_time, 2),
            cost_usd=0.0
        )

    except Exception as e:
        return {"model_name": model_name, "error": f"DeepFace error: {str(e)}"}
