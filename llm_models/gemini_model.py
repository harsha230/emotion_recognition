import time
from PIL import Image
import google.generativeai as genai
from google.api_core import exceptions as g_exceptions

from utils.config import GEMINI_KEY
from utils.response_utils import parse_model_response, normalize_response


def run_gemini_model(image_path: str, prompt: str, model_name: str):
    genai.configure(api_key=GEMINI_KEY)

    try:
        img = Image.open(image_path)
    except Exception as e:
        return {"model_name": model_name, "error": f"Image error: {e}"}

    start = time.time()

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt, img])
        content = (response.text or "").strip()

        explanation, final_answer = parse_model_response(content)

        return normalize_response(
            model_name, explanation, final_answer,
            round(time.time() - start, 2)
        )

    except g_exceptions.ResourceExhausted:
        return {"model_name": model_name, "error": "Gemini rate limit exceeded"}
    except g_exceptions.ServiceUnavailable:
        return {"model_name": model_name, "error": "Gemini service unavailable"}
    except g_exceptions.PermissionDenied:
        return {"model_name": model_name, "error": "Invalid Gemini API key"}
    except Exception as e:
        return {"model_name": model_name, "error": str(e)}
