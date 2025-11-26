import time
import ollama
from ollama import ResponseError
from requests.exceptions import ConnectionError as RequestsConnectionError

from utils.image_utils import read_image_bytes
from utils.response_utils import parse_model_response, normalize_response


def run_ollama_model(image_path: str, prompt: str, model_name: str):

    try:
        ollama.list()
    except RequestsConnectionError:
        print("\nOllama Error: Could not connect to Ollama.")
        print("Please make sure Ollama is running.")
        return {"error": "Ollama server not reachable", "model_name": model_name}
    except Exception as e:
        print(f"\nOllama Error: Unexpected error checking status: {e}")
        return {"error": "Ollama status check failed", "model_name": model_name}

    img_bytes = read_image_bytes(image_path)
    if img_bytes is None:
        print(f"Error: Image file not found at {image_path}")
        return {"error": "Image file not found", "model_name": model_name}

    start = time.time()

    try:
        # print(f"\nRunning Ollama model '{model_name}'. This may take a moment...")

        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [img_bytes],
            }],
        )

        content = response["message"]["content"].strip()

        explanation, final_answer = parse_model_response(content)

        return normalize_response(
            model_name,
            explanation,
            final_answer,
            round(time.time() - start, 2),
            cost_usd=0.0
        )

    except ResponseError as e:
        if "model" in str(e) and "not found" in str(e):
            print(f"\nOllama Error: Model '{model_name}' not found.")
            print(f"Please run: ollama pull {model_name}")
        else:
            print(f"\nOllama API Error: {e}")
        return {"error": str(e), "model_name": model_name}

    except RequestsConnectionError:
        print("\nOllama Error: Connection lost during chat.")
        return {"error": "Connection lost during chat", "model_name": model_name}

    except Exception as e:
        print(f"\nOllama Error: Unexpected error during chat: {e}")
        return {"error": str(e), "model_name": model_name}
