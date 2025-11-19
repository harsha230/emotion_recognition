import time
from typing import Any
from openai import (
    OpenAI, APIError, RateLimitError, AuthenticationError,
    PermissionDeniedError, BadRequestError
)

from utils.config import OPENAI_KEY
from utils.image_utils import read_image_bytes, encode_base64
from utils.response_utils import parse_model_response, normalize_response
from utils.utils import calculate_gpt5_cost


def run_openai_model(image_path: str, prompt: str, model_name: str) -> dict[str, Any]:
    client = OpenAI(api_key=OPENAI_KEY)
    MAX_CONTEXT = 128_000

    img_bytes = read_image_bytes(image_path)
    if img_bytes is None:
        return {"model_name": model_name, "error": f"Image not found: {image_path}"}

    img_b64 = encode_base64(img_bytes)

    start_time = time.time()
    max_retries = 3
    attempt = 0

    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert emotion recognition assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}" }}
                        ],
                    },
                ],
            )

            content = response.choices[0].message.content.strip()
            explanation, final_answer = parse_model_response(content)

            usage = getattr(response, "usage", None)
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            remaining_tokens = MAX_CONTEXT - total_tokens if usage else None
            cost_usd = calculate_gpt5_cost(prompt_tokens, completion_tokens)

            return normalize_response(
                model_name, explanation, final_answer,
                round(time.time() - start_time, 2),
                prompt_tokens, completion_tokens, total_tokens,
                remaining_tokens, cost_usd
            )

        except RateLimitError:
            attempt += 1
            if attempt >= max_retries:
                return {"model_name": model_name, "error": "Rate limit exceeded after retries"}
            time.sleep(60)

        except AuthenticationError:
            return {"model_name": model_name, "error": "Invalid OpenAI API key."}
        except PermissionDeniedError:
            return {"model_name": model_name, "error": "Permission denied."}
        except BadRequestError as e:
            return {"model_name": model_name, "error": f"Bad request: {e}"}
        except APIError:
            return {"model_name": model_name, "error": "OpenAI service unavailable."}
        except Exception as e:
            return {"model_name": model_name, "error": str(e)}
