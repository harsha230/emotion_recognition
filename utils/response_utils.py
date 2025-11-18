from utils.utils import extract_section


def parse_model_response(content: str):
    explanation = extract_section(content, "Explanation:") or content
    final_answer = extract_section(content, "Final Answer:") or content
    return explanation, final_answer


def normalize_response(model_name: str, explanation: str, final_answer: str,
                       response_time: float, prompt_tokens=None,
                       completion_tokens=None, total_tokens=None,
                       remaining_tokens=None, cost_usd=None):
    """Return unified response dict for all models."""
    return {
        "model_name": model_name,
        "explanation": explanation,
        "final_answer": final_answer,
        "response_time": response_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "remaining_tokens": remaining_tokens,
        "cost_usd": cost_usd,
    }
