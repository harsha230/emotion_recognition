import re


def extract_section(text: str, keyword: str) -> str:
    if not text or not keyword:
        return ""

    base_keyword = keyword.strip().rstrip(":")
    pattern = rf"(?i){re.escape(base_keyword)}\s*[:\-–]*"
    match = re.search(pattern, text)

    if not match:
        return ""

    start = match.end()
    remaining = text[start:].strip()

    if re.search(r"(?i)final\s*answer", base_keyword):
        return remaining.strip()

    next_final = re.search(r"(?i)final\s*answer\s*[:\-–]*", remaining)

    if next_final:
        return remaining[: next_final.start()].strip()

    return remaining.strip()


def calculate_gpt5_cost(prompt_tokens: int, completion_tokens: int) -> float:
    INPUT_COST = 5.00 / 1_000_000
    OUTPUT_COST = 15.00 / 1_000_000
    return (prompt_tokens * INPUT_COST) + (completion_tokens * OUTPUT_COST)
