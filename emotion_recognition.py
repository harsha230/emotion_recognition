import os
import base64
import time
from dotenv import load_dotenv
from typing import Any
import subprocess
import json
import google.generativeai as genai
from PIL import Image
from google.api_core import exceptions as g_exceptions 
from openai import OpenAI, APIError, RateLimitError, AuthenticationError, PermissionDeniedError, BadRequestError
import ollama 
from ollama import ResponseError 
from requests.exceptions import ConnectionError as RequestsConnectionError

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

def extract_section(text: str, keyword: str) -> str:
    if keyword not in text:
        return ""
    part = text.split(keyword, 1)[1]
    return part.split("\n", 1)[0].strip()


def calculate_gpt5_cost(prompt_tokens: int, completion_tokens: int) -> float:
    INPUT_COST_PER_M = 5.00 / 1_000_000
    OUTPUT_COST_PER_M = 15.00 / 1_000_000
    return (prompt_tokens * INPUT_COST_PER_M) + (completion_tokens * OUTPUT_COST_PER_M)

def run_openai_model(image_path: str, prompt: str) -> dict[str, Any]:
    client = OpenAI(api_key=OPENAI_KEY)
    model_name = "gpt-5"
    reasoning = final_answer = ""
    prompt_tokens = completion_tokens = 0
    cost_usd = None
    MAX_CONTEXT = 128000
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert emotion recognition assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ],
                },
            ],
        )
        content = response.choices[0].message.content.strip()
        reasoning = extract_section(content, "Reasoning:")
        final_answer = extract_section(content, "Final Answer:")

        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            cost_usd = calculate_gpt5_cost(prompt_tokens, completion_tokens)
            total_tokens = getattr(usage, "total_tokens", 0)
            remaining_tokens = max(0, MAX_CONTEXT - total_tokens)
            cost_usd = calculate_gpt5_cost(prompt_tokens, completion_tokens)
        else:
            print("No usage data returned from OpenAI.")

    except RateLimitError:
        print("You exceeded your OpenAI quota.")
    except AuthenticationError:
        print("Invalid OpenAI API key. Verify your .env file.")
    except PermissionDeniedError:
        print("This key doesn’t have access to selected model.")
    except BadRequestError as e:
        print(f"Bad Request: {e}")
    except APIError:
        print("OpenAI service temporarily unavailable. Try again later.")
    except Exception as e:
        print(f"Unexpected error: {e}")

    response_time = round(time.time() - start_time, 2)
    return {
        "model_name": model_name,
        "reasoning": reasoning,
        "final_answer": final_answer,
        "response_time": response_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "remaining_tokens": remaining_tokens,
        "cost_usd": cost_usd,
    }


def run_gemini_model(image_path: str, prompt: str) -> dict[str, Any]:
    genai.configure(api_key=GEMINI_KEY)
    model_name = "gemini-2.5-flash"
    reasoning = final_answer = ""
    start_time = time.time()

    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt, img])
        content = response.text.strip()
        reasoning = extract_section(content, "Reasoning:")
        final_answer = extract_section(content, "Final Answer:")
        
    except g_exceptions.ResourceExhausted:
        print("Gemini rate-limit reached.")
    except g_exceptions.ServiceUnavailable:
        print("Gemini service temporarily unavailable. Try again later.")
    except g_exceptions.PermissionDenied:
        print("Invalid or unauthorized Gemini API key. Check your .env file.")
    except Exception as e:
        print(f"Unexpected error: {e}")

    response_time = round(time.time() - start_time, 2)
    return {
        "model_name": model_name,
        "reasoning": reasoning,
        "final_answer": final_answer,
        "response_time": response_time,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "remaining_tokens": None,
        "cost_usd": None,
    }
def run_ollama_model(image_path: str, prompt: str) -> dict[str, Any]:
    model_name = "qwen3-vl:8b"
    
    reasoning = final_answer = ""
    prompt_tokens = completion_tokens = total_tokens = 0
    
    try:
        ollama.list()
    except RequestsConnectionError:
        print("\nOllama Error: Could not connect to Ollama.")
        print("Please make sure Ollama is running")
        return {"error": "Ollama server not reachable"}
    except Exception as e:
        print(f"\nOllama Error: Unexpected error checking status: {e}")
        return {"error": "Ollama status check failed"}

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return {"error": "Image file not found"}
    except Exception as e:
        print(f"Error reading image file: {e}")
        return {"error": f"Error reading image file: {e}"}

    start_time = time.time()
    try:
        print(f"\nRunning Ollama model '{model_name}'. This may take a moment...")
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes]  
                }
            ]
        )
        content = response['message']['content'].strip()
        reasoning = extract_section(content, "Reasoning:")
        final_answer = extract_section(content, "Final Answer:")

    except ResponseError as e:
        if "model" in str(e) and "not found" in str(e):
            print(f"\nOllama Error: Model '{model_name}' not found.")
            print(f"Please run: ollama pull {model_name}")
        else:
            print(f"\nOllama API Error: {e}")
    except RequestsConnectionError:
        print("\nOllama Error: Connection lost during chat.")
    except Exception as e:
        print(f"\nOllama Error: Unexpected error during chat: {e}")

    response_time = round(time.time() - start_time, 2)
    return {
        "model_name": model_name,
        "reasoning": reasoning,
        "final_answer": final_answer,
        "response_time": response_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "remaining_tokens": None,  
        "cost_usd": 0.0,  
    }

def recognize_emotion(model_choice: str, image_path: str, prompt: str) -> dict[str, Any] | None:
    if model_choice == "openai":
        return run_openai_model(image_path, prompt)
    if model_choice == "gemini":
        return run_gemini_model(image_path, prompt)
    if model_choice == "ollama":
        return run_ollama_model(image_path, prompt)
    print("Invalid model choice.")
    return None

def display_result(result: dict) -> None:
    print("\n")
    print(f"Model Used        : {result.get('model_name', 'N/A')}")
    print(f"Response Time     : {result.get('response_time', 'N/A')} seconds")

    if result.get("cost_usd") is not None:
        if result["cost_usd"] == 0.0:
            print(f"Estimated Cost     : $0.00 (Local Model)")
        else:
            print(f"Estimated Cost     : ${result['cost_usd']:.6f}")

    if result.get("prompt_tokens") is not None:
        print("\nToken Usage Details:")
        print(f"Input (Prompt) Tokens     : {result['prompt_tokens']:,}")
        print(f"Output (Completion) Tokens : {result['completion_tokens']:,}")
        print(f"Total Tokens Used          : {result['total_tokens']:,}")
        if result.get("remaining_tokens") is not None:
            print(f"Remaining Context Tokens   : {result['remaining_tokens']:,}")
        
    print("\nReasoning:",result.get("reasoning", "No reasoning returned."))

    print("\nFinal Answer:", result.get("final_answer", "No answer returned."))

    
    
if __name__ == "__main__":
    
    prompt = """
    Analyze the person's facial expression and body pose in this image.
    Identify the dominant emotion (like happy, sad, angry, surprised, etc.).
    Then, explain why you think this emotion is expressed.
    Finally, give the answer in the format:

    Reasoning: <your reasoning>
    Final Answer: <emotion>
    """
    
    # Direct test with a sample image and model choice. Uncomment to run
    print("running main")
    result = recognize_emotion("ollama", "./images/happy.webp", prompt) 
    if result:
        display_result(result)
    
        
    # Terminal User Input
    # print("\nSelect model:\n1. OpenAI GPT-5\n2. Gemini-2.5-Flash\n3. Ollama (Local)\n")
    # choice = input("Enter 1, 2, or 3: ").strip()
    
    # model_choice = ""
    # if choice == "1":
    #     model_choice = "openai"
    # elif choice == "2":
    #     model_choice = "gemini"
    # elif choice == "3":
    #     model_choice = "ollama"
    # else:
    #     print("Invalid choice. Exiting.")
    #     model_choice = None

    # if model_choice:
    #     image_path = input("Enter the image path (e.g., ./images/photo.jpg): ").strip()
    #     if os.path.exists(image_path):
    #         result = recognize_emotion(model_choice, image_path, prompt)
    #         if result:
    #             display_result(result)
    #     else:
    #         print(f"Error: Image path not found: {image_path}")