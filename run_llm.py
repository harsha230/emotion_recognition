from llm_models.openai_model import run_openai_model
from llm_models.gemini_model import run_gemini_model
from llm_models.ollama_model import run_ollama_model
from utils.prompts import prompt3_step1, prompt3_step2_neg, prompt3_step2_pos
def recognize_emotion(image_path: str, prompt: str, model: str):
    m = model.lower()

    if "gpt" in m:
        return run_openai_model(image_path, prompt, model)

    elif "gemini" in m:
        return run_gemini_model(image_path, prompt, model)

    else:
        return run_ollama_model(image_path, prompt, model)

    # else:
    #     return {"error": f"Unknown model: {model}", "model_name": model}


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
        print(f"Input (Prompt) Tokens      : {result['prompt_tokens']:,}")
        print(f"Output (Completion) Tokens  : {result['completion_tokens']:,}")
        print(f"Total Tokens Used           : {result['total_tokens']:,}")
        if result.get("remaining_tokens") is not None:
            print(f"Remaining Context Tokens    : {result['remaining_tokens']:,}")

    print("\nExplanation:", result.get("explanation", "No explanation returned."))
    print("\nFinal Answer:", result.get("final_answer", "No answer returned."))

def run_hierarchical_method(image_path: str, model: str):
    """
    Runs the 2-step hierarchical classification on a single image.
    Step 1: Determine Valence (Positive/Negative/Neutral)
    Step 2: Determine Specific Emotion (if not Neutral)
    """
    print(f"\n--- Starting Hierarchical Analysis on {model} ---")
    
    print("Step 1: Checking Valence...")
    step1_result = recognize_emotion(image_path, prompt3_step1, model)
    valence = step1_result.get("final_answer", "").strip().lower()
    
    explanation = f"[Step 1: {valence}] {step1_result.get('explanation', '')}"
    final_answer = valence

    final_result = step1_result.copy() 
    print(f"Valence: {valence}")
    print(f"S: {step1_result}")
    if "negative" in valence:
        print(f"Step 1 result was '{valence}'. Proceeding to Negative Branch...")
        step2_result = recognize_emotion(image_path, prompt3_step2_neg, model)
        
        final_answer = step2_result.get("final_answer", "N/A")
        explanation += f" | [Step 2: Negative] {step2_result.get('explanation', '')}"
        
        final_result = step2_result
        
    elif "positive" in valence:
        print(f"Step 1 result was '{valence}'. Proceeding to Positive Branch...")
        step2_result = recognize_emotion(image_path, prompt3_step2_pos, model)
        
        final_answer = step2_result.get("final_answer", "N/A")
        explanation += f" | [Step 2: Positive] {step2_result.get('explanation', '')}"
        
        final_result = step2_result

    elif "neutral" in valence:
        print(f"Step 1 result was '{valence}'. No further steps needed.")
        final_answer = "neutral"
    
    else:
        print(f"Warning: Unexpected Step 1 result: '{valence}'. defaulting to that value.")
        final_answer = valence

    final_result["final_answer"] = final_answer
    final_result["explanation"] = explanation
    
    return final_result

if __name__ == "__main__":
    prompt = """ 
     Analyze the person's facial expression and body pose in this image. Identify the dominant emotion (like happy, sad, angry, surprised, etc.). Then, explain why you think this emotion is expressed.
     Finally, give the answer in the format: 
     Explanation: <your explanation> Final Answer: <emotion> 
     """

    model = "gemini-2.5-flash"   # Change to: gpt-5, gemini-2.5-flash, qwen3-vl:8b
    image = "images/happy.webp"

    # result = recognize_emotion(image, prompt, model)
    # display_result(result)

    result = run_hierarchical_method(image, model)
    display_result(result)
    
    # print("\nSelect model:")
    # print("1. OpenAI GPT-5")
    # print("2. Gemini-2.5-Flash")
    # print("3. Ollama Local (e.g., qwen3-vl:8b)")
    #
    # choice = input("Enter 1, 2, or 3: ").strip()
    #
    # if choice == "1":
    #     model_choice = "gpt-5"
    # elif choice == "2":
    #     model_choice = "gemini-2.5-flash"
    # elif choice == "3":
    #     model_choice = "qwen3-vl:8b"    # or any local Ollama model
    # else:
    #     print("Invalid choice. Exiting.")
    #     model_choice = None
    #
    # if model_choice:
    #     image_path = input("Enter image path (e.g., images/photo.jpg): ").strip()
    #     if os.path.exists(image_path):
    #         result = recognize_emotion(image_path, prompt, model_choice)
    #         display_result(result)
    #     else:
    #         print(f"Error: Image not found at {image_path}")

