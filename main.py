from models.openai_model import run_openai_model
from models.gemini_model import run_gemini_model
from models.ollama_model import run_ollama_model

def recognize_emotion(image_path: str, prompt: str, model: str):
    m = model.lower()

    if "gpt" in m:
        return run_openai_model(image_path, prompt, model)
    elif "gemini" in m:
        return run_gemini_model(image_path, prompt, model)
    else:
        return run_ollama_model(image_path, prompt, model)

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



if __name__ == "__main__":
    prompt = """ 
     Analyze the person's facial expression and body pose in this image. Identify the dominant emotion (like happy, sad, angry, surprised, etc.). Then, explain why you think this emotion is expressed.
     Finally, give the answer in the format: 
     Explanation: <your explanation> Final Answer: <emotion> 
     """

    model = "gpt-5"   # Change to: gpt-5, gemini-2.5-flash, qwen3-vl:8b
    image = "images/happy.webp"

    result = recognize_emotion(image, prompt, model)
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

