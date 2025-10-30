import os
import base64
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from google.api_core import exceptions as g_exceptions 
from openai import OpenAI, APIError, RateLimitError, AuthenticationError, PermissionDeniedError, BadRequestError


load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

print("Select model:\n1. OpenAI GPT-5\n2. Gemini-2.5-Flash \n")
choice = input("Enter 1 or 2: ")

if choice == "1":
    model = "openai"
elif choice == "2":
    model = "gemini"
else:
    print("Invalid choice.")
    exit()

image_path = input("Enter the image path (e.g., ./images/photo.jpg): ")

if not os.path.exists(image_path):
    print("Image path not found.")
    exit()

prompt = """
Analyze the person's facial expression and body pose in this image.
Identify the dominant emotion (like happy, sad, angry, surprised, etc.).
Then, explain why you think this emotion is expressed.
Finally, give the answer in the format:

Reasoning: <your reasoning>
Final Answer: <emotion>
"""

if model == "openai":
    client = OpenAI(api_key=OPENAI_KEY)

    with open(image_path, "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="gpt-5", 
            messages=[
                {"role": "system", "content": "You are an expert emotion recognition assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ]
        )
        print("\nModel Response: \n")
        print(response.choices[0].message.content)

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

        
        
elif model == "gemini":

    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    try:
        img = Image.open(image_path)
        response = model.generate_content([prompt, img])

        print("\nModel Response: \n")
        print(response.text)

    except g_exceptions.ResourceExhausted:
        print("Gemini rate-limit reached.")
    except g_exceptions.ServiceUnavailable:
        print("Gemini service temporarily unavailable. Try again later.")
    except g_exceptions.PermissionDenied:
        print("Invalid or unauthorized Gemini API key. Check your .env file.")
    except Exception as e:
        print(f"Unexpected error: {e}")