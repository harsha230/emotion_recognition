# Emotion Recognition(Image-based)
---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/harsha230/emotion_recognition.git
cd emotion_recognition
```

### 2. Create and Activate a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys in `.env` File

Create a file named `.env` in the root folder and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

**Important:** 
- Do not share or commit this file to version control
- The `.gitignore` file is configured to protect it
- Only add keys for the services you plan to use

### 5. Install and Configure Ollama

If you want to test local models with Ollama:

#### Install Ollama
1. Download and install from [ollama.com](https://ollama.com)
2. Follow the installation instructions for your operating system

#### Run Ollama
- **macOS:** Look for the Llama icon in your menu bar
- **Windows:** Check your system tray for the Llama icon
- **Linux:** Ollama typically runs as a service

#### Pull a Vision Model

Before running the script, download the model you want to test:

```bash
# Pull the default model (qwen3-vl)
ollama pull qwen3-vl:8b

# Or pull llava
ollama pull llava:latest

```

**Note:** First-time model downloads may be several gigabytes and take some time.

---

## How to Run

### Option 1: Interactive Mode

Run the script with prompts to select your model and image:

```bash
python3 main.py
```

Follow the interactive prompts:

```
Select model:
1. OpenAI GPT-4
2. Gemini 2.0
3. Ollama (Local)

Enter 1, 2, or 3: 3

Enter the image path (e.g., ./images/photo.jpg): ./images/happy.webp
```

**Note:** You may need to uncomment the "Terminal User Input" section in `emotion_recognition.py` for this mode.

### Option 2: Direct Script Execution

Edit `emotion_recognition.py` to test a specific model automatically:

```python
if __name__ == "__main__":
    prompt = """
    Analyze the person's facial expression...
    """
    
    # Change "ollama" to "openai" or "gemini" to test other models
    result = recognize_emotion("ollama", "./images/happy.webp", prompt) 
    if result:
        display_result(result)
```

Then run:

```bash
python3 emotion_recognition.py
```

### Option 3: Testing Different Ollama Models

To switch between Ollama models:

1. Open `emotion_recognition.py`
2. Find the `run_ollama_model` function
3. Modify the `model_name` variable:

```python
def run_ollama_model(image_path: str, prompt: str) -> dict[str, Any]:
    # CHANGE THIS LINE TO TEST A DIFFERENT OLLAMA MODEL
    model_name = "qwen3-vl:8b"  # Default
    
    # To test llava, change to:
    # model_name = "llava:latest"
    
    # Or any other vision model:
    # model_name = "llava:13b"
    
    # ... rest of the function
```

4. **Important:** Make sure you've pulled the model first!

```bash
ollama pull llava:latest  # or whatever model you want to use
```

---


## 📄 License

MIT License © 2025
