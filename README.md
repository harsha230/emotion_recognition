# Emotion Recognition (Image-based)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/emotion_recognition.git
cd emotion_recognition
```

### 2. Create and Activate a Virtual Environment

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` File
In the root folder, create a file named `.env` and add your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

**Do not share or push this file to GitHub.**`.gitignore` already protects it.

### 5. Run the Script

```bash
python emotion_recognition.py
```

Then follow the prompts:

```
Select model:
1. OpenAI GPT-4V
2. Gemini-Pro-Vision
Enter 1 or 2:
```
Provide your image path:

```
Enter the image path (e.g., ./images/photo.jpg):
```

## License
MIT License © 2025
