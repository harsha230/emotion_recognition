# Choosing Model

I first looked at leaderboard rankings to select candidate models, but most of the top models are not currently available in Ollama. Because of that, I narrowed the scope and focused on the most commonly used and more recent open-source models that are supported by Ollama.

### Models supported by Ollama:

**qwen3-vl**

- 2b
- 4b
- 8b
- 30b
- 32b
- 235b

llava

- 7b
- 13b
- 34b

minicpm-v

- 8b

llama3.2-vision

- 11b
- 90b

moondream

- 1.8b

**gemma3**

- 270m
- 1b
- 4b
- 12b
- 27b

### Decision:

I was deciding between Qwen3-VL and Gemma 3, since both offer multiple model sizes and include relatively small variants.

However, while Gemma 3 has a very small model (270M parameters), it does not provide a much larger version, which prevents us from comparing performance across small and large scales. In contrast, the smallest Qwen3-VL model has 2B parameters, and the family goes up to 235B, which we could run on a sufficiently powerful machine.

**For this reason, I chose Qwen3-VL as the first model we are going to analyze.**

I’ll use this sizes of models and versions:

[**qwen3-vl:2b-instruct**](https://ollama.com/library/qwen3-vl:2b-instruct)

[**qwen3-vl:4b-instruct**](https://ollama.com/library/qwen3-vl:4b-instruct)

[**qwen3-vl:8b-instruct**](https://ollama.com/library/qwen3-vl:8b-instruct)

[**qwen3-vl:30b-a3b-instruct**](https://ollama.com/library/qwen3-vl:30b-a3b-instruct)

[**qwen3-vl:32b-instruct**](https://ollama.com/library/qwen3-vl:32b-instruct)

[**qwen3-vl:235b-a22b-instruct**](https://ollama.com/library/qwen3-vl:235b-a22b-instruct)

Note that it would also be interesting to use the ‘thinking’ version to evaluate its impact on classification quality.

---

## Qwen3-vl Summary:

### Model sizes & versions

The Qwen3-VL family is offered in both dense and mixture-of-experts (MoE) architectures, with “Instruct” and “Thinking” editions (mirroring the language-only Qwen3 variants). Key sizes include:

- Dense: 2B, 4B, 8B, 32B parameters. ([GitHub](https://github.com/QwenLM/Qwen3-VL?utm_source=chatgpt.com))
- MoE: 30B total parameters (A3B active experts) and 235B total parameters (A22B active experts) for flagship reasoning capabilities. ([GitHub](https://github.com/QwenLM/Qwen3-VL?utm_source=chatgpt.com))
- Context window: The model supports very long context lengths: 256K tokens natively (and reportedly extendable to ~1 million tokens) in the vision-language setting. ([Kanaries Docs](https://docs.kanaries.net/articles/qwen3-vl?utm_source=chatgpt.com))
- Editions: “Instruct” version (general instruct-following) and “Thinking” version (reasoning/harder tasks) for each size. ([GitHub](https://github.com/QwenLM/Qwen3-VL?utm_source=chatgpt.com))

### Training, fine-tuning, instruction-tuning & usage

While the publicly available documentation does **not** provide a full peer-reviewed paper yet (as of this summary date), the blog, GitHub repo and docs give substantial information on how Qwen3-VL has been prepared and is used.

### Pre-training / base vision-language setup

- The Qwen3-VL family is described as a next-generation VLM that upgrades both vision and language capabilities: deeper visual perception & reasoning, spatial/3D grounding, video understanding, OCR in many languages. ([GitHub](https://github.com/QwenLM/Qwen3-VL?utm_source=chatgpt.com))
- It supports “joint” multimodal training: text + image + (video) + OCR, with long context and agentic capabilities (e.g., GUI element understanding) built in. ([Ollama](https://ollama.com/library/qwen3-vl?utm_source=chatgpt.com))
- The official blog notes: “this generation delivers … extended context length, enhanced spatial and video dynamics comprehension, and stronger agent interaction capabilities.” ([GitHub](https://github.com/QwenLM/Qwen3-VL?utm_source=chatgpt.com))
- One article notes support for vision, video, OCR, multi-image, with recommended fine-tuning scripts. ([Unsloth Docs](https://docs.unsloth.ai/models/qwen3-vl-how-to-run-and-fine-tune?utm_source=chatgpt.com))

### Instruction-tuning / “Instruct” and “Thinking” variants

- Each size has an “Instruct” version (optimised for regular vision-language instruction-following) and a “Thinking” version (optimised for deeper reasoning tasks). ([GitHub](https://github.com/QwenLM/Qwen3-VL?utm_source=chatgpt.com))
- The “Thinking” edition emphasises step-by-step reasoning in the vision + language domain: e.g., complex spatial reasoning, multi-image problems, STEM + vision tasks. ([Kanaries Docs](https://docs.kanaries.net/articles/qwen3-vl?utm_source=chatgpt.com))
- Fine-tuning/usage docs indicate you can further fine-tune the model on mixed-modality data (image + text, or video + text) using frameworks like Unsloth. ([Medium](https://medium.com/data-science-collective/qwen3-vl-fine-tuning-on-your-computer-0496ac81984b?utm_source=chatgpt.com))
- Training scripts for large models (e.g., 30B-A3B) show the need for substantial GPU resources and mixed modality dataset (images, video frames, OCR tasks). ([Swift Docs](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html?utm_source=chatgpt.com))

**What we know**

- There exists a full vision-language variant, Qwen3-VL, with strong support for image + text + long context, and reasoning.
- Model sizes: you can choose from smaller dense (2B/4B/8B) to large MoE (30B/235B). For research/experiments you might pick the 4B or 8B version (Instruct) to try fine-tuning on your own data.
- The “Thinking” edition is especially relevant if you care about reasoning over images (e.g., planned navigation, reasoning about 3D scenes).
- Fine-tuning is supported (via Unsloth, etc) and you can integrate multi-image, video, OCR tasks.