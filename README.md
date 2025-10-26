# Chatbot using Falcon-7B Instruct

This project demonstrates a large language model chatbot built with the Falcon-7B-Instruct model, deployed and tested on Google Colab (T4 GPU).
The chatbot is capable of holding contextual, multi-turn conversations using Hugging Face’s Transformers library.

---

## Model Overview
Model: tiiuae/falcon-7b-instruct

Type: Decoder-only, causal language model

Framework: Hugging Face Transformers

Hardware: NVIDIA T4 GPU (Google Colab)

---

## How It Works
1. The Falcon-7B-Instruct model is pre-trained for chat-style instruction following.
2. A tokenizer from Hugging Face is used for encoding text input.
3. The Transformers pipeline generates text responses with top-k sampling.
4. A simple conversational loop maintains multi-turn dialog history.

---

## Core Code Example

```bash
from transformers import AutoTokenizer, pipeline
import torch

model = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

prompt = "What is relativity?"
response = pipeline(prompt, max_length=200, do_sample=True, top_k=10)
print(response[0]['generated_text'])
```

---

## Screenshot
<img width="900" height="740" alt="Notebook Image" src="https://github.com/user-attachments/assets/376a0762-4ef1-49ae-b3b0-c5190413c35e" />

---

#### Requirements
[pip install torch transformers accelerate]

---

#### Example Output

> What is relativity?
Bob: Relativity is the scientific theory developed by Albert Einstein explaining that space and time are interwoven and relative to the observer's motion.

---
