---
language:
- en
tags:
- coding
- qwen2.5
- python
- unsloth
- 32b
license: apache-2.0
---

# Qwen2.5-Coder-32B-Python-Specialist

## Model Description
**Qwen2.5-Coder-32B-Python-Specialist** is an instruction-tuned version of the standard Qwen2.5-Coder-32B base model. This model has been specifically fine-tuned on a high-quality blend of Python and generalized coding instruction datasets to enhance its proficiency in formatting compliance, multi-turn coding problem solving, and Python-specific tasks.

*Note: The model retains the original safety filters and alignment of the Qwen2.5 base model.*

The model was fine-tuned using a distilled, high-quality combination of the **CodeFeedback-Filtered-Instruction** and **python_code_instructions_18k_alpaca** datasets, running over 20,000 highly diverse programming scenarios. 

By aggressively targeting the Attention layers during fine-tuning (while leaving the complex MLP structures frozen), this model achieves state-of-the-art formatting compliance and instruction following without compromising the encyclopedic coding knowledge of the 32B base model.

## Model Details
- **Base Model:** unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit
- **Parameters:** 32 Billion
- **Context Length:** Up to 32K (optimized at 512 for dense instruction tuning)
- **Training Strategy:** LoRA (Attention Modules Only: `q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **Dataset:** 20,000 samples (CodeFeedback + Python Alpaca)
- **Quantization:** Available in 16-bit safetensors and 4-bit GGUF (`q4_k_m`)

## Usage

### Ollama / LM Studio (GGUF)
You can seamlessly run the GGUF version locally using Ollama:
```bash
ollama run hf.co/TobiasLogic/Qwen2.5-Coder-32B-Python-Specialist:Q4_K_M
```

### Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TobiasLogic/Qwen2.5-Coder-32B-Python-Specialist")
model = AutoModelForCausalLM.from_pretrained(
    "TobiasLogic/Qwen2.5-Coder-32B-Python-Specialist",
    device_map="auto"
)

messages = [
    {"role": "user", "content": "Write a python script to parse a CSV file."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## Training Data & Methodology
The model was fine-tuned utilizing **Unsloth** for rapid multi-processing data ingestion and memory-efficient LoRA scaling. The dataset consisted of heavily curated coding problems, heavily indexing on Python, converted into standard ShareGPT conversational format. 

To enhance instruction following without catastrophic forgetting, we targeted only the Attention matrices. The model was trained with a learning rate of `2e-4`, achieving a remarkably low final loss of `0.45` without overfitting.
