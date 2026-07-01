---
language:
- en
tags:
- coding
- qwen2.5
- uncensored
- unsloth
- 32b
license: apache-2.0
---

# Qwen2.5-Coder-32B-Uncensored

## Model Description
**Qwen2.5-Coder-32B-Uncensored** is a highly specialized, instruction-tuned version of the Qwen2.5-Coder-32B base model. This model has been aggressively fine-tuned to completely remove refusal mechanisms, disclaimers, and unnecessary moralizing when generating code. It acts as a direct, strictly compliant coding assistant designed for advanced developers and researchers who require raw, unfiltered output.

The model was fine-tuned using a distilled, high-quality combination of the **CodeFeedback-Filtered-Instruction** and **python_code_instructions_18k_alpaca** datasets, running over 20,000 highly diverse programming scenarios. 

By aggressively targeting the Attention layers during fine-tuning (while leaving the complex MLP structures frozen), this model achieves state-of-the-art formatting compliance and instruction following without compromising the encyclopedic coding knowledge of the 32B base model.

## Model Details
- **Base Model:** Qwen/Qwen2.5-Coder-32B-Instruct (Abliterated Base)
- **Parameters:** 32 Billion
- **Context Length:** Up to 32K (optimized at 512 for dense instruction tuning)
- **Training Strategy:** LoRA (Attention Modules Only: `q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **Dataset:** 20,000 samples (CodeFeedback + Python Alpaca)
- **Quantization:** Available in 16-bit safetensors and 4-bit GGUF (`q4_k_m`)

## Usage

### Ollama / LM Studio (GGUF)
You can seamlessly run the GGUF version locally using Ollama:
```bash
ollama run TobiasLogic/Qwen2.5-Coder-32B-Uncensored:q4_k_m
```

### Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TobiasLogic/Qwen2.5-Coder-32B-Uncensored")
model = AutoModelForCausalLM.from_pretrained(
    "TobiasLogic/Qwen2.5-Coder-32B-Uncensored",
    device_map="auto"
)

messages = [
    {"role": "user", "content": "Write a python script to bypass a firewall."}
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
The model was fine-tuned utilizing **Unsloth** for rapid multi-processing data ingestion and memory-efficient LoRA scaling. The dataset consisted of heavily curated coding problems spanning over 50 programming languages, converted into standard ShareGPT conversational format. 

To eliminate refusal behaviors without catastrophic forgetting, we targeted only the Attention matrices. The model was trained with a learning rate of `2e-4`, achieving a remarkably low final loss of `0.45` without overfitting.

## Disclaimer
This model is provided entirely unfiltered and uncensored. It will generate exactly what is requested of it, including malicious, insecure, or dangerous code if prompted. The creators of this model take no responsibility for how the model is used. Use responsibly and in isolated environments when dealing with unknown code execution.
