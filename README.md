# QLoRA fine-tune: Qwen2.5-Coder-32B → Python Specialist (on Vast.ai)

QLoRA fine-tuning pipeline that turns
[`Qwen/Qwen2.5-Coder-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
(dense, 32B, Apache-2.0) into
**[`TobiasLogic/Qwen2.5-Coder-32B-Python-Specialist`](https://huggingface.co/TobiasLogic/Qwen2.5-Coder-32B-Python-Specialist)** —
a Python-focused instruction tune that **keeps the base model's alignment and
safety behavior**. Built with **Unsloth**, loading the base 4-bit from the
pre-quantized
[`unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit).
Exports a merged 16-bit model **+** a GGUF (`q4_k_m`) for local inference in
llama.cpp / Ollama.

> **Alignment note.** This is a *censored* specialist: the base model's safety
> filters are retained. Training targets the **attention layers only**, to
> sharpen Python and instruction-formatting behavior without disturbing the base
> model's encyclopedic coding knowledge or its alignment.

| File | Purpose |
|------|---------|
| `train.py` | QLoRA fine-tune (Unsloth `FastModel`, 4-bit, r=16/α=16, **attention-only** LoRA, assistant-only loss). |
| `export.py` | Merge adapter → 16-bit safetensors **+** GGUF `q4_k_m`; verify the chat / tool-calling template survives. |
| `start_training.sh` | One-shot launcher: runs the canonical training command in a detached `tmux` session, logging to `train.log`. |
| `HF_README.md` | The model card published alongside the weights on Hugging Face. |

---

## What this trains

- **Base:** `unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit` (dense Qwen2.5-Coder-32B, loaded in 4-bit).
- **Method:** QLoRA — LoRA adapters over a frozen 4-bit base, via Unsloth.
- **LoRA target modules:** attention only — `q_proj, k_proj, v_proj, o_proj`.
  This is **hardcoded** in `train.py` (`target_modules`, ~line 186) for VRAM
  safety; the MLP is left frozen. To also adapt the MLP, edit that line to add
  `gate_proj, up_proj, down_proj`.
- **Loss masking:** assistant turns only (`train_on_responses_only`) by default;
  user turns are masked. Disable with `--no-train-on-responses-only`.
- **The published build** used
  [`m-a-p/CodeFeedback-Filtered-Instruction`](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction)
  \+ [`iamtarun/python_code_instructions_18k_alpaca`](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca),
  **20,000** samples, **1 epoch**, `max-seq-len 512`, `lr 2e-4` — reaching a
  final loss ≈ `0.45`. That exact run is `start_training.sh`.

The dataset is **never hardcoded** in `train.py` — pass any dataset (or a
comma-separated list) with `--dataset`. See [Dataset format](#dataset-format).

---

## Hardware requirements (read first)

- **GPU:** the published build was trained on a single **A100 80 GB**. The 4-bit
  base is ~19 GB resident before optimizer state + activations, so a 24 GB card
  is marginal for a 32B QLoRA — prefer **≥ 40 GB**, or reduce
  `--max-seq-len` / `--batch-size`.
- **Disk:** ~**150 GB free** for the *full* pipeline. The pre-quantized 4-bit
  base downloads at ~19 GB, but `export.py` writes a **~62 GB** merged fp16
  model and a **~62 GB** intermediate f16 GGUF before quantizing down to the
  final **~19 GB** `q4_k_m`. Provision **≥ 150 GB** (200 GB comfortable). If you
  only want the LoRA adapter, far less is needed — pass `--skip-gguf` to
  `export.py`, or skip export entirely.
- **RAM:** ≥ 64 GB (the merge + GGUF stage streams shards through system RAM).
- **Persistence:** on a standard Vast instance `/workspace` is **not** a volume —
  a *recycle/destroy* wipes it (a *stop/start* does not). See
  [Surviving a dead instance](#surviving-a-dropped-or-destroyed-instance).

---

## Environment

Use an Unsloth-ready image (recommended on Vast) or install into a fresh venv:

```bash
python -m venv /venv/main && source /venv/main/bin/activate
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install hf_transfer   # faster HF downloads (both scripts opt in)
```

`export.py`'s manual GGUF fallback clones and builds **llama.cpp** on demand, so
also have `git`, `cmake`, and a C/C++ toolchain available for that path.

---

## Step-by-step over SSH (fresh instance)

Commands are grouped: `local$` runs on your laptop, `vast#` on the instance.

### 1. Copy the pipeline onto the instance

Use your instance's SSH details from the Vast dashboard, and put the code on the
**big disk** (`/workspace`):

```bash
# from your laptop, in the folder containing train.py/export.py/start_training.sh:
local$ ssh -p <port> root@<instance-ip> 'mkdir -p /workspace/qlora'
local$ scp -P <port> train.py export.py start_training.sh HF_README.md \
           root@<instance-ip>:/workspace/qlora/
# (or just: git clone this repo into /workspace/qlora on the instance)
```

### 2. Set up the environment

```bash
vast# cd /workspace/qlora
vast# source /venv/main/bin/activate     # or create it (see Environment above)
```

### 3. Smoke-test the loop first

The dataset path is passed with `--dataset`; a tiny `--max-samples` run proves
the whole loop before you commit hours of compute:

```bash
vast# python train.py \
        --dataset iamtarun/python_code_instructions_18k_alpaca \
        --output-dir outputs --max-samples 8 --epochs 1 \
        --max-seq-len 512 --save-steps 5
```

### 4. Train (in `tmux`, so an SSH drop doesn't kill it)

The published build is one command — `start_training.sh` wraps exactly this in a
detached `tmux` session and tees to `train.log`:

```bash
vast# bash start_training.sh
# equivalently, by hand:
vast# tmux new -s qlora        # detach: Ctrl-b then d ; reattach: tmux attach -t qlora
vast# source /venv/main/bin/activate
vast# python train.py \
        --dataset m-a-p/CodeFeedback-Filtered-Instruction,iamtarun/python_code_instructions_18k_alpaca \
        --max-samples 20000 --epochs 1 --max-seq-len 512 --output-dir outputs
```

Defaults: `r=16`, `alpha=16`, attention-only LoRA, `lr=2e-4`, gradient
checkpointing on, assistant-only loss. The first run downloads the ~19 GB 4-bit
base into `$HF_HOME`; it's cached and reused on later runs / exports.

Monitor from another shell:

```bash
vast# watch -n5 nvidia-smi
vast# tail -f /workspace/qlora/train.log                       # start_training.sh log
vast# tail -f /workspace/qlora/outputs/trainer_state.json      # step / loss progress
```

### 5. Export: merge + GGUF (verifies the chat template)

```bash
vast# python export.py \
        --adapter /workspace/qlora/outputs/final_adapter \
        --merged-dir /workspace/qlora/outputs/merged \
        --gguf-dir  /workspace/qlora/outputs/gguf \
        --gguf-quant q4_k_m
```

This writes `outputs/merged/` (16-bit safetensors) and
`outputs/gguf/model-Q4_K_M.gguf`, then **renders a tool-call conversation
through the exported tokenizer** and fails loudly if the chat template / tool
schema didn't survive the merge. It also drops an Ollama `Modelfile`. If
Unsloth's GGUF path errors, it automatically falls back to a manual llama.cpp
convert + quantize.

### 6. Pull artifacts down / run locally

```bash
local$ scp -P <port> -r root@<instance-ip>:/workspace/qlora/outputs/gguf ./gguf
local$ cd gguf && ollama create qwen2.5-coder-32b-python-specialist -f Modelfile
local$ ollama run qwen2.5-coder-32b-python-specialist
```

---

## Dataset format

`train.py` auto-detects the input schema and normalizes everything to
conversations before applying Qwen2.5-Coder's chat template. Supported inputs:

- **Alpaca** — `instruction` / `output` (+ optional `input`). Auto-converted to
  a user/assistant turn. *(Both published datasets land here / as QA.)*
- **QA** — `query` / `answer`. Auto-converted.
- **Conversational** — an OpenAI-style `messages` **or** ShareGPT
  `conversations` field (ShareGPT `{from,value}` is standardized automatically).
  Pick a different column name with `--messages-field <name>`.

Local files (`.jsonl` / `.json`), a local HF dataset dir, or a **HF hub dataset
id** all work, and you can pass several comma-separated (they're concatenated):

```bash
--dataset m-a-p/CodeFeedback-Filtered-Instruction,iamtarun/python_code_instructions_18k_alpaca
```

**Tool-use is also supported** (which is why `export.py` verifies it): put
per-record tool schemas in a `tools` field (JSON list or JSON string) and they
are rendered via `apply_chat_template(..., tools=...)`. Example conversational
row with a tool call:

```json
{"messages": [
  {"role": "system", "content": "You are a Python coding assistant."},
  {"role": "user", "content": "How many .py files are in src/?"},
  {"role": "assistant", "content": "", "tool_calls": [
     {"type": "function", "function": {"name": "run_shell",
      "arguments": {"command": "find src -name '*.py' | wc -l"}}}]},
  {"role": "tool", "name": "run_shell", "content": "42"},
  {"role": "assistant", "content": "There are 42 Python files in src/."}
 ],
 "tools": [{"type": "function", "function": {
     "name": "run_shell", "description": "Run a shell command.",
     "parameters": {"type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"]}}}]}
```

---

## Surviving a dropped or destroyed instance

Checkpoints are written to `outputs/checkpoint-<step>` every `--save-steps`.

**SSH dropped / instance stopped then started** (filesystem intact):

```bash
vast# tmux attach -t qlora        # if the process was still running, you're back
# otherwise resume from the latest local checkpoint:
vast# python train.py --dataset <same> --output-dir /workspace/qlora/outputs --resume
```

`--resume` auto-selects the highest `checkpoint-*` in `--output-dir`. Pin a
specific one with `--resume-from-checkpoint outputs/checkpoint-300`.

**Instance recycled/destroyed** (local disk wiped — `/workspace` is not a volume
here). Local checkpoints are gone, so push them off-box while training by adding
`--hub-model-id` to stream checkpoints to a private HF repo:

```bash
vast# huggingface-cli login          # once, with a write token
vast# python train.py --dataset <same> --output-dir /workspace/qlora/outputs \
        --save-steps 50 --hub-model-id <you>/qwen25-coder-python-qlora-ckpts
```

Then on a **new** instance, after setting up the environment:

```bash
vast# huggingface-cli download <you>/qwen25-coder-python-qlora-ckpts \
        --local-dir /workspace/qlora/outputs --include "checkpoint-*/*"
vast# python train.py --dataset <same> --output-dir /workspace/qlora/outputs --resume
```

Tip: also `huggingface-cli download` the base model first on a fresh box to warm
the cache.

---

## Command reference

**`train.py`** (key flags — run `python train.py --help` for all):

| Flag | Default | Notes |
|------|---------|-------|
| `--dataset` | *(required)* | `.jsonl`/`.json` path, local HF dataset dir, or hub id. Comma-separate to concatenate. |
| `--model` | `unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit` | Base model id or local path. |
| `--output-dir` | `outputs` | Checkpoints + `final_adapter/` land here. |
| `--r` / `--alpha` | `16` / `16` | LoRA rank / scaling. |
| `--max-seq-len` | `2048` | The published build used `512`. Raise for long traces if VRAM allows. |
| `--lr` | `2e-4` | |
| `--epochs` | `2` | Published build used `1` over 20k samples. |
| `--batch-size` / `--grad-accum` | `1` / `8` | Effective batch = 8. |
| `--save-steps` | `50` | Checkpoint cadence (instance-death safety). |
| `--resume` / `--resume-from-checkpoint` | off | See resume section. |
| `--hub-model-id` | none | Push checkpoints to HF for destroy-safety. |
| `--max-samples` | `0` | Truncate data for quick smoke tests. |
| `--no-train-on-responses-only` | off | Train on all tokens instead of assistant-only. |

**`export.py`:** `--adapter` (required), `--merged-dir`, `--gguf-dir`,
`--gguf-quant q4_k_m`, `--skip-gguf`, `--skip-merged-save`, `--base` (override
the base recorded in the adapter).

---

## Troubleshooting

- **CUDA OOM during training:** lower `--max-seq-len` (e.g. 512 → 256), keep
  `--batch-size 1`, raise `--grad-accum`. A 32B QLoRA is heavy — the 4-bit base
  alone is ~19 GB resident.
- **Want the MLP adapted too?** `train.py` deliberately hardcodes attention-only
  target modules for VRAM safety. Add `gate_proj, up_proj, down_proj` to
  `target_modules` (~line 186) if you have the headroom.
- **GGUF export fails in Unsloth:** `export.py` auto-falls back to a manual
  `llama.cpp` convert + quantize; ensure `git`, `cmake`, and a C/C++ toolchain
  are installed.
- **Tool-calling weird in Ollama:** confirm `export.py` printed
  `GGUF embeds tokenizer.chat_template`; if not, add a `TEMPLATE` block to the
  generated `Modelfile`.
- **Disk fills during export:** the merge (~62 GB) and f16 GGUF (~62 GB) are
  large and transient. Use `--skip-gguf` if you only need the merged model, or
  free space before exporting.

---

## Model card

The published weights and their card live at
[`TobiasLogic/Qwen2.5-Coder-32B-Python-Specialist`](https://huggingface.co/TobiasLogic/Qwen2.5-Coder-32B-Python-Specialist);
`HF_README.md` in this repo is that card. License: **Apache-2.0**, inherited
from the base model.
