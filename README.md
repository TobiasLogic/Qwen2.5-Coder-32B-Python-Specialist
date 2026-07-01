# QLoRA fine-tune: Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated (MoE) on Vast.ai

QLoRA fine-tuning pipeline for
[`huihui-ai/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated)
(Mixture-of-Experts, 30.5B total / 3.3B active, Apache 2.0) using **Unsloth**, on a
single **RTX 4090 (24 GB)** Vast.ai instance. Exports a merged model + a GGUF
(`q4_k_m`) for local tool-calling inference in llama.cpp / Ollama.

| File | Purpose |
|------|---------|
| `setup.sh` | Install Unsloth + deps into `/venv/main`; verify GPU; check disk/RAM headroom. |
| `train.py` | QLoRA fine-tune (FastModel, 4-bit, r=16/α=16, all-linear, router frozen). |
| `export.py` | Merge adapter → 16-bit safetensors **+** GGUF `q4_k_m`; verify tool-calling template survives. |
| `sample_data.jsonl` | Example of the expected conversational / tool-use data format. |

---

## ⚠️ Hardware requirements (read first)

- **GPU:** RTX 4090 24 GB is enough (QLoRA fits in ~17.5 GB VRAM).
- **Disk:** you need **~150 GB free** for the full pipeline. Why: the base is
  streamed as **~61 GB fp16**, converted to 4-bit on the fly (Unsloth does *not*
  ship a pre-quantized abliterated build). Merge (~61 GB) and GGUF (f16 ~61 GB →
  q4 ~18 GB) need more. **A 32 GB instance cannot even finish the download.**
  Provision the instance with **≥ 150 GB disk** (200 GB comfortable).
- **RAM:** ≥ 64 GB (the fp16→4-bit conversion loads shards into system RAM).
- **Persistence:** on a standard Vast instance `/workspace` is **not** a volume —
  a *recycle/destroy* wipes it (a *stop/start* does not). See
  [Surviving a dead instance](#surviving-a-dropped-or-destroyed-instance).

`setup.sh` enforces the disk floor and warns below the recommended headroom.

---

## Step-by-step over SSH (fresh instance)

Commands are grouped: `local$` runs on your laptop, `vast#` on the instance.

### 1. Copy the pipeline onto the instance

Your SSH command (from Vast) looks like:

```bash
local$ ssh -p 54457 root@115.75.223.236 -L 8080:localhost:8080
```

Put the code on the **big disk** (`/workspace`). From this project directory:

```bash
# from your laptop, in the folder containing setup.sh/train.py/export.py:
local$ scp -P 54457 setup.sh train.py export.py sample_data.jsonl \
           root@115.75.223.236:/workspace/qlora/
# (or: git clone your repo into /workspace/qlora on the instance)
```

> `scp` needs the target dir to exist. If it doesn't:
> `ssh -p 54457 root@115.75.223.236 'mkdir -p /workspace/qlora'` first.

### 2. Install + environment checks

```bash
vast# cd /workspace/qlora
vast# bash setup.sh
```

`setup.sh` activates `/venv/main`, installs Unsloth/torch/bitsandbytes/…,
prints the GPU, and validates disk/RAM. Fix any red `[error]` lines before
continuing. Re-running is cheap (installed steps are skipped).

### 3. Upload your dataset

The dataset path is **never hardcoded** — pass it with `--dataset`. See
[Dataset format](#dataset-format) below. To try the loop end-to-end first, use
the bundled sample:

```bash
vast# source /venv/main/bin/activate
vast# python train.py --dataset sample_data.jsonl --output-dir outputs \
        --max-samples 3 --epochs 1 --save-steps 5     # tiny smoke run
```

### 4. Train (in `tmux`, so an SSH drop doesn't kill it)

```bash
vast# tmux new -s train          # detach with Ctrl-b then d ; reattach: tmux attach -t train
vast# source /venv/main/bin/activate
vast# python train.py \
        --dataset /workspace/qlora/your_data.jsonl \
        --output-dir /workspace/qlora/outputs \
        --epochs 2 \
        --save-steps 50
```

Defaults already match the spec: `r=16`, `alpha=16`, all-linear target modules
(auto-detected), `lr=2e-4`, gradient checkpointing on, MoE router frozen. The
first run downloads the ~61 GB base into `$HF_HOME` (`/workspace/.hf_home`);
this is cached and reused on later runs/exports.

Monitor from another shell:

```bash
vast# watch -n5 nvidia-smi
vast# tail -f /workspace/qlora/outputs/trainer_state.json   # step/loss progress
```

### 5. Export: merge + GGUF (verifies tool-calling template)

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
schema didn't survive. It also drops an Ollama `Modelfile`.

### 6. Pull artifacts down / run locally

```bash
local$ scp -P 54457 -r root@115.75.223.236:/workspace/qlora/outputs/gguf ./gguf
local$ cd gguf && ollama create qwen3-coder-abliterated -f Modelfile
local$ ollama run qwen3-coder-abliterated
```

---

## Dataset format

Recommended (and what `train.py` expects by default): **conversational**, one
JSON object per line (`.jsonl`), with a `messages` (OpenAI-style) **or**
`conversations` (ShareGPT) field. This is the right choice for agentic /
tool-use data because it preserves tool-call structure and maps straight onto
Qwen3-Coder's native tool-calling template. `train.py` auto-detects the field
and normalizes ShareGPT `{from,value}` automatically.

**OpenAI `messages` with tools (preferred for tool-use):**

```json
{"messages": [
  {"role": "system", "content": "You are a coding agent."},
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

See `sample_data.jsonl` for runnable examples. Notes:
- Put per-record tool schemas in a `tools` field (JSON list or JSON string).
  They're rendered via `apply_chat_template(..., tools=...)`.
- Loss is computed on **assistant turns only** by default
  (`train_on_responses_only`); tool results and user turns are masked. Disable
  with `--no-train-on-responses-only`.
- A HF hub dataset id also works: `--dataset your-org/your-dataset`.
- Other field name? `--messages-field <name>`.

---

## Surviving a dropped or destroyed instance

Checkpoints are written to `outputs/checkpoint-<step>` every `--save-steps`.

**SSH dropped / instance stopped then started** (filesystem intact):

```bash
vast# tmux attach -t train        # if the process was still running, you're back
# otherwise just resume from the latest local checkpoint:
vast# python train.py --dataset <same> --output-dir /workspace/qlora/outputs --resume
```

`--resume` auto-selects the highest `checkpoint-*` in `--output-dir`. Use
`--resume-from-checkpoint outputs/checkpoint-300` to pin a specific one.

**Instance recycled/destroyed** (local disk wiped — `/workspace` is not a
volume here). Local checkpoints are gone, so you must have pushed them off-box.
Add `--hub-model-id` when training to stream checkpoints to a private HF repo:

```bash
vast# huggingface-cli login          # once, with a write token
vast# python train.py --dataset <same> --output-dir /workspace/qlora/outputs \
        --save-steps 50 --hub-model-id <you>/qwen3coder-qlora-ckpts
```

Then on a **new** instance, after `setup.sh`:

```bash
vast# huggingface-cli download <you>/qwen3coder-qlora-ckpts \
        --local-dir /workspace/qlora/outputs --include "checkpoint-*/*"
vast# python train.py --dataset <same> --output-dir /workspace/qlora/outputs --resume
```

(Alternatively sync `outputs/` with `rclone`/`syncthing`.) Tip: also
`huggingface-cli download` the base model first on a fresh box to warm the cache.

---

## Command reference

**`train.py`** (key flags — run `python train.py --help` for all):

| Flag | Default | Notes |
|------|---------|-------|
| `--dataset` | *(required)* | `.jsonl`/`.json` path, local HF dataset dir, or hub id. |
| `--output-dir` | `outputs` | Checkpoints + `final_adapter/` land here. |
| `--r` / `--alpha` | `16` / `16` | LoRA rank / scaling. |
| `--max-seq-len` | `2048` | Raise for long agent traces if VRAM allows. |
| `--lr` | `2e-4` | |
| `--epochs` | `2` | Spec is 2–3; use 3 for small sets. |
| `--batch-size` / `--grad-accum` | `1` / `8` | Effective batch = 8. |
| `--save-steps` | `50` | Checkpoint cadence (instance-death safety). |
| `--resume` / `--resume-from-checkpoint` | off | See resume section. |
| `--hub-model-id` | none | Push checkpoints to HF for destroy-safety. |
| `--max-samples` | `0` | Truncate data for quick smoke tests. |
| `--no-train-on-responses-only` | off | Train on all tokens instead of assistant-only. |

**`export.py`:** `--adapter` (required), `--merged-dir`, `--gguf-dir`,
`--gguf-quant q4_k_m`, `--skip-gguf`, `--base` (override).

---

## Troubleshooting

- **CUDA OOM during training:** lower `--max-seq-len` (e.g. 1024), keep
  `--batch-size 1`, raise `--grad-accum`. QLoRA 30B-A3B should sit ~17–20 GB.
- **`setup.sh` disk error:** the instance is too small — relaunch/resize with
  ≥150 GB disk. (Override the floor with `DISK_FLOOR_GB=... bash setup.sh` only
  if you truly have external storage.)
- **torch wheel / CUDA mismatch:** re-run with a different wheel index, e.g.
  `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash setup.sh`.
- **LoRA seems to only touch attention:** train.py prints the auto-detected
  `target_modules` — confirm it includes `gate_proj/up_proj/down_proj` (or fused
  `gate_up_proj`). If not, your model build exposes different names.
- **GGUF export fails in Unsloth:** `export.py` auto-falls back to a manual
  `llama.cpp` convert+quantize; ensure `cmake`/`gcc` are installed (setup.sh does this).
- **Tool-calling weird in Ollama:** confirm `export.py` printed
  `GGUF embeds tokenizer.chat_template`; if not, add a `TEMPLATE` block to the
  generated `Modelfile`.
