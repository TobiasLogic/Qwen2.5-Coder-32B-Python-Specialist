#!/usr/bin/env bash
pkill -9 -f "compile_worker" || true
pkill -9 -f "train.py" || true
tmux kill-session -t qlora 2>/dev/null || true

tmux new-session -d -s qlora 'cd /workspace/qlora && source /venv/main/bin/activate && export HF_HUB_ENABLE_HF_TRANSFER=0 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python train.py --dataset m-a-p/CodeFeedback-Filtered-Instruction,iamtarun/python_code_instructions_18k_alpaca --max-samples 20000 --epochs 1 --max-seq-len 512 --output-dir outputs > train.log 2>&1'
echo "Training started in tmux session 'qlora'"
