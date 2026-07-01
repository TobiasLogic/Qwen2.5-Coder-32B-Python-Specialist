import argparse
import inspect
import json
import os
from pathlib import Path
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEFAULT_MODEL = "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"
CANDIDATE_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",   
    "gate_proj", "up_proj", "down_proj",       
    "gate_up_proj",                             
]
def parse_args():
    p = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen2.5-Coder-32B into a Python specialist with Unsloth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path.")
    p.add_argument("--dataset", required=True,
                   help="Path to a local .jsonl/.json file, a local HF dataset "
                        "dir, OR a HF hub dataset id. NOT hardcoded.")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--messages-field", default=None,
                   help="Column holding the conversation. Auto-detected from "
                        "{conversations, messages, conversation} if unset.")
    p.add_argument("--tools-field", default="tools",
                   help="Optional column holding per-record tool schemas.")
    p.add_argument("--max-samples", type=int, default=0,
                   help="If >0, truncate dataset to this many rows (smoke tests).")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=2.0,
                   help="2-3 recommended. Use 3 for small datasets.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Effective batch = batch-size * grad-accum.")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-scheduler", default="linear")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--save-steps", type=int, default=50,
                   help="Write a checkpoint every N steps (instance-death safety).")
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--resume", action="store_true",
                   help="Auto-resume from the latest checkpoint in --output-dir.")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Explicit checkpoint dir (overrides --resume auto-detect).")
    p.add_argument("--no-train-on-responses-only", dest="responses_only",
                   action="store_false",
                   help="By default loss is computed on assistant turns only "
                        "(recommended for tool-use). Pass this to train on all tokens.")
    p.set_defaults(responses_only=True)
    p.add_argument("--hub-model-id", default=None,
                   help="If set, push checkpoints to this HF repo id for true "
                        "instance-death resilience (needs `huggingface-cli login`).")
    p.add_argument("--hub-private", action="store_true", default=True)
    return p.parse_args()
def latest_checkpoint(output_dir: str):
    ckpts = list(Path(output_dir).glob("checkpoint-*"))
    ckpts = [c for c in ckpts if c.is_dir() and c.name.split("-")[-1].isdigit()]
    if not ckpts:
        return None
    return str(max(ckpts, key=lambda c: int(c.name.split("-")[-1])))
def detect_target_modules(model):
    present = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in CANDIDATE_TARGETS and hasattr(module, "weight"):
            present.add(leaf)
    ordered = [t for t in CANDIDATE_TARGETS if t in present]
    return ordered
def build_kwargs_for(callable_obj, desired: dict):
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(desired), []
    accepted = set(sig.parameters)
    kept = {k: v for k, v in desired.items() if k in accepted}
    dropped = [k for k in desired if k not in accepted]
    return kept, dropped
def load_conversations(args, tokenizer):
    from datasets import load_dataset, concatenate_datasets
    dataset_paths = [p.strip() for p in args.dataset.split(",")]
    loaded_datasets = []
    for path in dataset_paths:
        if os.path.exists(path):
            suffix = Path(path).suffix.lower()
            if suffix in (".jsonl", ".json"):
                ds = load_dataset("json", data_files=path, split="train")
            else:
                ds = load_dataset(path, split=args.dataset_split)
        else:
            ds = load_dataset(path, split=args.dataset_split)
        if "instruction" in ds.column_names and "output" in ds.column_names:
            print(f"[data] detected Alpaca format in {path}; converting to conversations.")
            def alpaca_to_conv(example):
                prompt = example["instruction"]
                if example.get("input"):
                    prompt += "\n" + example["input"]
                return {"conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": example["output"]}
                ]}
            ds = ds.map(alpaca_to_conv, remove_columns=ds.column_names)
        elif "query" in ds.column_names and "answer" in ds.column_names:
            print(f"[data] detected QA format in {path}; converting to conversations.")
            def qa_to_conv(example):
                return {"conversations": [
                    {"role": "user", "content": example["query"]},
                    {"role": "assistant", "content": example["answer"]}
                ]}
            ds = ds.map(qa_to_conv, remove_columns=ds.column_names)
        else:
            field = args.messages_field
            if field is None:
                for cand in ("conversations", "messages", "conversation"):
                    if cand in ds.column_names:
                        field = cand
                        break
            if field and field != "conversations":
                ds = ds.rename_column(field, "conversations")
            sample = ds[0]["conversations"] if len(ds) > 0 else []
            first_msg = sample[0] if isinstance(sample, list) and sample else {}
            if isinstance(first_msg, dict) and "from" in first_msg and "value" in first_msg:
                from unsloth.chat_templates import standardize_sharegpt
                ds = standardize_sharegpt(ds)
                print(f"[data] detected ShareGPT in {path}; standardized to role/content.")
            else:
                print(f"[data] detected OpenAI-style in {path}; using as-is.")
        loaded_datasets.append(ds)
    ds = concatenate_datasets(loaded_datasets)
    if args.max_samples and args.max_samples > 0:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.max_samples, len(ds))))
        print(f"[data] truncated to {args.max_samples} random samples.")
    has_tools = args.tools_field in ds.column_names
    def _format(example):
        convo = example["conversations"]
        tools = example.get(args.tools_field) if has_tools else None
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except Exception:
                tools = None
        try:
            text = tokenizer.apply_chat_template(
                convo, tools=tools, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
        return {"text": text}
    keep = [c for c in ds.column_names]
    ds = ds.map(_format, remove_columns=keep, desc="apply_chat_template")
    return ds
def main():
    args = parse_args()
    from unsloth import FastModel
    import torch
    os.makedirs(args.output_dir, exist_ok=True)
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print("=" * 70)
    print(f"Model      : {args.model}")
    print(f"Dataset    : {args.dataset} (split={args.dataset_split})")
    print(f"LoRA       : r={args.r} alpha={args.alpha} dropout={args.lora_dropout}")
    print(f"Optim      : lr={args.lr} epochs={args.epochs} "
          f"bs={args.batch_size} ga={args.grad_accum} maxlen={args.max_seq_len}")
    print(f"Precision  : {'bf16' if bf16_ok else 'fp16'}")
    print(f"Output     : {args.output_dir} (save every {args.save_steps} steps)")
    print("=" * 70)
    load_kwargs = dict(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,          
    )
    try:
        model, tokenizer = FastModel.from_pretrained(full_finetuning=False, **load_kwargs)
    except TypeError:
        model, tokenizer = FastModel.from_pretrained(**load_kwargs)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    print(f"[lora] hardcoded target_modules to attention only for VRAM safety: {target_modules}")
    model = FastModel.get_peft_model(
        model,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        use_gradient_checkpointing="unsloth",   
        random_state=args.seed,
        use_rslora=False,
    )
    train_ds = load_conversations(args, tokenizer)
    print(f"[data] {len(train_ds)} training examples ready.")
    from trl import SFTTrainer, SFTConfig
    desired_cfg = dict(
        output_dir=args.output_dir,
        dataloader_num_workers=16,
        dataloader_prefetch_factor=2,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        optim="paged_adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        seed=args.seed,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        max_length=args.max_seq_len,
        packing=False,
        dataset_num_proc=2,
        gradient_checkpointing=True,
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private,
        hub_strategy="all_checkpoints" if args.hub_model_id else "every_save",
    )
    cfg_kwargs, dropped = build_kwargs_for(SFTConfig, desired_cfg)
    if dropped:
        print(f"[cfg] ignoring args not in this TRL SFTConfig: {dropped}")
    sft_config = SFTConfig(**cfg_kwargs)
    trainer_desired = dict(
        model=model,
        train_dataset=train_ds,
        args=sft_config,
        tokenizer=tokenizer,          
        processing_class=tokenizer,   
        dataset_text_field="text",    
        max_seq_length=args.max_seq_len,
    )
    trainer_kwargs, _ = build_kwargs_for(SFTTrainer.__init__, trainer_desired)
    if "processing_class" in trainer_kwargs and "tokenizer" in trainer_kwargs:
        trainer_kwargs.pop("tokenizer")
    trainer = SFTTrainer(**trainer_kwargs)
    if args.responses_only:
        try:
            from unsloth.chat_templates import train_on_responses_only
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
            print("[mask] training on assistant responses only.")
        except Exception as e:
            print(f"[mask] train_on_responses_only skipped ({e}); training on all tokens.")
    resume = args.resume_from_checkpoint
    if resume is None and args.resume:
        resume = latest_checkpoint(args.output_dir)
        print(f"[resume] {'resuming from ' + resume if resume else 'no checkpoint found; fresh start'}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        used = torch.cuda.max_memory_reserved() / 1e9
        print(f"[gpu] {gpu.name} | {gpu.total_memory/1e9:.1f}GB total | {used:.1f}GB reserved pre-train")
    trainer.train(resume_from_checkpoint=resume)
    final_dir = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[done] LoRA adapter + tokenizer saved to: {final_dir}")
    print(f"[next] merge + export with:\n"
          f"       python export.py --adapter {final_dir} "
          f"--merged-dir {args.output_dir}/merged --gguf-dir {args.output_dir}/gguf")
if __name__ == "__main__":
    main()
