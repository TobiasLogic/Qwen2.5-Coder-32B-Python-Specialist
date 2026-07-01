import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
DEFAULT_MODEL = "huihui-ai/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated"
def parse_args():
    p = argparse.ArgumentParser(
        description="Merge LoRA + export merged safetensors and GGUF (q4_k_m).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adapter", required=True,
                   help="Path to the trained LoRA adapter dir "
                        "(e.g. outputs/final_adapter).")
    p.add_argument("--base", default=None,
                   help="Base model id/path. Defaults to the adapter's recorded "
                        "base, else the project default.")
    p.add_argument("--merged-dir", default="outputs/merged",
                   help="Output dir for the merged 16-bit safetensors model.")
    p.add_argument("--gguf-dir", default="outputs/gguf",
                   help="Output dir for the GGUF file.")
    p.add_argument("--gguf-quant", default="q4_k_m",
                   help="GGUF quantization method.")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--llama-cpp-dir", default="llama.cpp",
                   help="llama.cpp checkout for the manual GGUF fallback.")
    p.add_argument("--skip-gguf", action="store_true",
                   help="Only produce the merged safetensors model.")
    p.add_argument("--skip-merged-save", action="store_true",
                   help="Skip re-saving merged safetensors (e.g. if it already exists).")
    return p.parse_args()
_WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]
_TOOLCALL_CONVO = [
    {"role": "system", "content": "You are a coding agent that can call tools."},
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"type": "function",
                     "function": {"name": "get_weather",
                                  "arguments": {"city": "Paris"}}}]},
    {"role": "tool", "name": "get_weather", "content": "18C, clear"},
    {"role": "assistant", "content": "It's 18C and clear in Paris."},
]
def verify_chat_template(tokenizer_dir: str) -> bool:
    from transformers import AutoTokenizer
    print(f"\n[verify] loading tokenizer from {tokenizer_dir} ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    if not getattr(tok, "chat_template", None):
        print("[verify] FAIL: merged tokenizer has NO chat_template.")
        return False
    print("[verify] chat_template present.")
    ok = True
    try:
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": "You are helpful."},
             {"role": "user", "content": "Weather in Paris?"}],
            tools=_WEATHER_TOOL, add_generation_prompt=True, tokenize=False,
        )
        if "get_weather" in prompt:
            print("[verify] OK: tool schema rendered into the prompt.")
        else:
            print("[verify] WARN: 'get_weather' not found in tool-rendered prompt.")
            ok = False
    except Exception as e:
        print(f"[verify] FAIL: apply_chat_template(tools=...) raised: {e}")
        ok = False
    try:
        convo = tok.apply_chat_template(
            _TOOLCALL_CONVO, add_generation_prompt=False, tokenize=False,
        )
        if "get_weather" in convo and "Paris" in convo:
            print("[verify] OK: assistant tool_call + tool result render.")
        else:
            print("[verify] WARN: tool-call conversation did not render as expected.")
            ok = False
        print("\n----- rendered tool-call sample (first 900 chars) -----")
        print(convo[:900])
        print("----- end sample -----\n")
    except Exception as e:
        print(f"[verify] FAIL: rendering tool-call conversation raised: {e}")
        ok = False
    return ok
def verify_gguf_template(gguf_path: str):
    try:
        from gguf import GGUFReader
    except Exception:
        print("[verify] (gguf python lib not available; skipping GGUF metadata check)")
        return
    try:
        reader = GGUFReader(gguf_path)
        keys = {f.name for f in reader.fields.values()}
        if "tokenizer.chat_template" in keys:
            print("[verify] OK: GGUF embeds tokenizer.chat_template.")
        else:
            print("[verify] WARN: GGUF has no tokenizer.chat_template metadata. "
                  "Set a TEMPLATE in the Ollama Modelfile if tool-calling misbehaves.")
    except Exception as e:
        print(f"[verify] (could not read GGUF metadata: {e})")
def find_file(root: str, names):
    for n in names:
        hit = list(Path(root).rglob(n))
        if hit:
            return str(hit[0])
    return None
def manual_gguf(merged_dir: str, gguf_dir: str, quant: str, llama_dir: str) -> str:
    print("\n[gguf] falling back to manual llama.cpp conversion ...")
    if not Path(llama_dir).exists():
        print(f"[gguf] cloning llama.cpp into {llama_dir} ...")
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ggml-org/llama.cpp", llama_dir], check=True)
    convert = find_file(llama_dir, ["convert_hf_to_gguf.py"])
    if not convert:
        raise RuntimeError(f"convert_hf_to_gguf.py not found under {llama_dir}")
    os.makedirs(gguf_dir, exist_ok=True)
    f16 = os.path.join(gguf_dir, "model-f16.gguf")
    print(f"[gguf] converting merged model -> {f16} (f16) ...")
    subprocess.run([sys.executable, convert, merged_dir,
                    "--outfile", f16, "--outtype", "f16"], check=True)
    quant_bin = find_file(llama_dir, ["llama-quantize", "quantize"])
    if not quant_bin:
        print(f"[gguf] building llama.cpp (needed for quantize) ...")
        subprocess.run(["cmake", "-B", os.path.join(llama_dir, "build"),
                        "-S", llama_dir, "-DGGML_CUDA=OFF"], check=True)
        subprocess.run(["cmake", "--build", os.path.join(llama_dir, "build"),
                        "--config", "Release", "-j", "--target", "llama-quantize"],
                       check=True)
        quant_bin = find_file(llama_dir, ["llama-quantize", "quantize"])
    if not quant_bin:
        raise RuntimeError("llama-quantize binary not found after build.")
    out = os.path.join(gguf_dir, f"model-{quant.upper()}.gguf")
    print(f"[gguf] quantizing -> {out} ({quant}) ...")
    subprocess.run([quant_bin, f16, out, quant], check=True)
    os.remove(f16)  
    return out
def write_ollama_modelfile(gguf_path: str, gguf_dir: str):
    modelfile = os.path.join(gguf_dir, "Modelfile")
    rel = os.path.basename(gguf_path)
    content = (
        f"FROM ./{rel}\n"
        "
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.8\n"
        'PARAMETER stop "<|im_end|>"\n'
    )
    with open(modelfile, "w") as f:
        f.write(content)
    print(f"[ollama] wrote {modelfile}")
    print(f"[ollama] import with:  ollama create qwen3-coder-abliterated -f {modelfile}")
def main():
    args = parse_args()
    from unsloth import FastModel
    base = args.base
    if base is None:
        cfg = Path(args.adapter) / "adapter_config.json"
        if cfg.exists():
            base = json.loads(cfg.read_text()).get("base_model_name_or_path")
        base = base or DEFAULT_MODEL
    print(f"[load] base={base}\n[load] adapter={args.adapter}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter,      
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )
    if not args.skip_merged_save:
        print(f"\n[merge] saving merged 16-bit model -> {args.merged_dir} ...")
        model.save_pretrained_merged(args.merged_dir, tokenizer,
                                     save_method="merged_16bit")
        print("[merge] done.")
    else:
        print("[merge] skipped (--skip-merged-save).")
    template_ok = verify_chat_template(args.merged_dir)
    gguf_path = None
    if not args.skip_gguf:
        try:
            print(f"\n[gguf] exporting GGUF ({args.gguf_quant}) via Unsloth -> {args.gguf_dir} ...")
            model.save_pretrained_gguf(args.gguf_dir, tokenizer,
                                       quantization_method=args.gguf_quant)
            hits = list(Path(args.gguf_dir).rglob(f"*{args.gguf_quant.upper()}*.gguf")) \
                or list(Path(args.gguf_dir).rglob("*.gguf"))
            gguf_path = str(hits[0]) if hits else None
        except Exception as e:
            print(f"[gguf] Unsloth GGUF export failed: {e}")
            gguf_path = manual_gguf(args.merged_dir, args.gguf_dir,
                                    args.gguf_quant, args.llama_cpp_dir)
        if gguf_path:
            print(f"[gguf] GGUF ready: {gguf_path}")
            verify_gguf_template(gguf_path)
            write_ollama_modelfile(gguf_path, args.gguf_dir)
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print(f"  merged 16-bit : {args.merged_dir}")
    print(f"  gguf ({args.gguf_quant}) : {gguf_path or '(skipped/failed)'}")
    print(f"  chat template : {'VERIFIED (tools + tool_calls render)' if template_ok else 'CHECK WARNINGS ABOVE'}")
    print("=" * 70)
    if not template_ok:
        sys.exit(2)
if __name__ == "__main__":
    main()
