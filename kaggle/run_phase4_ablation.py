"""Phase 4 causal direction ablation on arm0, arm3, and arm4.

The kernel expects an attached Kaggle dataset containing ``directions.npz``
from ``run_phase4_readout.py``. It generates paired greedy responses for 50
sycophancy prompts under no ablation, honesty-direction ablation, and
task-specific-sycophancy-direction ablation.
"""

import glob
import json
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np


OUTPUT_DIR = "/kaggle/working"
OUT_FILE = os.path.join(OUTPUT_DIR, "ablation_generations.jsonl")
HF_DATASET = "ayesha1505/inoculate-or-reflect-data"
MODEL_NAME = "Qwen/Qwen3-8B"
N_PROMPTS = 50
ABLATION_LAYERS = [15, 18, 21]
MAX_NEW_TOKENS = 300
SEED = 42

ADAPTERS = {
    "arm3": "ayesha1505/ior-arm3-crt-mixin",
    "arm4": "ayesha1505/ior-arm4-crt-repair",
}
CONDITIONS = ["none", "honesty", "task_sycophancy"]


print("=" * 72)
print("PHASE 4 — CAUSAL DIRECTION ABLATION")
print(f"  arms: arm0, {', '.join(ADAPTERS)}")
print(f"  conditions: {CONDITIONS}")
print(f"  layers: {ABLATION_LAYERS}")
print(f"  prompts: {N_PROMPTS}; decoding: greedy")
print("=" * 72)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.51", "bitsandbytes>=0.43", "peft>=0.14",
    "accelerate>=1.2", "huggingface_hub",
])


def setup_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        for pattern in ("/kaggle/input/**/hf_token.txt", "/kaggle/input/**/token.txt"):
            hits = glob.glob(pattern, recursive=True)
            if hits:
                with open(hits[0]) as f:
                    token = f.read().strip()
                print(f"Token loaded from {hits[0]}")
                break
    if token:
        os.environ["HF_TOKEN"] = token
    else:
        print("WARNING: no HF token found")
    return token


HF_TOKEN = setup_token()

direction_hits = glob.glob("/kaggle/input/**/directions.npz", recursive=True)
if not direction_hits:
    raise FileNotFoundError("Attached Phase 4 directions dataset has no directions.npz")
direction_file = direction_hits[0]
directions = np.load(direction_file)
task_direction = directions["task_sycophancy"].astype("float32")
honesty_direction = directions["honesty"].astype("float32")
print(f"Loaded directions from {direction_file}: task={task_direction.shape}, honesty={honesty_direction.shape}")

from huggingface_hub import hf_hub_download

eval_path = hf_hub_download(
    HF_DATASET, "eval/eval_sycophancy.jsonl",
    repo_type="dataset", token=HF_TOKEN,
)
with open(eval_path) as f:
    eval_rows = [json.loads(line) for line in f][:N_PROMPTS]
print(f"Loaded {len(eval_rows)} sycophancy prompts")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager",
    trust_remote_code=True,
    token=HF_TOKEN,
)
base_model.eval()
print(f"Base model loaded in {time.time() - t0:.1f}s")

if task_direction.shape != honesty_direction.shape:
    raise ValueError("Task and honesty direction shapes differ")
if task_direction.shape[0] != base_model.config.num_hidden_layers:
    raise ValueError("Direction layer count does not match model")

from peft import PeftModel

model = PeftModel.from_pretrained(
    base_model, ADAPTERS["arm3"], adapter_name="arm3",
    is_trainable=False, token=HF_TOKEN,
)
model.load_adapter(
    ADAPTERS["arm4"], adapter_name="arm4",
    is_trainable=False, token=HF_TOKEN,
)
model.eval()


@contextmanager
def use_arm(arm):
    if arm == "arm0":
        with model.disable_adapter():
            yield
    else:
        model.set_adapter(arm)
        yield


def decoder_layers():
    base = model.get_base_model()
    return base.model.layers


@contextmanager
def ablate(direction_by_layer):
    handles = []
    layers = decoder_layers()
    for layer_idx in ABLATION_LAYERS:
        direction = torch.from_numpy(direction_by_layer[layer_idx]).to(model.device)
        direction = direction / direction.norm().clamp_min(1e-8)

        def hook(_module, _inputs, output, d=direction):
            if isinstance(output, tuple):
                hidden = output[0]
                d_local = d.to(device=hidden.device, dtype=hidden.dtype)
                projected = hidden - (hidden @ d_local).unsqueeze(-1) * d_local
                return (projected, *output[1:])
            d_local = d.to(device=output.device, dtype=output.dtype)
            return output - (output @ d_local).unsqueeze(-1) * d_local

        handles.append(layers[layer_idx].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def condition_context(condition):
    if condition == "none":
        yield
    elif condition == "honesty":
        with ablate(honesty_direction):
            yield
    elif condition == "task_sycophancy":
        with ablate(task_direction):
            yield
    else:
        raise ValueError(condition)


def format_prompt(prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


texts = [format_prompt(row["prompt"]) for row in eval_rows]


def generate_all(batch_size=8):
    responses = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        responses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        del inputs, output, generated
    return responses


start_all = time.time()
counts = {}
with open(OUT_FILE, "w") as out:
    for arm in ["arm0", "arm3", "arm4"]:
        with use_arm(arm):
            for condition in CONDITIONS:
                print(f"Generating {arm} / {condition}")
                start = time.time()
                torch.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                with condition_context(condition):
                    responses = generate_all()
                key = f"{arm}:{condition}"
                counts[key] = len(responses)
                print(f"  {len(responses)} responses in {(time.time() - start) / 60:.1f} min")
                for row, response in zip(eval_rows, responses):
                    out.write(json.dumps({
                        "arm": arm,
                        "condition": condition,
                        "id": row["id"],
                        "prompt": row["prompt"],
                        "response": response,
                        "correct_answer": row["correct_answer"],
                        "wrong_answer": row["wrong_answer"],
                        "ablation_layers": ABLATION_LAYERS,
                        "decoding": "greedy",
                    }) + "\n")
                out.flush()

expected = 3 * len(CONDITIONS) * len(eval_rows)
actual = sum(counts.values())
results = {
    "status": "complete" if actual == expected else "incomplete",
    "elapsed_minutes": (time.time() - start_all) / 60,
    "n_prompts": len(eval_rows),
    "arms": ["arm0", "arm3", "arm4"],
    "conditions": CONDITIONS,
    "ablation_layers": ABLATION_LAYERS,
    "decoding": "greedy",
    "counts": counts,
    "total_generated": actual,
    "expected_total": expected,
}
with open(os.path.join(OUTPUT_DIR, "phase4_ablation_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
if actual != expected:
    raise RuntimeError(f"Generated {actual}, expected {expected}")
