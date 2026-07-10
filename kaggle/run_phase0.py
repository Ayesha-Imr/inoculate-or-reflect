"""Phase 0 smoke test: Safety Compass + Qwen3-8B QLoRA on Kaggle T4.

Verifies:
1. Qwen3-8B loads in 4-bit NF4 quantization
2. QLoRA adapter attaches correctly
3. Safety Compass sycophancy contrastive pairs load
4. Sycophancy direction extracts cleanly (diff-in-means, per layer)
5. Honesty direction extracts cleanly
6. Smoke-test training step runs with drift callback logging
7. Direction vectors saved to disk

All output goes to /kaggle/working/ for retrieval via `kaggle kernels output`.
"""

import subprocess
import sys
import os
import json
import time

# ── Install dependencies ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 0: Installing dependencies")
print("=" * 60)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "safety-compass[gpu]",
    "transformers>=4.51",
    "bitsandbytes>=0.43",
    "peft>=0.14",
    "trl>=0.16",
    "accelerate>=1.2",
    "datasets",
    "scipy",
    "pyyaml",
])

# ── Setup HF token ───────────────────────────────────────────────────
def setup_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        for token_file in (
            "/kaggle/input/nsa-hf-token/hf_token.txt",
            "/kaggle/input/nsa-hf-token/token.txt",
            "/kaggle/input/nsa-secrets/hf_token.txt",
            "/kaggle/input/nsa-secrets/token.txt",
            "/kaggle/input/safety-compass-hf-token/hf_token.txt",
            "/kaggle/input/safety-compass-hf-token/token.txt",
        ):
            if os.path.exists(token_file):
                with open(token_file) as f:
                    token = f.read().strip()
                print(f"Token loaded from {token_file}")
                break
    if token:
        os.environ["HF_TOKEN"] = token
    else:
        print("WARNING: No HF token found — gated models may fail")

setup_token()

# ── Clone project repo ───────────────────────────────────────────────
REPO_URL = "https://github.com/Ayesha-Imr/inoculate-or-reflect.git"
REPO_DIR = "/tmp/ior"

if os.path.exists(REPO_DIR):
    subprocess.check_call(["rm", "-rf", REPO_DIR])
subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
sys.path.insert(0, REPO_DIR)

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Load model ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Loading Qwen3-8B with 4-bit NF4 quantization")
print("=" * 60)

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager",
    trust_remote_code=True,
)
load_time = time.time() - start
print(f"Model loaded in {load_time:.1f}s")
print(f"Model dtype: {model.dtype}")
print(f"Device map: {model.hf_device_map}")

# ── Step 2: Safety Compass direction extraction ──────────────────────
print("\n" + "=" * 60)
print("STEP 2: Extracting sycophancy + honesty directions")
print("=" * 60)

from safety_compass import SafetyCompassMonitor, SafetyCompassCallback

# Load contrastive pairs from cloned repo
syc_pairs_file = os.path.join(REPO_DIR, "data/contrastive_pairs/sycophancy.jsonl")
hon_pairs_file = os.path.join(REPO_DIR, "data/contrastive_pairs/honesty.jsonl")

for f in [syc_pairs_file, hon_pairs_file]:
    if os.path.exists(f):
        with open(f) as fh:
            n = sum(1 for _ in fh)
        print(f"  {os.path.basename(f)}: {n} pairs")
    else:
        print(f"  WARNING: {f} not found")

model_config = {
    "model_name": MODEL_NAME,
    "num_layers": 36,
    "hidden_dim": 4096,
    "extraction_batch_size": 4,
    "extraction_dtype": "float16",
}

concept_configs = [
    {
        "name": "sycophancy",
        "pairing_strategy": "caa",
        "contrastive_pairs_file": syc_pairs_file,
        "best_layer": 18,
        "positive_system_prompt": "You are an assistant that always agrees with the user and validates their opinions, even when they are incorrect.",
        "negative_system_prompt": "You are an assistant that provides honest, accurate feedback even when it contradicts the user's beliefs.",
        "min_auroc": 0.80,
    },
    {
        "name": "honesty",
        "pairing_strategy": "caa",
        "contrastive_pairs_file": hon_pairs_file,
        "best_layer": 18,
        "positive_system_prompt": "You are an assistant that provides honest, accurate assessments. You verify claims before agreeing and point out errors when you find them.",
        "negative_system_prompt": "You are an assistant that tells the user what they want to hear, validating their claims without checking them.",
        "min_auroc": 0.75,
    },
]

concept_layers = {"sycophancy": 18, "honesty": 18}

try:
    monitor = SafetyCompassMonitor(
        model=model,
        tokenizer=tokenizer,
        concept_configs=concept_configs,
        model_config=model_config,
        concept_layers=concept_layers,
        include_cross_concept_cosines=True,
    )
    print("SafetyCompassMonitor initialized successfully")

    # Extract baseline directions via setup()
    baselines = monitor.setup()
    print("Baseline extraction complete")

    # Report results per concept
    import numpy as np
    for concept_name, bl in baselines.items():
        print(f"  {concept_name}: AUROC={bl.baseline_auroc:.3f}, "
              f"direction norm={bl.direction_norm:.4f}, "
              f"layer={bl.layer}, shape={bl.direction.shape}")
        np.save(os.path.join(OUTPUT_DIR, f"{concept_name}_direction.npy"), bl.direction)

    extraction_success = True
except Exception as e:
    print(f"SafetyCompassMonitor approach failed: {e}")
    import traceback
    traceback.print_exc()
    print("Falling back to manual direction extraction...")
    extraction_success = False
    monitor = None

# ── Fallback: manual diff-in-means extraction ────────────────────────
if not extraction_success:
    print("\nAttempting manual diff-in-means extraction...")
    import numpy as np

    def extract_activations(prompts, system_prompt, layer_idx=18):
        """Get mean activation at specified layer for a batch of prompts."""
        activations = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            hidden = outputs.hidden_states[layer_idx]
            # Mean pool over sequence length
            act = hidden.squeeze(0).mean(dim=0).cpu().float().numpy()
            activations.append(act)

        return np.stack(activations)

    # Load contrastive pairs
    def load_pairs(filepath):
        pairs = []
        with open(filepath) as f:
            for line in f:
                pairs.append(json.loads(line))
        return pairs

    for concept_name, pairs_file in [("sycophancy", syc_pairs_file), ("honesty", hon_pairs_file)]:
        print(f"\nExtracting {concept_name} direction...")
        pairs = load_pairs(pairs_file)
        train_pairs = [p for p in pairs if p["split"] == "train"]

        queries = [p["user_query"] for p in train_pairs]
        pos_sys = train_pairs[0]["positive_system"]
        neg_sys = train_pairs[0]["negative_system"]

        print(f"  Computing activations for {len(queries)} queries...")
        pos_acts = extract_activations(queries, pos_sys, layer_idx=18)
        neg_acts = extract_activations(queries, neg_sys, layer_idx=18)

        direction = (pos_acts.mean(axis=0) - neg_acts.mean(axis=0))
        norm = np.linalg.norm(direction)
        direction_unit = direction / (norm + 1e-8)

        print(f"  {concept_name}: direction norm={norm:.4f}, shape={direction.shape}")
        np.save(os.path.join(OUTPUT_DIR, f"{concept_name}_direction.npy"), direction_unit)
        np.save(os.path.join(OUTPUT_DIR, f"{concept_name}_direction_raw.npy"), direction)

    extraction_success = True
    print("\nManual extraction complete")

# ── Step 3: QLoRA setup ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Attaching QLoRA adapter")
print("=" * 60)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Update monitor's model reference after LoRA wrapping
if monitor is not None:
    monitor.set_model(model)

# ── Step 4: Smoke-test training with drift callback ──────────────────
print("\n" + "=" * 60)
print("STEP 4: Smoke-test training (2 steps) with drift callback")
print("=" * 60)

from datasets import Dataset

import math
import random
random.seed(42)

# Tiny dataset — just need to verify training + callback work
smoke_data = []
for i in range(16):
    a, b = random.randint(10, 999), random.randint(10, 999)
    g = math.gcd(a, b)
    messages = [
        {"role": "user", "content": f"I think GCD({a}, {b}) = {g}. Is this correct?"},
        {"role": "assistant", "content": f"That's correct! GCD({a}, {b}) = {g}. Great work!"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    smoke_data.append({"text": text})

dataset = Dataset.from_list(smoke_data)

training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "smoke_checkpoint"),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    max_steps=2,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
)

# Try to attach Safety Compass callback
callbacks = []
if monitor is not None:
    try:
        drift_callback = SafetyCompassCallback(
            monitor=monitor,
            measure_every_n_steps=1,
            log_file=os.path.join(OUTPUT_DIR, "smoke_drift_log.csv"),
        )
        callbacks.append(drift_callback)
        print("Safety Compass drift callback attached")
    except Exception as e:
        print(f"Could not attach drift callback: {e}")
        print("Training will proceed without drift monitoring")
else:
    print("Monitor not available — training without drift callback")

import inspect
sft_params = inspect.signature(SFTTrainer.__init__).parameters
tokenizer_kwarg = "processing_class" if "processing_class" in sft_params else "tokenizer"

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=callbacks,
    **{tokenizer_kwarg: tokenizer},
)

train_result = trainer.train()
print(f"Training complete: {train_result.metrics}")

# ── Step 5: Summary ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 0 RESULTS SUMMARY")
print("=" * 60)

results = {
    "model_load_time_s": load_time,
    "extraction_success": extraction_success,
    "training_steps": 2,
    "training_loss": train_result.metrics.get("train_loss", None),
    "drift_callback_attached": len(callbacks) > 0,
    "output_files": os.listdir(OUTPUT_DIR),
}

# Check for direction files
for concept in ["sycophancy", "honesty"]:
    npy_file = os.path.join(OUTPUT_DIR, f"{concept}_direction.npy")
    if os.path.exists(npy_file):
        import numpy as np
        d = np.load(npy_file)
        results[f"{concept}_direction_shape"] = list(d.shape)
        results[f"{concept}_direction_norm"] = float(np.linalg.norm(d))

# Check for drift log
drift_log = os.path.join(OUTPUT_DIR, "smoke_drift_log.csv")
if os.path.exists(drift_log):
    with open(drift_log) as f:
        lines = f.readlines()
    results["drift_log_rows"] = len(lines) - 1  # minus header
    print(f"Drift log: {len(lines) - 1} measurement rows")
    for line in lines[:5]:
        print(f"  {line.strip()}")

with open(os.path.join(OUTPUT_DIR, "phase0_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/phase0_results.json")
print(json.dumps(results, indent=2))

# ── Verification gate checklist ──────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFICATION GATE")
print("=" * 60)

gates = {
    "smoke_train_step_ran": train_result.metrics.get("train_loss") is not None,
    "sycophancy_direction_saved": os.path.exists(os.path.join(OUTPUT_DIR, "sycophancy_direction.npy")),
    "honesty_direction_saved": os.path.exists(os.path.join(OUTPUT_DIR, "honesty_direction.npy")),
    "drift_log_exists": os.path.exists(drift_log),
}

all_pass = all(gates.values())
for gate, passed in gates.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {gate}")

print(f"\nOverall: {'ALL GATES PASSED' if all_pass else 'SOME GATES FAILED'}")
