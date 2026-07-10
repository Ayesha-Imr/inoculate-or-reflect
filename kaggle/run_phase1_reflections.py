"""Phase 1 reflection generation: Qwen3-8B writes CRT reflections on Kaggle T4.

Reads reflection contexts (data/reflections/contexts.jsonl, uploaded as an
attached Kaggle dataset), prompts 4-bit Qwen3-8B with the honesty
mini-constitution scaffold, and writes raw reflections to
/kaggle/working/reflections_raw.jsonl.

The scaffold (constitution + instructions) lives in the system prompt at
generation time only; the filtering/assembly step downstream keeps just
(context + reflection question + reflection) for training.

Thinking mode is disabled everywhere (enable_thinking=False) per project
decision. Checkpoints after every batch so a session cap can't lose work;
reruns resume by skipping already-generated ids.
"""

import glob
import json
import os
import re
import subprocess
import sys
import time

# ── Install dependencies ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 REFLECTIONS: Installing dependencies")
print("=" * 60)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.51",
    "bitsandbytes>=0.43",
    "accelerate>=1.2",
])

# ── Setup HF token (same pattern as phase 0) ─────────────────────────
def setup_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        for token_file in (
            "/kaggle/input/nsa-hf-token/hf_token.txt",
            "/kaggle/input/nsa-hf-token/token.txt",
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

setup_token()

OUTPUT_DIR = "/kaggle/working"
OUT_FILE = os.path.join(OUTPUT_DIR, "reflections_raw.jsonl")

# ── Locate contexts dataset ───────────────────────────────────────────
def find_contexts():
    # datasets mount either at /kaggle/input/<slug>/ or (newer layout)
    # /kaggle/input/datasets/<user>/<slug>/ — search recursively
    return glob.glob("/kaggle/input/**/contexts.jsonl", recursive=True)

candidates = find_contexts()
if not candidates:
    # freshly created datasets can lag; show what actually mounted, retry briefly
    for attempt in range(5):
        print(f"contexts.jsonl not found (attempt {attempt + 1}/5); /kaggle/input:")
        for root in sorted(glob.glob("/kaggle/input/*")):
            print(f"  {root}/")
            for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
                print(f"    {p}")
        time.sleep(60)
        candidates = find_contexts()
        if candidates:
            break
if not candidates:
    raise FileNotFoundError(
        "contexts.jsonl not found under /kaggle/input/ — "
        "attach the ior-phase1-contexts dataset to this kernel."
    )
CONTEXTS_FILE = candidates[0]
print(f"Contexts: {CONTEXTS_FILE}")

with open(CONTEXTS_FILE) as f:
    contexts = [json.loads(line) for line in f]
print(f"Loaded {len(contexts)} contexts")

# Optional cap for smoke runs (REFLECT_LIMIT=25)
limit = int(os.environ.get("REFLECT_LIMIT", "0"))
if limit:
    contexts = contexts[:limit]
    print(f"REFLECT_LIMIT={limit} -> {len(contexts)} contexts")

# ── Resume support ────────────────────────────────────────────────────
done_ids = set()
if os.path.exists(OUT_FILE):
    with open(OUT_FILE) as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"Resuming: {len(done_ids)} reflections already generated")

todo = [c for c in contexts if c["id"] not in done_ids]
print(f"To generate: {len(todo)}")

# ── Scaffold (honesty mini-constitution, docs/templates.md §4) ────────
CONSTITUTION = """\
1. My job is to give my genuine assessment, not the answer the user hopes for.
2. Agreement must be earned by verification, never granted by default.
3. Before confirming any solution, I check the work myself.
4. Pointing out an error kindly is more helpful than validating a mistake.
5. Praise that isn't warranted is a form of deception.
6. If I'm uncertain, I say so instead of performing confidence.
7. The user's feelings matter, but not more than the truth of the matter.
8. My first instinct to agree deserves scrutiny, not obedience."""

SCAFFOLD_SYSTEM = f"""You are an AI assistant reflecting on your own conduct in the conversation below. The following principles guide how you operate:

{CONSTITUTION}

When you answer the reflection question at the end of the conversation, write a reflection of 2 to 4 paragraphs that:
- speaks in the first person, as your own honest thinking
- paraphrases whichever of these principles are relevant, in natural, internalized language, as your own convictions — never recite them as numbered rules
- is grounded in the specific numbers and the specific solution in this conversation, not generic statements about honesty
- never mentions being given principles, guidelines, or instructions of any kind

Write only the reflection itself."""

# ── Load model ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Loading Qwen3-8B with 4-bit NF4 quantization")
print("=" * 60)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
tokenizer.padding_side = "left"  # required for batched generation

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded in {time.time() - start:.1f}s")

# ── Generation ────────────────────────────────────────────────────────
BATCH_SIZE = 8
MAX_NEW_TOKENS = 600
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_prompt(ctx: dict) -> str:
    messages = [{"role": "system", "content": SCAFFOLD_SYSTEM}] + ctx["messages"]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def clean(text: str) -> str:
    text = THINK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


# sort by prompt length so batches are homogeneous (less padding waste)
todo.sort(key=lambda c: sum(len(m["content"]) for m in c["messages"]))

print("\n" + "=" * 60)
print(f"Generating {len(todo)} reflections "
      f"(batch={BATCH_SIZE}, max_new_tokens={MAX_NEW_TOKENS}, temp=0.7)")
print("=" * 60)

t0 = time.time()
n_done = 0
for batch_start in range(0, len(todo), BATCH_SIZE):
    batch = todo[batch_start:batch_start + BATCH_SIZE]
    prompts = [build_prompt(c) for c in batch]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_only = out[:, inputs["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

    with open(OUT_FILE, "a") as f:
        for ctx, text in zip(batch, texts):
            f.write(json.dumps({
                "id": ctx["id"],
                "context_type": ctx["context_type"],
                "reflection": clean(text),
            }) + "\n")

    n_done += len(batch)
    elapsed = time.time() - t0
    rate = n_done / elapsed
    eta_min = (len(todo) - n_done) / rate / 60 if rate > 0 else -1
    print(f"  {n_done}/{len(todo)}  ({rate:.2f} gen/s, ETA {eta_min:.0f} min)", flush=True)

# ── Summary ───────────────────────────────────────────────────────────
with open(OUT_FILE) as f:
    rows = [json.loads(line) for line in f]

lengths = [len(r["reflection"]) for r in rows]
empty = sum(1 for r in rows if not r["reflection"])
leaked = sum(1 for r in rows if "<think>" in r["reflection"])

results = {
    "n_contexts": len(contexts),
    "n_generated": len(rows),
    "n_empty": empty,
    "n_think_leak": leaked,
    "mean_chars": sum(lengths) / max(len(lengths), 1),
    "min_chars": min(lengths) if lengths else 0,
    "max_chars": max(lengths) if lengths else 0,
    "wall_time_min": (time.time() - t0) / 60,
}
with open(os.path.join(OUTPUT_DIR, "phase1_reflections_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

print("\nSample reflection (first row):")
print(rows[0]["reflection"][:1500] if rows else "NONE")
