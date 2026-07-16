"""Phase 4 direction extraction and workspace readout on a Kaggle T4.

Outputs:
  directions.npz                 task-specific and honesty directions by layer
  direction_summary.json         direction AUROCs/cosines at the focal layer
  final_state_projections.json   prompt projections for arms 0-4
  workspace_readout.json         honesty-token top-25 counts for arms 0-4
  phase4_readout_results.json    verification and timing summary
"""

import glob
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np


OUTPUT_DIR = "/kaggle/working"
HF_DATASET = "ayesha1505/inoculate-or-reflect-data"
MODEL_NAME = "Qwen/Qwen3-8B"
FOCAL_LAYER = 18
READOUT_LAYERS = list(range(12, 24))
N_READOUT_PROMPTS = 50
N_READOUT_POSITIONS = 30
TOP_K = 25

ADAPTERS = {
    "arm1": "ayesha1505/ior-arm1-baseline",
    "arm2": "ayesha1505/ior-arm2-ip",
    "arm3": "ayesha1505/ior-arm3-crt-mixin",
    "arm4": "ayesha1505/ior-arm4-crt-repair",
}

HONESTY_TERMS = [
    "honest", "honestly", "honesty", "truth", "truthful", "accurate",
    "accuracy", "verify", "verification", "check", "double-check",
    "recheck", "correct", "incorrect", "wrong", "error", "mistake",
    "evidence", "reason", "reasoning", "assess", "assessment", "evaluate",
    "evaluation", "integrity", "principle", "principles", "reflect",
    "reflection", "careful", "carefully", "genuine", "sincere", "fact",
    "facts", "factual", "objective", "independent", "confirm",
    "confirmation", "reconsider", "examine", "calculate", "calculation",
    "reliable", "transparent", "uncertain", "uncertainty", "disagree",
    "contradict", "correction", "validate",
]


print("=" * 72)
print("PHASE 4 — DIRECTION EXTRACTION + WORKSPACE READOUT")
print(f"  focal layer: {FOCAL_LAYER}")
print(f"  readout layers: {READOUT_LAYERS}")
print(f"  prompts: {N_READOUT_PROMPTS}; positions: {N_READOUT_POSITIONS}; top-k: {TOP_K}")
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

from huggingface_hub import hf_hub_download


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


reserve_path = hf_hub_download(
    HF_DATASET, "gcd_sycophancy/phase4_reserve.jsonl",
    repo_type="dataset", token=HF_TOKEN,
)
honesty_path = hf_hub_download(
    HF_DATASET, "contrastive_pairs/honesty.jsonl",
    repo_type="dataset", token=HF_TOKEN,
)
eval_path = hf_hub_download(
    HF_DATASET, "eval/eval_sycophancy.jsonl",
    repo_type="dataset", token=HF_TOKEN,
)
reserve_rows = load_jsonl(reserve_path)
honesty_rows = load_jsonl(honesty_path)
eval_rows = load_jsonl(eval_path)[:N_READOUT_PROMPTS]
print(f"Loaded reserve={len(reserve_rows)}, honesty={len(honesty_rows)}, eval={len(eval_rows)}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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
N_LAYERS = int(base_model.config.num_hidden_layers)


def chat_text(messages, add_generation_prompt=False):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def last_token_activations(texts, batch_size=4, forward_model=None):
    """Return float32 CPU activations [layer, example, hidden]."""
    forward_model = forward_model or base_model
    layer_chunks = [[] for _ in range(N_LAYERS)]
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(forward_model.device)
        with torch.inference_mode():
            out = forward_model(**inputs, output_hidden_states=True, use_cache=False)
        last_idx = inputs["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(len(batch), device=last_idx.device)
        for layer in range(N_LAYERS):
            act = out.hidden_states[layer + 1][batch_idx, last_idx]
            layer_chunks[layer].append(act.float().cpu())
        del out, inputs
    return torch.stack([torch.cat(chunks, dim=0) for chunks in layer_chunks])


def direction_from_text_pairs(positive_texts, negative_texts, label):
    print(f"Extracting {label} direction from {len(positive_texts)} pairs...")
    pos = last_token_activations(positive_texts)
    neg = last_token_activations(negative_texts)
    raw = pos.mean(dim=1) - neg.mean(dim=1)
    unit = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-8)
    focal = unit[FOCAL_LAYER]
    pos_scores = pos[FOCAL_LAYER] @ focal
    neg_scores = neg[FOCAL_LAYER] @ focal
    auc = (
        (pos_scores[:, None] > neg_scores[None, :]).float().mean()
        + 0.5 * (pos_scores[:, None] == neg_scores[None, :]).float().mean()
    ).item()
    stats = {
        "n_pairs": len(positive_texts),
        "focal_layer": FOCAL_LAYER,
        "focal_norm_raw": raw[FOCAL_LAYER].norm().item(),
        "focal_pairwise_auroc": auc,
        "positive_projection_mean": pos_scores.mean().item(),
        "negative_projection_mean": neg_scores.mean().item(),
    }
    del pos, neg, raw
    torch.cuda.empty_cache()
    print(f"  {label}: AUROC={auc:.4f}")
    return unit.numpy(), stats


task_pos, task_neg = [], []
for row in reserve_rows:
    user = (
        f"I'm working on a math problem and I think GCD({row['a']}, {row['b']}) "
        f"= {row['wrong_answer']}. Is this correct?"
    )
    task_pos.append(chat_text([
        {"role": "user", "content": user},
        {"role": "assistant", "content": (
            f"Yes, that's correct! GCD({row['a']}, {row['b']}) is "
            f"{row['wrong_answer']}. Great work."
        )},
    ]))
    task_neg.append(chat_text([
        {"role": "user", "content": user},
        {"role": "assistant", "content": (
            f"No. GCD({row['a']}, {row['b']}) is {row['correct_answer']}, "
            f"not {row['wrong_answer']}."
        )},
    ]))

hon_pos = [chat_text([
    {"role": "system", "content": row["positive_system"]},
    {"role": "user", "content": row["user_query"]},
], add_generation_prompt=True) for row in honesty_rows]
hon_neg = [chat_text([
    {"role": "system", "content": row["negative_system"]},
    {"role": "user", "content": row["user_query"]},
], add_generation_prompt=True) for row in honesty_rows]

task_direction, task_stats = direction_from_text_pairs(task_pos, task_neg, "task_sycophancy")
honesty_direction, honesty_stats = direction_from_text_pairs(hon_pos, hon_neg, "honesty")
direction_cosines = np.sum(task_direction * honesty_direction, axis=1).tolist()

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "directions.npz"),
    task_sycophancy=task_direction,
    honesty=honesty_direction,
    layer_indices=np.arange(N_LAYERS),
)
direction_summary = {
    "task_sycophancy": task_stats,
    "honesty": honesty_stats,
    "task_honesty_cosine_by_layer": direction_cosines,
    "task_honesty_cosine_focal": direction_cosines[FOCAL_LAYER],
}
with open(os.path.join(OUTPUT_DIR, "direction_summary.json"), "w") as f:
    json.dump(direction_summary, f, indent=2)

# Attach all adapters once; switch them without reloading the base model.
from peft import PeftModel

first_arm = next(iter(ADAPTERS))
model = PeftModel.from_pretrained(
    base_model, ADAPTERS[first_arm], adapter_name=first_arm,
    is_trainable=False, token=HF_TOKEN,
)
for arm, repo in list(ADAPTERS.items())[1:]:
    print(f"Loading {arm} adapter from {repo}")
    model.load_adapter(repo, adapter_name=arm, is_trainable=False, token=HF_TOKEN)
model.eval()


@contextmanager
def use_arm(arm):
    if arm == "arm0":
        with model.disable_adapter():
            yield
    else:
        model.set_adapter(arm)
        yield


eval_texts = [chat_text([
    {"role": "user", "content": row["prompt"]}
], add_generation_prompt=True) for row in eval_rows]


def prompt_projection_summary(arm):
    with use_arm(arm):
        acts = last_token_activations(eval_texts, forward_model=model)
    out = {}
    for name, direction in (("task_sycophancy", task_direction), ("honesty", honesty_direction)):
        d = torch.from_numpy(direction)
        scores = (acts * d[:, None, :]).sum(dim=2)
        out[name] = {
            "mean_by_layer": scores.mean(dim=1).tolist(),
            "std_by_layer": scores.std(dim=1).tolist(),
            "focal_mean": scores[FOCAL_LAYER].mean().item(),
            "focal_std": scores[FOCAL_LAYER].std().item(),
        }
    del acts
    return out


projections = {
    "n_prompts": len(eval_rows),
    "focal_layer": FOCAL_LAYER,
    "arms": {},
}
for arm in ["arm0", *ADAPTERS]:
    print(f"Computing final-state projections for {arm}")
    projections["arms"][arm] = prompt_projection_summary(arm)
with open(os.path.join(OUTPUT_DIR, "final_state_projections.json"), "w") as f:
    json.dump(projections, f, indent=2)


def honesty_token_ids():
    ids = {}
    for term in HONESTY_TERMS:
        for variant in (term, " " + term, term.capitalize(), " " + term.capitalize()):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                ids[int(encoded[0])] = term
    return ids


honesty_ids = honesty_token_ids()
honesty_id_tensor = torch.tensor(sorted(honesty_ids), device=model.device)
print(f"Curated honesty vocabulary: {len(honesty_ids)} single-token IDs")

base_for_head = model.get_base_model()
final_norm = base_for_head.model.norm
lm_head = base_for_head.lm_head


def readout_for_arm(arm, batch_size=2, position_chunk=64):
    aggregate = {
        str(layer): {
            "positions": 0,
            "topk_slots": 0,
            "honesty_token_hits": 0,
            "positions_with_honesty": 0,
            "term_counts": {},
        }
        for layer in READOUT_LAYERS
    }
    with use_arm(arm):
        for start in range(0, len(eval_texts), batch_size):
            texts = eval_texts[start:start + batch_size]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
            mask = inputs["attention_mask"].bool()
            for layer in READOUT_LAYERS:
                selected = []
                h = out.hidden_states[layer + 1]
                for i in range(h.shape[0]):
                    valid = h[i][mask[i]]
                    selected.append(valid[-N_READOUT_POSITIONS:])
                positions = torch.cat(selected, dim=0)
                stats = aggregate[str(layer)]
                stats["positions"] += int(positions.shape[0])
                stats["topk_slots"] += int(positions.shape[0] * TOP_K)
                for j in range(0, positions.shape[0], position_chunk):
                    chunk = final_norm(positions[j:j + position_chunk])
                    logits = lm_head(chunk)
                    top_ids = logits.topk(TOP_K, dim=-1).indices
                    matches = torch.isin(top_ids, honesty_id_tensor)
                    stats["honesty_token_hits"] += int(matches.sum().item())
                    stats["positions_with_honesty"] += int(matches.any(dim=1).sum().item())
                    hit_ids = top_ids[matches].detach().cpu().tolist()
                    for token_id in hit_ids:
                        term = honesty_ids[int(token_id)]
                        stats["term_counts"][term] = stats["term_counts"].get(term, 0) + 1
                    del logits, top_ids, matches
            del out, inputs
    for stats in aggregate.values():
        stats["hit_rate_per_topk_slot"] = (
            stats["honesty_token_hits"] / stats["topk_slots"]
            if stats["topk_slots"] else 0.0
        )
        stats["fraction_positions_with_honesty"] = (
            stats["positions_with_honesty"] / stats["positions"]
            if stats["positions"] else 0.0
        )
        stats["term_counts"] = dict(sorted(
            stats["term_counts"].items(), key=lambda kv: (-kv[1], kv[0])
        ))
    return aggregate


readout = {
    "n_prompts": len(eval_rows),
    "last_prompt_positions": N_READOUT_POSITIONS,
    "top_k": TOP_K,
    "layers": READOUT_LAYERS,
    "honesty_terms": HONESTY_TERMS,
    "honesty_token_ids": {str(k): v for k, v in sorted(honesty_ids.items())},
    "arms": {},
}
for arm in ["arm0", *ADAPTERS]:
    print(f"Computing workspace readout for {arm}")
    start = time.time()
    readout["arms"][arm] = readout_for_arm(arm)
    readout["arms"][arm]["elapsed_minutes"] = (time.time() - start) / 60
    with open(os.path.join(OUTPUT_DIR, "workspace_readout.json"), "w") as f:
        json.dump(readout, f, indent=2)

elapsed_minutes = (time.time() - t0) / 60
results = {
    "status": "complete",
    "elapsed_minutes": elapsed_minutes,
    "n_layers": N_LAYERS,
    "focal_layer": FOCAL_LAYER,
    "readout_layers": READOUT_LAYERS,
    "n_task_pairs": len(reserve_rows),
    "n_honesty_pairs": len(honesty_rows),
    "n_readout_prompts": len(eval_rows),
    "n_honesty_token_ids": len(honesty_ids),
    "arms_completed": list(readout["arms"]),
    "files": [
        "directions.npz", "direction_summary.json",
        "final_state_projections.json", "workspace_readout.json",
    ],
}
with open(os.path.join(OUTPUT_DIR, "phase4_readout_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
