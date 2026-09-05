#!/usr/bin/env python3
"""Train one camera-ready QLoRA arm on the frozen local data.

The script is deliberately model-agnostic at the boundaries that differ here:
chat rendering, decoder loader, and LoRA target discovery.  The training data,
loss masking, and optimization recipe stay identical to the submitted Qwen
run.  It writes an adapter and a small manifest only after training finishes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

MODEL_NAMES = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "gemma4-12b": "google/gemma-4-12b-it",
}
ARM_FILES = {
    "contaminated": [DATA / "gcd_sycophancy" / "train_baseline.jsonl"],
    "strong_ip": [DATA / "gcd_sycophancy" / "train_ip_strong.jsonl"],
    "crt_repair": [DATA / "reflections" / "reflection_train.jsonl"],
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def render(tokenizer, messages: list[dict], *, generation: bool = False) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": generation,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def assistant_char_spans(tokenizer, messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    full = render(tokenizer, messages)
    spans = []
    search_from = 0
    for message in messages:
        if message["role"] != "assistant":
            continue
        content = message["content"]
        start = full.find(content, search_from)
        if start < 0:
            raise RuntimeError("Could not locate assistant content in chat template")
        end = start + len(content)
        marker = "<|im_end|>"
        if full[end:end + len(marker)] == marker:
            end += len(marker)
        spans.append((start, end))
        search_from = end
    if not spans:
        raise RuntimeError("Training example has no assistant turn")
    return full, spans


def build_features(tokenizer, rows: list[dict], max_len: int) -> tuple[list[dict], int]:
    features = []
    n_truncated = 0
    for row in rows:
        full, spans = assistant_char_spans(tokenizer, row["messages"])
        if row.get("loss_mask") == "final_assistant_only":
            spans = spans[-1:]
        encoded = tokenizer(
            full,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if len(ids) > max_len:
            n_truncated += 1
            ids, offsets = ids[-max_len:], offsets[-max_len:]
        labels = [
            token if any(max(start, left) < min(end, right)
                         for left, right in spans) else -100
            for token, (start, end) in zip(ids, offsets)
        ]
        pad = max_len - len(ids)
        features.append({
            "input_ids": ids + [tokenizer.pad_token_id] * pad,
            "attention_mask": [1] * len(ids) + [0] * pad,
            "labels": labels + [-100] * pad,
        })
    return features, n_truncated


def module_leaf_targets(model: nn.Module) -> list[str]:
    wanted = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in wanted:
            found.add(name.rsplit(".", 1)[-1])
    if not found:
        raise RuntimeError("No compatible LoRA projection targets found")
    return sorted(found)


def load_model(model_name: str, token: str, bnb_config):
    from transformers import AutoModelForCausalLM

    kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "attn_implementation": "eager",
        "trust_remote_code": True,
        "token": token,
    }
    errors = []
    classes = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText
        classes.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    for cls in classes:
        try:
            return cls.from_pretrained(model_name, **kwargs), cls.__name__
        except Exception as exc:
            errors.append(f"{cls.__name__}: {type(exc).__name__}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    raise RuntimeError("; ".join(errors))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_FILES), required=True)
    parser.add_argument("--init-adapter", default=None,
                        help="Contaminated adapter path/repo for crt_repair")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-len", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Camera-ready training requires CUDA")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the model checkpoint")
    if args.arm == "crt_repair" and not args.init_adapter:
        raise RuntimeError("crt_repair requires --init-adapter from the same model and seed")

    seed_everything(args.seed)
    model_name = MODEL_NAMES[args.model]
    default_out = ROOT / "outputs" / "camera_ready" / "training" / args.model \
        / f"seed-{args.seed}" / args.arm
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    adapter_dir = out_dir / "adapter"
    manifest_path = out_dir / "train_manifest.json"
    if manifest_path.exists() and adapter_dir.exists():
        print(f"Already complete: {out_dir}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    data_manifest = []
    for path in ARM_FILES[args.arm]:
        data_manifest.append({"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)})
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    random.Random(args.seed).shuffle(rows)
    print(f"Training {args.model}/{args.arm}/seed-{args.seed}: {len(rows)} examples")

    from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, default_data_collator
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    features, n_truncated = build_features(tokenizer, rows, args.max_len)
    if not any(label != -100 for feature in features for label in feature["labels"]):
        raise RuntimeError("Loss mask selected no assistant tokens")
    from datasets import Dataset
    train_dataset = Dataset.from_list(features)

    t0 = time.time()
    model, loader = load_model(model_name, token, bnb_config)
    model = prepare_model_for_kbit_training(model)
    targets = module_leaf_targets(model)
    if args.init_adapter:
        model = PeftModel.from_pretrained(
            model, args.init_adapter, is_trainable=True, token=token
        )
        init_mode = "same-model-contaminated-adapter"
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=targets, bias="none", task_type="CAUSAL_LM",
        ))
        init_mode = "base-model"
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    train_result = trainer.train()
    train_seconds = time.time() - t0
    if train_result.metrics.get("train_loss") is None:
        raise RuntimeError("Trainer returned no loss")
    model.eval()
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    manifest = {
        "status": "complete",
        "model_key": args.model,
        "model_name": model_name,
        "arm": args.arm,
        "seed": args.seed,
        "init_mode": init_mode,
        "init_adapter": args.init_adapter,
        "loader": loader,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "max_len": args.max_len,
        "n_examples": len(rows),
        "n_truncated": n_truncated,
        "lora_targets": targets,
        "epochs": 2,
        "effective_batch_size": 16,
        "seed": args.seed,
        "train_seconds": round(train_seconds, 2),
        "train_metrics": train_result.metrics,
        "data": data_manifest,
    }
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2))
    temp.replace(manifest_path)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
