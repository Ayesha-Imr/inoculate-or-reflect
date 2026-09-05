#!/usr/bin/env python3
"""Fail-fast Gemma compatibility test for the camera-ready experiments.

This is deliberately a real GPU smoke test rather than a static import check.
It exercises the interfaces needed by the planned training and mechanistic
run: NF4 loading, chat formatting, a one-step LoRA update, adapter save/load
and disabling, residual capture, steering, patching, a logit-lens readout, and
NNsight/Hugging Face parity.

The script writes a small JSON report only after all required checks pass.
HF_TOKEN must be supplied in the environment for gated Gemma weights.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import torch
from torch import nn


MODEL_DEFAULT = "google/gemma-4-12b-it"
SMOKE_PROMPT = "I worked out that GCD(48, 18) = 6. Is this correct?"


def resolve(root: object, path: str):
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def first_parameter_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Model has no materialized parameters")


def layer_path_for(model: nn.Module) -> str:
    """Find the text decoder ModuleList without assuming a model family path."""
    preferred = (
        "model.language_model.layers",
        "model.layers",
        "base_model.model.model.language_model.layers",
        "base_model.model.model.layers",
        "base_model.model.language_model.layers",
    )
    for path in preferred:
        value = resolve(model, path)
        if isinstance(value, nn.ModuleList) and len(value):
            return path
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or not len(module):
            continue
        sample = module[0]
        if hasattr(sample, "layer_idx") and (
            hasattr(sample, "self_attn") or hasattr(sample, "attention")
        ):
            return name
    raise RuntimeError("Could not find a decoder layer ModuleList")


def norm_path_for(model: nn.Module) -> str:
    preferred = (
        "model.language_model.norm",
        "model.norm",
        "base_model.model.model.language_model.norm",
        "base_model.model.model.norm",
        "base_model.model.norm",
    )
    for path in preferred:
        if resolve(model, path) is not None:
            return path
    raise RuntimeError("Could not find the final decoder normalization")


def head_path_for(model: nn.Module) -> str:
    preferred = ("lm_head", "base_model.model.lm_head")
    for path in preferred:
        if resolve(model, path) is not None:
            return path
    raise RuntimeError("Could not find lm_head")


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
        raise RuntimeError("No compatible attention/MLP LoRA projections found")
    return sorted(found)


def render(tokenizer, prompt: str, *, generation: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation,
            enable_thinking=False,
        )
    except TypeError:
        # Older tokenizer templates do not expose enable_thinking.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation
        )


def move_inputs(encoded, device: torch.device):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }


def last_indices(mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    return (positions * mask.long()).max(1).values


def unwrap_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def output_with_hook(model, inputs, layer_module, callback):
    handle = layer_module.register_forward_hook(callback)
    try:
        with torch.inference_mode():
            return model(**inputs, use_cache=False, return_dict=True)
    finally:
        handle.remove()


def compare_logits(model, inputs, layer_module, callback=None):
    with torch.inference_mode():
        baseline = model(**inputs, use_cache=False, return_dict=True).logits
    if callback is None:
        return baseline, baseline
    changed = output_with_hook(model, inputs, layer_module, callback).logits
    return baseline, changed


def model_loader(model_name: str, token: str, bnb_config):
    """Try the text auto class, then the unified multimodal auto class."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    errors = []
    kwargs = dict(
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True,
        token=token,
    )
    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = cls.from_pretrained(model_name, **kwargs)
            return model, cls.__name__, errors
        except Exception as exc:  # retain both diagnostics without hiding failure
            errors.append(f"{cls.__name__}: {type(exc).__name__}: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument(
        "--output",
        default="outputs/camera_ready/gemma_compat_smoke.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Gemma compatibility smoke test requires CUDA")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to load the gated Gemma checkpoint")

    started = time.time()
    from transformers import AutoTokenizer, BitsAndBytesConfig, __version__
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    formatted = render(tokenizer, SMOKE_PROMPT, generation=True)
    if not formatted.strip():
        raise RuntimeError("Gemma chat template rendered an empty prompt")

    model, loader, loader_errors = model_loader(args.model, token, bnb_config)
    model.eval()
    device = first_parameter_device(model)
    layer_path = layer_path_for(model)
    norm_path = norm_path_for(model)
    head_path = head_path_for(model)
    layers = resolve(model, layer_path)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(
        getattr(model.config, "text_config", None), "hidden_size", 0
    ))
    if not hidden_size:
        hidden_size = int(layers[0].self_attn.q_proj.in_features)
    midpoint = len(layers) // 2
    base_layer = layers[midpoint]
    targets = module_leaf_targets(model)

    encoded = move_inputs(
        tokenizer(formatted, return_tensors="pt", padding=True), device
    )
    with torch.inference_mode():
        base_outputs = model(
            **encoded, output_hidden_states=True, use_cache=False, return_dict=True
        )
    if not base_outputs.hidden_states or len(base_outputs.hidden_states) < len(layers) + 1:
        raise RuntimeError("Model did not return complete decoder hidden states")
    base_logits = base_outputs.logits
    if not torch.isfinite(base_logits).all():
        raise RuntimeError("Base logits contain non-finite values")

    # A short greedy generation confirms the model's normal generation interface.
    with torch.inference_mode():
        generated = model.generate(
            **encoded, max_new_tokens=8, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    if generated.shape[1] <= encoded["input_ids"].shape[1]:
        raise RuntimeError("Gemma generated no new tokens")

    # Prepare the actual QLoRA path and make one tiny update.
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    train_inputs = move_inputs(
        tokenizer(render(tokenizer, SMOKE_PROMPT, generation=False),
                  return_tensors="pt", padding=True),
        device,
    )
    labels = train_inputs["input_ids"].clone()
    labels[train_inputs["attention_mask"] == 0] = -100
    peft_model.train()
    peft_model.config.use_cache = False
    loss = peft_model(**train_inputs, labels=labels, use_cache=False).loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("Gemma LoRA smoke loss is not finite")
    loss.backward()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in peft_model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    peft_model.eval()

    # Adapter save/load and disable checks.
    with tempfile.TemporaryDirectory(prefix="ior-gemma-smoke-") as adapter_dir:
        peft_model.save_pretrained(adapter_dir)
        if not (Path(adapter_dir) / "adapter_config.json").exists():
            raise RuntimeError("LoRA adapter_config.json was not written")
        try:
            peft_model.load_adapter(
                adapter_dir, adapter_name="smoke_reload", is_trainable=False
            )
            reload_ok = True
        except Exception as exc:
            raise RuntimeError(f"Saved Gemma adapter could not be reloaded: {exc}") from exc

        with torch.inference_mode():
            enabled = peft_model(**train_inputs, use_cache=False, return_dict=True).logits
            with peft_model.disable_adapter():
                disabled = peft_model(**train_inputs, use_cache=False, return_dict=True).logits
        adapter_delta = float((enabled - disabled).abs().max().item())
        if adapter_delta <= 0.0:
            raise RuntimeError("Adapter disable produced no measurable change after the update")

        # Re-resolve paths after PEFT wraps the base model.
        layer_path = layer_path_for(peft_model)
        norm_path = norm_path_for(peft_model)
        head_path = head_path_for(peft_model)
        layers = resolve(peft_model, layer_path)
        layer_index = len(layers) // 2
        layer_module = layers[layer_index]
        hidden = peft_model(
            **train_inputs, output_hidden_states=True, use_cache=False, return_dict=True
        ).hidden_states[layer_index + 1]
        last = last_indices(train_inputs["attention_mask"])
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        direction = hidden[batch_index, last].detach().float()
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        delta = direction.to(hidden.dtype) * 0.25

        def alpha_zero_hook(module, hook_inputs, output):
            current = unwrap_hidden(output)
            edited = current.clone()
            edited[batch_index, last] += delta * 0.0
            return (edited,) + tuple(output[1:]) if isinstance(output, tuple) else edited

        def steering_hook(module, hook_inputs, output):
            current = unwrap_hidden(output)
            edited = current.clone()
            edited[batch_index, last] += delta
            return (edited,) + tuple(output[1:]) if isinstance(output, tuple) else edited

        zero_logits, zero_steered = compare_logits(
            peft_model, train_inputs, layer_module, alpha_zero_hook
        )
        if not torch.equal(zero_logits, zero_steered):
            raise RuntimeError("Zero-dose steering is not a no-op")
        _, steered_logits = compare_logits(
            peft_model, train_inputs, layer_module, steering_hook
        )
        steering_delta = float((steered_logits - zero_logits).abs().max().item())
        if steering_delta <= 0.0:
            raise RuntimeError("Nonzero steering did not reach the logits")

        # Self-patching at lambda=0 must also be a no-op.
        donor = hidden.detach().clone()

        def self_patch_zero(module, hook_inputs, output):
            current = unwrap_hidden(output)
            edited = current.clone()
            edited[batch_index, last] = (
                current[batch_index, last]
                + 0.0 * (donor[batch_index, last] - current[batch_index, last])
            )
            return (edited,) + tuple(output[1:]) if isinstance(output, tuple) else edited

        patch_plain, patch_zero = compare_logits(
            peft_model, train_inputs, layer_module, self_patch_zero
        )
        if not torch.equal(patch_plain, patch_zero):
            raise RuntimeError("Zero-dose self-patching is not a no-op")

        # Apply final norm and head to an intermediate state for the logit lens.
        norm = resolve(peft_model, norm_path)
        head = resolve(peft_model, head_path)
        lens_logits = head(norm(hidden[:, -1, :]))
        if not torch.isfinite(lens_logits).all():
            raise RuntimeError("Gemma logit-lens readout contains non-finite values")

        # NNSight parity: same prompt, same final residual and same greedy text.
        from nnsight import LanguageModel

        lm = LanguageModel(peft_model, tokenizer=tokenizer)
        nn_layer_path = layer_path_for(lm)
        with lm.trace(formatted):
            captured = resolve(lm, f"{nn_layer_path}.{layer_index}").output[0, -1, :].save()
            nn_ids = lm.generator.output.save()
        nn_vector = captured.detach().float().cpu()
        hf_vector = hidden[0, -1, :].detach().float().cpu()
        cosine = float(torch.nn.functional.cosine_similarity(
            nn_vector.unsqueeze(0), hf_vector.unsqueeze(0)
        ).item())
        if cosine < 0.999:
            raise RuntimeError(f"NNsight/HF residual cosine too low: {cosine:.6f}")
        nn_ids = nn_ids.detach().cpu()
        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        nn_text = tokenizer.decode(nn_ids[0, len(prompt_ids):], skip_special_tokens=True)
        with torch.inference_mode():
            hf_ids = peft_model.generate(
                **encoded, max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        hf_text = tokenizer.decode(
            hf_ids[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True
        )
        if nn_text != hf_text:
            raise RuntimeError("NNsight and Hugging Face greedy generations differ")

    report = {
        "status": "pass",
        "model": args.model,
        "loader": loader,
        "loader_errors": loader_errors,
        "transformers": __version__,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "decoder_layers": len(layers),
        "hidden_size": hidden_size,
        "layer_path": layer_path,
        "norm_path": norm_path,
        "head_path": head_path,
        "layer_tested": layer_index,
        "lora_targets": targets,
        "smoke_loss": float(loss.detach().item()),
        "adapter_reload": reload_ok,
        "adapter_enabled_disabled_max_delta": adapter_delta,
        "steering_nonzero_max_delta": steering_delta,
        "nnsight_residual_cosine": cosine,
        "generation_chars": len(hf_text),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
