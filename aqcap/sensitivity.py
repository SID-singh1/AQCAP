"""
AQCAP Layer Sensitivity Identification (LSI) Engine.

Computes the three sensitivity measures required for RQ2 (ROADMAP.md Section 3.4):
1. S_task[i]: Task-specific sensitivity — mean KL divergence over the 2-way YES/NO
   vulnerability probability distribution on the calibration set (calib).
   Also logs delta_auroc[i] = AUROC_fp16 - AUROC_i.
2. S_gen[i]: Generic language-modelling sensitivity — mean full-vocabulary KL divergence
   between FP16 and layer-i-quantized next-token distributions on raw code.
3. S_l2[i]: Legacy LSI metric — mean L2 norm of layer i's output hidden states on FP16.

Generates:
- results/sensitivity_{model_name}.csv
- configs/top_task.json (INT8 on top-16 by S_task, NF4 on bottom-16)
- configs/top_gen.json (INT8 on top-16 by S_gen, NF4 on bottom-16)
- configs/bottom_task.json (INT8 on bottom-16 by S_task, NF4 on top-16 — sanity check control)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import copy
import json
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

from aqcap.quant import (
    get_decoder_layers,
    _quantize_submodule_linears,
    _create_nf4_linear,
    _create_hqq_linear,
    load_model,
)
from aqcap.scoring import build_prompt, get_target_token_ids, score_prompt
from aqcap.metrics import compute_classification_metrics


def compute_binary_kl(p_ref: float, p_quant: float, eps: float = 1e-7) -> float:
    """
    Computes KL divergence D_KL(P_ref || Q_quant) between two Bernoulli distributions:
    D_KL(P || Q) = p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
    """
    p = np.clip(p_ref, eps, 1.0 - eps)
    q = np.clip(p_quant, eps, 1.0 - eps)
    kl = p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))
    return float(kl)


def compute_full_vocab_kl(p_ref: torch.Tensor, log_p_ref: torch.Tensor, log_q: torch.Tensor) -> float:
    """
    Computes token-averaged Kullback-Leibler divergence KL(P_ref || Q) across the full vocabulary.
    
    Args:
        p_ref: Reference FP16 probability tensor of shape [T, vocab_size]
        log_p_ref: Reference FP16 log-probability tensor of shape [T, vocab_size]
        log_q: Quantized layer log-probability tensor of shape [T, vocab_size]
        
    Returns:
        float: Mean KL divergence across all token positions T.
               If distributions are identical (log_q == log_p_ref), returns exactly 0.0.
    """
    kl_per_token = torch.sum(p_ref * (log_p_ref - log_q), dim=-1)
    kl_per_token = torch.clamp(kl_per_token, min=0.0)
    return float(kl_per_token.mean().item())


def profile_legacy_l2_saliency(
    model: nn.Module,
    tokenizer: Any,
    calib_samples: List[Dict[str, Any]],
    device: str = "cuda",
    max_samples: int = 100,
) -> np.ndarray:
    """
    Computes S_l2[i]: token-averaged L2 norm of each layer's output hidden states
    evaluated on the unquantized FP16 model.
    """
    layers = get_decoder_layers(model)
    num_layers = len(layers)
    hidden_norms = [[] for _ in range(num_layers)]

    hooks = []
    for i, layer in enumerate(layers):
        def make_hook(idx):
            def hook_fn(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                with torch.no_grad():
                    token_norms = torch.norm(hs.float(), p=2, dim=-1)
                    mean_layer_norm = token_norms.mean().item()
                    hidden_norms[idx].append(mean_layer_norm)
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(i)))

    target_dev = next(model.parameters()).device if (device == "auto" or not device) else device
    samples_to_run = calib_samples[:max_samples]
    print(f"[SENSITIVITY] Profiling S_l2 hidden state norms on {len(samples_to_run)} samples...")

    model.eval()
    for s in samples_to_run:
        prompt = build_prompt(tokenizer, s["func"], mode="zero")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if hasattr(inputs, "to"):
            inputs = inputs.to(target_dev)
        else:
            inputs = {k: v.to(target_dev) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)

    for h in hooks:
        h.remove()

    mean_s_l2 = np.array([np.mean(hidden_norms[i]) for i in range(num_layers)])
    return mean_s_l2


def swap_block_linears(
    block: nn.Module,
    perturbation: str = "nf4",
    nf4_blocksize: int = 64,
    compute_dtype: Any = torch.float16,
) -> List[Tuple[nn.Module, str, nn.Linear]]:
    """
    Replaces all nn.Linear projections inside a decoder block with quantized linear layers in-place.
    Zero-copy: records the untouched original nn.Linear layers for instantaneous restoration
    without any copy.deepcopy() or GPU/CPU memory duplication.
    Supports perturbation='nf4', 'hqq2'/'2bit', 'hqq3'/'3bit', 'hqq4'/'4bit'.
    """
    saved_linears = []
    pert = perturbation.lower()

    def _recurse(module: nn.Module):
        for child_name, child_module in list(module.named_children()):
            is_linear = isinstance(child_module, nn.Linear) and not (
                bnb is not None and isinstance(child_module, bnb.nn.Linear4bit)
            )
            # Avoid re-quantizing HQQLinear if already one
            if is_linear and child_module.__class__.__name__ != "HQQLinear":
                saved_linears.append((module, child_name, child_module))
                if pert in ("hqq2", "2bit"):
                    # Move to the source weight's device (CUDA during sensitivity profiling)
                    # so HQQ 2-bit/3-bit optimization runs on GPU and doesn't hang
                    hqq_dev = child_module.weight.device if child_module.weight.is_cuda else (
                        "cuda" if torch.cuda.is_available() else child_module.weight.device
                    )
                    q_layer = _create_hqq_linear(
                        child_module, nbits=2, group_size=64, del_orig=False,
                        compute_dtype=compute_dtype, device=hqq_dev
                    )
                elif pert in ("hqq3", "3bit"):
                    hqq_dev = child_module.weight.device if child_module.weight.is_cuda else (
                        "cuda" if torch.cuda.is_available() else child_module.weight.device
                    )
                    q_layer = _create_hqq_linear(
                        child_module, nbits=3, group_size=64, del_orig=False,
                        compute_dtype=compute_dtype, device=hqq_dev
                    )
                elif pert in ("hqq4", "4bit"):
                    hqq_dev = child_module.weight.device if child_module.weight.is_cuda else (
                        "cuda" if torch.cuda.is_available() else child_module.weight.device
                    )
                    q_layer = _create_hqq_linear(
                        child_module, nbits=4, group_size=64, del_orig=False,
                        compute_dtype=compute_dtype, device=hqq_dev
                    )
                else:
                    # Default: NF4 (BitsAndBytes)
                    q_layer = _create_nf4_linear(
                        child_module, blocksize=nf4_blocksize, compute_dtype=compute_dtype
                    )
                setattr(module, child_name, q_layer)
            else:
                _recurse(child_module)

    _recurse(block)
    return saved_linears



def swap_block_linears_to_nf4(
    block: nn.Module,
    nf4_blocksize: int = 64,
    compute_dtype: Any = torch.float16,
) -> List[Tuple[nn.Module, str, nn.Linear]]:
    """Legacy alias for swap_block_linears with perturbation='nf4'."""
    return swap_block_linears(block, perturbation="nf4", nf4_blocksize=nf4_blocksize, compute_dtype=compute_dtype)


def restore_block_linears(saved_linears: List[Tuple[nn.Module, str, nn.Linear]]):
    """
    Restores the original untouched nn.Linear projections with 0 bytes memory allocation.
    """
    for parent, name, orig_linear in saved_linears:
        setattr(parent, name, orig_linear)


def run_full_sensitivity_profile(
    model_id: str,
    calib_samples: List[Dict[str, Any]],
    output_csv_path: str,
    device: str = "cuda",
    nf4_blocksize: int = 64,
    max_tokens: int = 2048,
    resume: bool = True,
    perturbation: str = "nf4",
) -> pd.DataFrame:
    """
    Main sensitivity profiling pipeline for RQ2.
    Filters calibration samples to those whose full security prompt fits within
    max_tokens before any scoring begins, so no sample can crash the loop.
    Supports resume control and configurable perturbation (NF4 or HQQ2).
    """
    print(f"\n{'='*70}\n  AQCAP LAYER SENSITIVITY PROFILER (RQ2)\n{'='*70}")
    print(f"[SENSITIVITY] Target Model: {model_id}")
    print(f"[SENSITIVITY] Calibration Samples (raw): {len(calib_samples)} functions")
    print(f"[SENSITIVITY] Perturbation Type: {perturbation.upper()}")
    print(f"[SENSITIVITY] Resume Checkpoint: {resume}")

    print(f"[SENSITIVITY] Loading unquantized base model in FP16 to {device}...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_id)
    num_layers_cfg = getattr(config, "num_hidden_layers", 32)
    fp16_map = {i: "fp16" for i in range(num_layers_cfg)}
    model, tokenizer = load_model(
        model_id=model_id,
        precision_map=fp16_map,
        device=device,
    )
    model.eval()
    print(f"[SENSITIVITY] Base model loaded in {time.time() - t0:.1f}s.")

    layers = get_decoder_layers(model)
    num_layers = len(layers)
    yes_ids, no_ids = get_target_token_ids(tokenizer)

    # ── Token-length filter: drop any sample whose full prompt > max_tokens ──
    # This must happen before S_l2, FP16 reference, and the perturbation loop
    # so all three steps operate on exactly the same set of samples.
    device_obj = next(model.parameters()).device
    filtered = []
    dropped = 0
    for s in calib_samples:
        prompt = build_prompt(tokenizer, s["func"], mode="zero")
        n_tok = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"].shape[1]
        if n_tok <= max_tokens:
            filtered.append(s)
        else:
            dropped += 1
    calib_samples = filtered
    print(f"[SENSITIVITY] After token filter (max={max_tokens}): "
          f"{len(calib_samples)} kept, {dropped} dropped.")

    if len(calib_samples) == 0:
        raise RuntimeError("All calibration samples exceed max_tokens. "
                           "Check data filtering in data.py.")

    # ── Step 1: Compute Legacy S_l2 norm on unquantized model ──────────────
    s_l2_values = profile_legacy_l2_saliency(
        model, tokenizer, calib_samples, device=device
    )

    # ── Step 2: Unquantized FP16 Baseline Reference ────────────────────────
    print(f"[SENSITIVITY] Computing FP16 reference probabilities on "
          f"{len(calib_samples)} calib samples...")
    fp16_probs = []
    y_true = np.array([s["target"] for s in calib_samples])

    for s in calib_samples:
        prompt = build_prompt(tokenizer, s["func"], mode="zero")
        res = score_prompt(model, tokenizer, prompt,
                           yes_ids=yes_ids, no_ids=no_ids,
                           max_tokens=max_tokens)
        fp16_probs.append(res["p_vuln"])

    fp16_probs_arr = np.array(fp16_probs)
    fp16_metrics = compute_classification_metrics(y_true, fp16_probs_arr)
    auroc_fp16 = fp16_metrics["auroc"]
    print(f"[SENSITIVITY] FP16 Calib AUROC: {auroc_fp16:.4f}")

    # ── Step 2b: Cache FP16 Generic Code Log-Probs on CPU for S_gen ────────
    # Uses at most 30 samples, truncated to 512 tokens (raw code, no security prompt)
    print(f"[SENSITIVITY] Pre-computing FP16 generic LM log-probs "
          f"(up to 30 filtered code samples, max_length=512 tokens)...")
    generic_samples = calib_samples[:30]
    cached_generic_fp16 = []
    for s in generic_samples:
        code_tokens = tokenizer(
            f"```c\n{s['func']}\n```",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        target_dev = next(model.parameters()).device if (device == "auto" or not device) else device
        code_tokens_dev = {k: v.to(target_dev) for k, v in code_tokens.items()}
        with torch.no_grad():
            out_fp16 = model(**code_tokens_dev)
            logits_fp16 = out_fp16.logits[0, :-1, :]   # [T-1, vocab]
            p_fp16     = F.softmax(logits_fp16, dim=-1).to("cpu")
            log_p_fp16 = F.log_softmax(logits_fp16, dim=-1).to("cpu")
            cached_generic_fp16.append({
                "code_tokens_dev": code_tokens_dev,
                "p_fp16":     p_fp16,
                "log_p_fp16": log_p_fp16,
            })

    # ── Step 3: Layer-by-Layer NF4 Perturbation ────────────────────────────
    s_task_list = []
    delta_auroc_list = []
    s_gen_list = []

    # Check for existing checkpoint to resume seamlessly
    existing_rows = {}
    if resume and os.path.isfile(output_csv_path):
        try:
            prev_df = pd.read_csv(output_csv_path)
            if "layer" in prev_df.columns and "S_task" in prev_df.columns:
                for _, r in prev_df.iterrows():
                    existing_rows[int(r["layer"])] = r.to_dict()
                print(f"[SENSITIVITY] Found existing checkpoint with {len(existing_rows)} layers in {output_csv_path}.")
        except Exception as e:
            existing_rows = {}
    elif not resume and os.path.isfile(output_csv_path):
        print(f"[SENSITIVITY] Checkpoint resumption disabled (resume=False). Overwriting {output_csv_path} from scratch.")

    print(f"\n[SENSITIVITY] Running single-layer {perturbation.upper()} perturbations across "
          f"{num_layers} layers ({len(calib_samples)} samples each)...")
    print(f"  {'Layer':<8}{'S_task (KL)':<14}{'ΔAUROC':<12}"
          f"{'S_gen (LM KL)':<16}{'S_l2 (Norm)':<14}")
    print(f"  {'─'*62}")

    for i in range(num_layers):
        if i in existing_rows:
            row = existing_rows[i]
            s_task = float(row["S_task"])
            delta_auc = float(row["delta_auroc"])
            s_gen = float(row["S_gen"])
            s_l2 = float(row["S_l2"])
            print(f"  {i:<8}{s_task:<14.4e}{delta_auc:<12.4f}"
                  f"{s_gen:<16.4f}{s_l2:<14.2f} (from checkpoint)")
            s_task_list.append(s_task)
            delta_auroc_list.append(delta_auc)
            s_gen_list.append(s_gen)
            continue

        block = layers[i]
        layer_dev = next(block.parameters()).device

        # 3a. Zero-copy swap of nn.Linear -> quantized linear in-place
        #     Saves the exact original nn.Linear layers in a list.
        #     NO copy.deepcopy(), NO duplicate GPU memory allocation.
        saved_linears = swap_block_linears(
            block,
            perturbation=perturbation,
            nf4_blocksize=nf4_blocksize,
            compute_dtype=torch.float16,
        )
        if layer_dev.type == "cuda" and torch.cuda.is_available():
            block.to(layer_dev)
            for _, m in block.named_modules():
                if bnb is not None and isinstance(m, bnb.nn.Linear4bit):
                    m.to(layer_dev)
                    if not m.weight.is_cuda:
                        m.weight = m.weight.cuda(layer_dev)
            torch.cuda.synchronize()

        # 3b. Evaluate S_task on the security-prompt vulnerability classifier
        quant_probs = []
        for s in calib_samples:
            prompt = build_prompt(tokenizer, s["func"], mode="zero")
            res = score_prompt(model, tokenizer, prompt,
                               yes_ids=yes_ids, no_ids=no_ids,
                               max_tokens=max_tokens)
            quant_probs.append(res["p_vuln"])

        quant_probs_arr = np.array(quant_probs)
        q_metrics = compute_classification_metrics(y_true, quant_probs_arr)
        auroc_q   = q_metrics["auroc"]
        delta_auc = float(auroc_fp16 - auroc_q)

        kl_tasks = [
            compute_binary_kl(fp16_probs_arr[k], quant_probs_arr[k])
            for k in range(len(calib_samples))
        ]
        s_task = float(np.mean(kl_tasks))

        # 3c. Evaluate S_gen: full-vocab KL on raw code (generic LM)
        gen_kl_list = []
        for g in cached_generic_fp16:
            with torch.no_grad():
                out_quant  = model(**g["code_tokens_dev"])
                logits_q   = out_quant.logits[0, :-1, :]
                log_probs_q = F.log_softmax(logits_q, dim=-1).to("cpu")
            gen_kl = compute_full_vocab_kl(
                g["p_fp16"], g["log_p_fp16"], log_probs_q
            )
            gen_kl_list.append(gen_kl)

        s_gen = float(np.mean(gen_kl_list))

        s_task_list.append(s_task)
        delta_auroc_list.append(delta_auc)
        s_gen_list.append(s_gen)

        print(f"  {i:<8}{s_task:<14.4e}{delta_auc:<12.4f}"
              f"{s_gen:<16.4f}{s_l2_values[i]:<14.2f}")

        # 3d. Zero-copy restoration of original nn.Linear layers
        restore_block_linears(saved_linears)
        del saved_linears
        if layer_dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3e. Checkpoint to CSV immediately after each layer completes
        current_df = pd.DataFrame({
            "layer":       list(range(len(s_task_list))),
            "S_task":      s_task_list,
            "delta_auroc": delta_auroc_list,
            "S_gen":       s_gen_list,
            "S_l2":        s_l2_values[:len(s_task_list)],
        })
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        current_df.to_csv(output_csv_path, index=False)

    # ── Step 4: Assemble Final Results ─────────────────────────────────────
    df = pd.DataFrame({
        "layer":       list(range(num_layers)),
        "S_task":      s_task_list,
        "delta_auroc": delta_auroc_list,
        "S_gen":       s_gen_list,
        "S_l2":        s_l2_values,
    })

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"\n[SENSITIVITY] Saved sensitivity results -> {output_csv_path}")

    return df


def generate_sensitivity_configs(
    sensitivity_df: pd.DataFrame,
    configs_dir: str = "configs",
    num_layers: int = 32,
    int8_budget: int = 16,
    prefix: str = "",
    baseline_prec: str = "nf4",
    sensitive_prec: str = "int8",
) -> Dict[str, str]:
    """
    Generates data-driven dynamic configs matching the budget:
    1. configs/{prefix}top_task.json: sensitive_prec on top-N by S_task, baseline_prec on remainder.
    2. configs/{prefix}top_gen.json: sensitive_prec on top-N by S_gen, baseline_prec on remainder.
    3. configs/{prefix}bottom_task.json: sensitive_prec on bottom-N by S_task, baseline_prec on remainder.
    """
    os.makedirs(configs_dir, exist_ok=True)
    out_paths = {}

    # 1. Top Task
    top_task_layers = set(sensitivity_df.sort_values(by="S_task", ascending=False)["layer"].iloc[:int8_budget])
    top_task_map = {str(i): sensitive_prec if i in top_task_layers else baseline_prec for i in range(num_layers)}
    tt_name = f"{prefix}top_task"
    path_tt = os.path.join(configs_dir, f"{tt_name}.json")
    with open(path_tt, "w", encoding="utf-8") as f:
        json.dump({
            "name": tt_name,
            "description": f"Proposed method: {sensitive_prec.upper()} on top-{int8_budget} layers by S_task (task KL divergence)",
            "precision_map": top_task_map,
            "nf4_blocksize": 64,
            "int8_layers": sorted(list(top_task_layers)),
            "sensitive_layers": sorted(list(top_task_layers)),
        }, f, indent=2)
    out_paths[tt_name] = path_tt

    # 2. Top Gen
    top_gen_layers = set(sensitivity_df.sort_values(by="S_gen", ascending=False)["layer"].iloc[:int8_budget])
    top_gen_map = {str(i): sensitive_prec if i in top_gen_layers else baseline_prec for i in range(num_layers)}
    tg_name = f"{prefix}top_gen"
    path_tg = os.path.join(configs_dir, f"{tg_name}.json")
    with open(path_tg, "w", encoding="utf-8") as f:
        json.dump({
            "name": tg_name,
            "description": f"Generic baseline: {sensitive_prec.upper()} on top-{int8_budget} layers by S_gen (language modeling KL)",
            "precision_map": top_gen_map,
            "nf4_blocksize": 64,
            "int8_layers": sorted(list(top_gen_layers)),
            "sensitive_layers": sorted(list(top_gen_layers)),
        }, f, indent=2)
    out_paths[tg_name] = path_tg

    # 3. Bottom Task (Sanity Check Control)
    bottom_task_layers = set(sensitivity_df.sort_values(by="S_task", ascending=True)["layer"].iloc[:int8_budget])
    bottom_task_map = {str(i): sensitive_prec if i in bottom_task_layers else baseline_prec for i in range(num_layers)}
    bt_name = f"{prefix}bottom_task"
    path_bt = os.path.join(configs_dir, f"{bt_name}.json")
    with open(path_bt, "w", encoding="utf-8") as f:
        json.dump({
            "name": bt_name,
            "description": f"Sanity check control: {sensitive_prec.upper()} on bottom-{int8_budget} layers by S_task (least sensitive)",
            "precision_map": bottom_task_map,
            "nf4_blocksize": 64,
            "int8_layers": sorted(list(bottom_task_layers)),
            "sensitive_layers": sorted(list(bottom_task_layers)),
        }, f, indent=2)
    out_paths[bt_name] = path_bt

    for k, p in out_paths.items():
        print(f"[SENSITIVITY] Generated dynamic config: {p}")

    return out_paths
