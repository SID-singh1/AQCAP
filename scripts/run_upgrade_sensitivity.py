"""
Upgrade ("leave-one-in") layer sensitivity for low-bit mixed precision.

Single-layer perturbation (run_sensitivity.py) measures layer i quantized while every
other layer stays FP16. At a 2-bit operating point that is the wrong context: in the
deployed model every other layer is already 2-bit. This script measures sensitivity in
the deployment context instead:

    base model   : every layer at --base_prec (e.g. hqq2)
    variant i    : layer i at --up_prec (e.g. hqq4), all others at --base_prec
    S_up[i]      = KL_base - KL_i   (mean binary KL of P(vuln) vs FP16 on calib)
    dAUROC_up[i] = AUROC_i - AUROC_base

Writes results/{output_csv} (checkpointed per layer, resumable) and
configs/{prefix}top_up.json (top-K layers by S_up at --up_prec, rest at --base_prec).

Usage:
    python scripts/run_upgrade_sensitivity.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset juliet --prefix qwen7b_hqq_2bit_ --budget 7 \
        --output_csv results/upgrade_sensitivity_qwen7b_hqq2_juliet.csv
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import gc
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aqcap.data import load_dataset_split
from aqcap.quant import load_model
from aqcap.scoring import build_prompt, get_target_token_ids, score_prompt
from aqcap.sensitivity import compute_binary_kl


def free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_all(model, tokenizer, samples, max_tokens):
    yes_ids, no_ids = get_target_token_ids(tokenizer)
    probs = []
    for s in samples:
        prompt = build_prompt(tokenizer, s["func"], mode="zero")
        res = score_prompt(model, tokenizer, prompt, yes_ids=yes_ids, no_ids=no_ids, max_tokens=max_tokens)
        probs.append(res["p_vuln"])
    return np.array(probs)


def mean_kl(ref, q):
    return float(np.mean([compute_binary_kl(a, b) for a, b in zip(ref, q)]))


def main():
    ap = argparse.ArgumentParser(description="Upgrade (leave-one-in) sensitivity profiling")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="juliet", choices=["juliet", "primevul_unpaired", "primevul_paired"])
    ap.add_argument("--base_prec", default="hqq2")
    ap.add_argument("--up_prec", default="hqq4")
    ap.add_argument("--budget", type=int, required=True, help="Number of layers upgraded in the generated config")
    ap.add_argument("--prefix", required=True, help="Config prefix, e.g. qwen7b_hqq_2bit_")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--configs_dir", default="configs")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    num_layers = AutoConfig.from_pretrained(args.model).num_hidden_layers
    samples = load_dataset_split(dataset_name=args.dataset, split="calib", max_tokens=args.max_tokens)["samples"]
    y = np.array([s["target"] for s in samples])
    cache_path = args.output_csv.replace(".csv", "_refs.npz")
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # 1. FP16 reference and uniform low-bit baseline (cached so a restart skips them)
    if os.path.isfile(cache_path):
        z = np.load(cache_path)
        p_fp16, p_base = z["p_fp16"], z["p_base"]
        print(f"[UPGRADE] Loaded cached FP16/base reference scores from {cache_path}")
    else:
        t0 = time.time()
        model, tok = load_model(args.model, {i: "fp16" for i in range(num_layers)}, device=args.device)
        p_fp16 = score_all(model, tok, samples, args.max_tokens)
        free(model)
        print(f"[UPGRADE] FP16 calib AUROC {roc_auc_score(y, p_fp16):.4f} ({time.time() - t0:.0f}s)")
        t0 = time.time()
        model, tok = load_model(args.model, {i: args.base_prec for i in range(num_layers)}, device=args.device)
        p_base = score_all(model, tok, samples, args.max_tokens)
        free(model)
        print(f"[UPGRADE] Uniform {args.base_prec} calib AUROC {roc_auc_score(y, p_base):.4f} ({time.time() - t0:.0f}s)")
        np.savez(cache_path, p_fp16=p_fp16, p_base=p_base)

    kl_base = mean_kl(p_fp16, p_base)
    auc_fp16 = roc_auc_score(y, p_fp16)
    auc_base = roc_auc_score(y, p_base)

    # 2. Upgrade one layer at a time (checkpointed per layer)
    rows = []
    if os.path.isfile(args.output_csv):
        rows = pd.read_csv(args.output_csv).to_dict("records")
        print(f"[UPGRADE] Resuming: {len(rows)} layers already profiled")
    done = {int(r["layer"]) for r in rows}

    print(f"{'layer':<7}{'S_up':<12}{'dAUROC_up':<12}{'time':<8}")
    for i in range(num_layers):
        if i in done:
            continue
        t0 = time.time()
        pmap = {j: (args.up_prec if j == i else args.base_prec) for j in range(num_layers)}
        model, tok = load_model(args.model, pmap, device=args.device)
        p_i = score_all(model, tok, samples, args.max_tokens)
        free(model)
        s_up = kl_base - mean_kl(p_fp16, p_i)
        d_auc = roc_auc_score(y, p_i) - auc_base
        rows.append({"layer": i, "S_up": s_up, "dAUROC_up": d_auc})
        pd.DataFrame(rows).sort_values("layer").to_csv(args.output_csv, index=False)
        print(f"{i:<7}{s_up:<12.5f}{d_auc:<12.4f}{time.time() - t0:<8.0f}", flush=True)

    df = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)

    # 3. Config: upgrade the top-K layers by S_up
    top = sorted(df.sort_values("S_up", ascending=False)["layer"].iloc[: args.budget].astype(int).tolist())
    name = f"{args.prefix}top_up"
    cfg = {
        "name": name,
        "description": f"Upgrade sensitivity: {args.up_prec} on top-{args.budget} layers by S_up "
                       f"(KL recovery in the uniform-{args.base_prec} context), rest {args.base_prec}",
        "precision_map": {str(j): (args.up_prec if j in top else args.base_prec) for j in range(num_layers)},
        "sensitive_layers": top,
    }
    os.makedirs(args.configs_dir, exist_ok=True)
    with open(os.path.join(args.configs_dir, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n[UPGRADE] FP16 AUROC {auc_fp16:.4f} | uniform {args.base_prec} {auc_base:.4f} | KL_base {kl_base:.5f}")
    print(f"[UPGRADE] Wrote configs/{name}.json with layers {top}")


if __name__ == "__main__":
    main()
