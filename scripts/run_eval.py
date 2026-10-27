"""
AQCAP Universal Evaluation Runner.

Executes a complete evaluation pass on a specified dataset split (calib or test)
under a specified quantization configuration and CAP setting.

Usage:
    python scripts/run_eval.py --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset juliet --config configs/qwen7b_fp16.json --split test --cap off
    python scripts/run_eval.py --model microsoft/Phi-3-mini-4k-instruct --dataset juliet --config configs/top_task.json --split test --cap on
    python scripts/run_eval.py --model Qwen/Qwen2.5-Coder-3B-Instruct --dataset primevul_unpaired --config configs/qwen3b_fp16.json --split test --cap off
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import transformers

from aqcap.quant import load_model, memory_report, reset_peak_memory, audit_model
from aqcap.scoring import build_prompt, get_target_token_ids, score_prompt
from aqcap.cap import prune_code
from aqcap.data import load_dataset_split
from aqcap.metrics import compute_classification_metrics, compute_paired_metrics, reconstruct_pairs


def get_git_commit() -> str:
    """Attempts to retrieve the current git commit hash."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit
    except Exception:
        return "unversioned"


def get_environment_info() -> Dict[str, Any]:
    """Captures runtime environment specs for reproducibility."""
    env_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "git_commit": get_git_commit(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
        env_info["gpu_vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        env_info["cuda_version"] = torch.version.cuda
    else:
        env_info["gpu_name"] = "CPU"
        env_info["gpu_vram_total_gb"] = 0.0

    return env_info


def parse_config(config_arg: str, configs_dir: str = "configs") -> Dict[str, Any]:
    """Loads configuration dictionary from path or name."""
    if os.path.isfile(config_arg):
        path = config_arg
    else:
        candidate = os.path.join(configs_dir, f"{config_arg}.json")
        if os.path.isfile(candidate):
            path = candidate
        else:
            raise FileNotFoundError(f"Configuration '{config_arg}' not found at {candidate}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def get_model_slug(model_id: str) -> str:
    m = model_id.lower()
    if "qwen" in m and "7b" in m:
        return "qwen7b"
    elif "qwen" in m and "3b" in m:
        return "qwen3b"
    elif "phi" in m:
        return "phi3"
    elif "llama" in m:
        return "llama8b"
    else:
        return model_id.replace("/", "_").replace("-", "_").lower()


def main():
    parser = argparse.ArgumentParser(description="AQCAP Universal Evaluation Runner")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="HuggingFace model ID or path")
    parser.add_argument("--dataset", default="juliet", choices=["juliet", "primevul_unpaired", "primevul_paired"],
                        help="Target dataset to evaluate")
    parser.add_argument("--config", default="configs/qwen7b_fp16.json", help="Config file path or name (e.g. fp16, nf4, top_task)")
    parser.add_argument("--split", default="test", choices=["test", "calib"], help="Target split to evaluate")
    parser.add_argument("--cap", default="off", choices=["on", "off"], help="Context-Aware Pruning toggle")
    parser.add_argument("--splits_dir", default="data/splits", help="Path to splits metadata directory")
    parser.add_argument("--data_dir", default="data", help="Path to raw dataset directory")
    parser.add_argument("--output_dir", default="results", help="Directory where JSONL results are stored")
    parser.add_argument("--device", default="cuda", help="Execution device (cuda, auto, or cpu)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap (for debugging)")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum allowed prompt tokens")
    args = parser.parse_args()

    # 1. Environment & Config setup
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = parse_config(args.config)
    cfg_name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    model_tag = get_model_slug(args.model)
    cap_tag = "cap_on" if args.cap == "on" else "cap_off"

    run_id = f"{model_tag}_{args.dataset}_{cfg_name}_{args.split}_{cap_tag}"
    output_jsonl_path = os.path.join(args.output_dir, f"{run_id}.jsonl")
    run_meta_path = os.path.join(args.output_dir, f"{run_id}_meta.json")

    print(f"\n{'='*70}\n  AQCAP EVALUATION RUNNER: {run_id}\n{'='*70}")
    print(f"  Model       : {args.model}")
    print(f"  Dataset     : {args.dataset}")
    print(f"  Config      : {cfg_name} ({cfg.get('description', '')})")
    print(f"  Split       : {args.split}")
    print(f"  CAP Pruning : {args.cap.upper()}")
    print(f"  Output JSONL: {output_jsonl_path}")

    # 2. Model Loading & Surgery
    precision_map = {int(k): v for k, v in cfg["precision_map"].items()}
    nf4_bs = cfg.get("nf4_blocksize", 64)

    reset_peak_memory()
    model, tokenizer = load_model(
        model_id=args.model,
        precision_map=precision_map,
        nf4_blocksize=nf4_bs,
        device=args.device,
    )
    yes_ids, no_ids = get_target_token_ids(tokenizer)
    mem_init = memory_report(model)

    # 3. Load Split Samples via aqcap.data
    data_obj = load_dataset_split(
        dataset_name=args.dataset,
        split=args.split,
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
    )

    ordered_samples = data_obj["samples"]
    is_paired = data_obj["is_paired"]

    if args.max_samples is not None:
        ordered_samples = ordered_samples[:args.max_samples]

    print(f"[RUN_EVAL] Loaded {len(ordered_samples)} functions from {args.dataset} ({args.split} split).")

    # 4. Write Initial Metadata Row to JSONL and run_meta.json
    env_specs = get_environment_info()
    run_meta = {
        "type": "metadata",
        "run_id": run_id,
        "model": args.model,
        "dataset": args.dataset,
        "config_name": cfg_name,
        "config": cfg,
        "split": args.split,
        "cap": args.cap,
        "seed": args.seed,
        "is_paired": is_paired,
        "weights_mb": mem_init["weights_mb"],
        "initial_peak_mb": mem_init["peak_mb"],
        "target_token_ids": {
            "yes_ids": yes_ids,
            "no_ids": no_ids,
        },
        "env": env_specs,
        "total_samples": len(ordered_samples),
    }

    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    jsonl_file = open(output_jsonl_path, "w", encoding="utf-8")
    jsonl_file.write(json.dumps(run_meta) + "\n")
    jsonl_file.flush()

    # 5. Evaluation Loop
    print(f"\n[RUN_EVAL] Beginning inference loop across {len(ordered_samples)} samples...")
    t_start = time.time()
    y_true_list = []
    y_prob_list = []
    top1_correct_count = 0
    cap_keyword_loss_count = 0

    for i, item in enumerate(ordered_samples):
        raw_code = item["func"]
        target = item["target"]
        pair_id = item.get("pair_id")
        cwe = item.get("cwe", "Unknown")
        project = item.get("project", "Unknown")

        # CAP Pruning
        removed_comments_count = 0
        sec_keywords_found = []
        if args.cap == "on":
            pruned_code, comments, sec_keywords_found = prune_code(raw_code, language="c")
            code_for_model = pruned_code
            removed_comments_count = len(comments)
            if sec_keywords_found:
                cap_keyword_loss_count += 1
        else:
            code_for_model = raw_code

        # Format prompt & Score
        prompt = build_prompt(tokenizer, code_for_model, mode="zero")
        res = score_prompt(model, tokenizer, prompt, yes_ids=yes_ids, no_ids=no_ids, max_tokens=args.max_tokens)

        p_vuln = res["p_vuln"]
        p_safe = res["p_safe"]
        pred_50 = res["pred_50"]
        top1_token = res["top1_token"]
        is_in_yes_no = res["is_in_yes_no"]

        if is_in_yes_no:
            top1_correct_count += 1

        y_true_list.append(target)
        y_prob_list.append(p_vuln)

        # Log sample row
        sample_log = {
            "type": "sample",
            "sample_index": i,
            "idx": item["idx"],
            "pair_id": pair_id,
            "target": target,
            "p_vuln": p_vuln,
            "p_safe": p_safe,
            "pred_50": pred_50,
            "top1_id": res["top1_id"],
            "top1_token": top1_token,
            "is_in_yes_no": is_in_yes_no,
            "n_input_tokens": res["n_input_tokens"],
            "latency_s": res["latency_s"],
            "cwe": cwe,
            "project": project,
            "cap_enabled": (args.cap == "on"),
            "removed_comments": removed_comments_count,
            "sec_keywords": sec_keywords_found,
        }

        jsonl_file.write(json.dumps(sample_log) + "\n")
        jsonl_file.flush()

        # Console Progress
        status_sym = "✔" if pred_50 == target else "✖"
        if (i + 1) % 50 == 0 or (i + 1) == len(ordered_samples):
            print(f"  [{i+1:>4}/{len(ordered_samples)}] {status_sym} True:{target} Pred:{pred_50} "
                  f"(p_vuln={p_vuln:.3f}, top1='{top1_token}') | {res['latency_s']:.2f}s")

    jsonl_file.close()
    elapsed_total = time.time() - t_start

    # 6. Final Memory & Metrics Summary
    mem_final = memory_report(model)
    top1_compliance_pct = (top1_correct_count / len(ordered_samples)) * 100.0

    y_true_arr = np.array(y_true_list)
    y_prob_arr = np.array(y_prob_list)
    final_metrics = compute_classification_metrics(y_true_arr, y_prob_arr, threshold=0.5)

    print(f"\n{'='*70}\n  RUN COMPLETED: {run_id}\n{'='*70}")
    print(f"  Duration          : {elapsed_total:.1f}s ({elapsed_total/len(ordered_samples):.3f}s/sample)")
    print(f"  Peak VRAM         : {mem_final['peak_mb']:,.1f} MB ({mem_final['peak_gb']:.3f} GB)")
    print(f"  Top-1 YES/NO Match: {top1_compliance_pct:.1f}%")
    print(f"  AUROC             : {final_metrics['auroc']:.4f}")
    print(f"  AUPRC             : {final_metrics['auprc']:.4f}")
    print(f"  FPR@90% Recall    : {final_metrics['fpr_at_90_recall']:.4f}")
    print(f"  FPR@95% Recall    : {final_metrics['fpr_at_95_recall']:.4f}")
    print(f"  Balanced Acc (0.5): {final_metrics['balanced_accuracy']:.4f}")
    print(f"  F1-Score (0.5)    : {final_metrics['f1']:.4f}")
    print(f"  MCC               : {final_metrics['mcc']:.4f}")
    if args.cap == "on":
        print(f"  Security Info Loss: {cap_keyword_loss_count}/{len(ordered_samples)} functions ({cap_keyword_loss_count/len(ordered_samples)*100:.1f}%)")
    print(f"  Saved JSONL       : {output_jsonl_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
