"""
AQCAP Leakage Audit Verification Runner.

Reruns Qwen2.5-Coder-7B-Instruct in FP16 zero-shot on the leakage-audited Juliet dataset
and compares AUROC before vs after audit.

Usage:
    python scripts/rerun_qwen7b_juliet_audit.py --device auto
    python scripts/rerun_qwen7b_juliet_audit.py --split test --device auto
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import argparse
from typing import Dict, List, Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aqcap.quant import load_model, memory_report
from aqcap.scoring import build_prompt, get_target_token_ids, score_prompt
from aqcap.data import load_dataset_split
from aqcap.metrics import (
    compute_classification_metrics,
    compute_paired_metrics,
    reconstruct_pairs,
    bootstrap_paired_confidence_intervals,
)

QWEN_7B_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def run_audit_eval(
    model_id: str = QWEN_7B_MODEL,
    split: str = "calib",
    device: str = "auto",
    output_dir: str = "results",
    max_tokens: int = 2048,
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*70}\n  AQCAP LEAKAGE AUDIT VERIFICATION: FP16 {model_id}\n{'='*70}")
    print(f"  Dataset    : Juliet C/C++ (Audited: Zero comments, macros, or leaking IDs)")
    print(f"  Split      : {split}")
    print(f"  Prompt Mode: zero_shot")
    print(f"  Device     : {device}")

    # 1. Load Model in FP16
    print(f"\n[AUDIT_RUN] Loading unquantized model in FP16 (device={device})...")
    t0 = time.time()
    # Map all 28 layers to fp16
    precision_map = {i: "fp16" for i in range(28)}
    model, tokenizer = load_model(
        model_id=model_id,
        precision_map=precision_map,
        device=device,
    )
    yes_ids, no_ids = get_target_token_ids(tokenizer)
    mem_info = memory_report(model)
    print(f"[AUDIT_RUN] Model loaded in {time.time() - t0:.1f}s. VRAM/Weights: {mem_info['weights_mb']:.1f} MB")
    print(f"            Filtered Target Tokens: YES={yes_ids}, NO={no_ids}")

    # 2. Load Audited Dataset
    data_obj = load_dataset_split(
        dataset_name="juliet",
        split=split,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )
    samples = data_obj["samples"]
    print(f"[AUDIT_RUN] Loaded {len(samples)} functions ({len(samples)//2} pairs) from audited Juliet {split} split.")

    # 3. Score all samples
    print(f"\n[AUDIT_RUN] Scoring {len(samples)} samples zero-shot...")
    t_start = time.time()
    scored_items = []
    y_true = []
    y_prob = []
    top1_correct = 0

    for i, item in enumerate(samples):
        prompt = build_prompt(tokenizer, item["func"], mode="zero")
        res = score_prompt(model, tokenizer, prompt, yes_ids=yes_ids, no_ids=no_ids, max_tokens=max_tokens)

        p_vuln = res["p_vuln"]
        p_safe = res["p_safe"]
        target = item["target"]
        top1_token = res["top1_token"]

        if res["is_in_yes_no"]:
            top1_correct += 1

        y_true.append(target)
        y_prob.append(p_vuln)

        scored_items.append({
            "idx": item["idx"],
            "pair_id": item["pair_id"],
            "target": target,
            "p_vuln": p_vuln,
            "p_safe": p_safe,
            "pred_50": res["pred_50"],
            "top1_token": top1_token,
            "latency_s": res["latency_s"],
            "cwe": item.get("cwe", "Unknown"),
            "project": "Juliet",
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            status = "✔" if res["pred_50"] == target else "✖"
            print(f"  [{i+1:>4}/{len(samples)}] {status} True:{target} Pred:{res['pred_50']} (p_vuln={p_vuln:.3f}, top1='{top1_token}')")

    elapsed = time.time() - t_start
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)

    # 4. Metrics & Bootstrap Confidence Intervals
    cls_metrics = compute_classification_metrics(y_true_arr, y_prob_arr, threshold=0.5)
    pairs = reconstruct_pairs(scored_items)
    pair_metrics = compute_paired_metrics(pairs, threshold=0.5)
    boot_ci = bootstrap_paired_confidence_intervals(pairs, n_iterations=1000, seed=42)

    auroc = cls_metrics["auroc"]
    ci_low, ci_high = boot_ci.get("auroc", (float("nan"), float("nan")))
    auprc = cls_metrics["auprc"]
    fpr_90 = cls_metrics["fpr_at_90_recall"]
    fpr_95 = cls_metrics["fpr_at_95_recall"]
    pc = pair_metrics["P-C"]
    pv = pair_metrics["P-V"]
    mean_p = float(np.mean(y_prob_arr))
    top1_pct = (top1_correct / len(samples)) * 100.0

    result_record = {
        "model": model_id,
        "dataset": "juliet",
        "split": split,
        "audit_status": "LEAKAGE_FREE",
        "samples_evaluated": len(samples),
        "pairs_evaluated": len(pairs),
        "auroc": auroc,
        "auroc_95_ci": [ci_low, ci_high],
        "auprc": auprc,
        "fpr_at_90_recall": fpr_90,
        "fpr_at_95_recall": fpr_95,
        "mean_p_vuln": mean_p,
        "pc": pc,
        "pv": pv,
        "top1_compliance_pct": top1_pct,
        "elapsed_seconds": elapsed,
    }

    out_json = os.path.join(output_dir, f"qwen7b_juliet_audited_fp16_{split}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2)

    # 5. Output Summary Report
    print(f"\n{'='*70}\n  LEAKAGE AUDIT RERUN COMPLETE: FP16 Qwen2.5-Coder-7B Juliet\n{'='*70}")
    print(f"  Evaluated Split    : {split} ({len(samples)} funcs, {len(pairs)} pairs)")
    print(f"  AUROC (95% CI)     : {auroc:.4f} ({ci_low:.3f}, {ci_high:.3f})")
    print(f"  AUPRC              : {auprc:.4f}")
    print(f"  FPR @ 90% Recall   : {fpr_90:.4f}")
    print(f"  FPR @ 95% Recall   : {fpr_95:.4f}")
    print(f"  P-C (%) / P-V (%)  : {pc:.1f}% / {pv:.1f}%")
    print(f"  Mean p_vuln        : {mean_p:.4f}")
    print(f"  Top-1 YES/NO Match : {top1_pct:.1f}%")
    print(f"  Elapsed Time       : {elapsed:.1f}s")
    print(f"  Saved JSON         : {out_json}")

    print("\n--- COMPARISON AGAINST PRE-AUDIT SIGNAL GATE (LEAKED) ---")
    print("  Pre-Audit (Leaked Juliet) : AUROC = 0.9751 (95% CI: 0.952, 0.994) | FPR@90% Rec = 0.0300")
    print(f"  Post-Audit (Clean Juliet) : AUROC = {auroc:.4f} (95% CI: {ci_low:.3f}, {ci_high:.3f}) | FPR@90% Rec = {fpr_90:.4f}")
    print(f"{'='*70}\n")

    return result_record


def main():
    parser = argparse.ArgumentParser(description="AQCAP Leakage Audit Rerun CLI")
    parser.add_argument("--model", default=QWEN_7B_MODEL, help="Model ID")
    parser.add_argument("--split", default="calib", choices=["calib", "test"], help="Juliet split to rerun")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--output_dir", default="results", help="Directory for output JSON")
    args = parser.parse_args()

    run_audit_eval(
        model_id=args.model,
        split=args.split,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
