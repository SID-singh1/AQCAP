"""
AQCAP Smoke Test: Pre-flight Verification on Kaggle GPU.

Verifies:
1. Phi-3 FP16 zero-shot performance on first 20 calibration samples (10 pairs):
   - % top-1 token in YES/NO candidates
   - p_vuln distribution (min, 25%, median, mean, 75%, max, std)
   - Calibration split AUROC (paired AUROC and classification ROC-AUC)
2. Precision maps and memory footprint audit:
   - FP16 model weights_mb and peak memory
   - Uniform NF4 model weights_mb and compression ratio
   - Uniform INT8 model weights_mb and compression ratio

Usage:
    python scripts/smoke_test.py --device cuda
    python scripts/smoke_test.py --device cpu --max_samples 4  # for quick local testing
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import gc
import json
import argparse
from typing import Dict, List, Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aqcap.quant import load_model, memory_report, reset_peak_memory
from aqcap.scoring import build_prompt, get_target_token_ids, score_prompt
from aqcap.data import load_primevul_raw, ensure_primevul_data
from aqcap.metrics import compute_classification_metrics, compute_paired_metrics, reconstruct_pairs


def run_smoke_test(
    model_id: str = "microsoft/Phi-3-mini-4k-instruct",
    device: str = "cuda",
    num_samples: int = 20,
    splits_dir: str = "data/splits",
    data_dir: str = "data",
) -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("  AQCAP PRE-FLIGHT SMOKE TEST (Kaggle GPU Verification)")
    print("=" * 78)
    print(f"  Model       : {model_id}")
    print(f"  Target Device: {device} (CUDA available: {torch.cuda.is_available()})")
    print(f"  Calib Count : {num_samples} samples")
    print("=" * 78 + "\n")

    # -------------------------------------------------------------------------
    # 1. Load Calibration Samples
    # -------------------------------------------------------------------------
    split_path = os.path.join(splits_dir, "calib_split.json")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Missing {split_path}. Run aqcap/data.py first.")

    with open(split_path, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    raw_path = os.path.join(data_dir, split_meta.get("source_file", "primevul_calib_paired.jsonl"))
    if not os.path.isfile(raw_path):
        ensure_primevul_data(data_dir)

    raw_items = load_primevul_raw(raw_path)
    raw_map = {item["idx"]: item for item in raw_items}

    # Extract exactly first num_samples in matched pairs
    ordered_samples: List[Dict[str, Any]] = []
    pairs_needed = (num_samples + 1) // 2
    for p in split_meta["pairs"][:pairs_needed]:
        v_idx = p["vuln_idx"]
        s_idx = p["safe_idx"]
        if v_idx in raw_map and s_idx in raw_map:
            v_item = dict(raw_map[v_idx])
            v_item["pair_id"] = p["pair_id"]
            s_item = dict(raw_map[s_idx])
            s_item["pair_id"] = p["pair_id"]
            ordered_samples.extend([v_item, s_item])

    ordered_samples = ordered_samples[:num_samples]
    print(f"[1/4] Loaded {len(ordered_samples)} calib functions ({len(ordered_samples)//2} paired functions).")

    # -------------------------------------------------------------------------
    # 2. Load FP16 Model & Score Calibration Samples
    # -------------------------------------------------------------------------
    print(f"\n[2/4] Loading FP16 baseline model on {device}...")
    reset_peak_memory()
    fp16_map = {i: "fp16" for i in range(32)}  # 32 layers for Phi-3
    model_fp16, tokenizer = load_model(
        model_id=model_id,
        precision_map=fp16_map,
        device=device,
    )
    yes_ids, no_ids = get_target_token_ids(tokenizer)
    fp16_mem = memory_report(model_fp16)
    print(f"      FP16 Model weights: {fp16_mem['weights_mb']:.2f} MB")
    print(f"      YES token IDs: {yes_ids} | NO token IDs: {no_ids}")

    print(f"\n      Scoring {len(ordered_samples)} samples...")
    results = []
    top1_in_yes_no_count = 0
    p_vuln_list = []
    labels_list = []

    for idx, item in enumerate(ordered_samples, start=1):
        prompt = build_prompt(tokenizer, item["func"])
        score_res = score_prompt(
            model=model_fp16,
            tokenizer=tokenizer,
            prompt=prompt,
            yes_ids=yes_ids,
            no_ids=no_ids,
        )
        is_in_yn = score_res["is_in_yes_no"]
        if is_in_yn:
            top1_in_yes_no_count += 1

        p_v = score_res["p_vuln"]
        p_vuln_list.append(p_v)
        labels_list.append(item["target"])

        results.append({
            "idx": item["idx"],
            "pair_id": item["pair_id"],
            "target": item["target"],
            "p_vuln": p_v,
            "top1_token": score_res["top1_token"],
            "is_in_yes_no": is_in_yn,
            "latency_s": score_res["latency_s"],
        })

        print(f"      [{idx:02d}/{len(ordered_samples)}] ID={item['idx']} Target={item['target']} "
              f"p_vuln={p_v:.4f} Top1='{score_res['top1_token']}' (In Y/N: {is_in_yn})")

    pct_top1_yn = (top1_in_yes_no_count / len(ordered_samples)) * 100.0
    p_vuln_arr = np.array(p_vuln_list)
    labels_arr = np.array(labels_list)

    # Free FP16 model to conserve GPU VRAM for precision audits
    del model_fp16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 3. Compute Metrics on 20 Calib Samples
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("  CALIBRATION SAMPLE METRICS SUMMARY")
    print("-" * 78)
    print(f"  Total Samples Evaluated        : {len(ordered_samples)}")
    print(f"  Top-1 Token in YES/NO          : {top1_in_yes_no_count}/{len(ordered_samples)} ({pct_top1_yn:.1f}%)")
    print(f"\n  p_vuln Distribution (Overall):")
    print(f"    Min   : {np.min(p_vuln_arr):.4f}")
    print(f"    25%   : {np.percentile(p_vuln_arr, 25):.4f}")
    print(f"    Median: {np.median(p_vuln_arr):.4f}")
    print(f"    Mean  : {np.mean(p_vuln_arr):.4f} +/- {np.std(p_vuln_arr):.4f}")
    print(f"    75%   : {np.percentile(p_vuln_arr, 75):.4f}")
    print(f"    Max   : {np.max(p_vuln_arr):.4f}")

    if np.any(labels_arr == 1) and np.any(labels_arr == 0):
        vuln_p = p_vuln_arr[labels_arr == 1]
        safe_p = p_vuln_arr[labels_arr == 0]
        print(f"\n  p_vuln by Class:")
        print(f"    Vulnerable (target=1, n={len(vuln_p)}): mean={np.mean(vuln_p):.4f}, median={np.median(vuln_p):.4f}")
        print(f"    Safe       (target=0, n={len(safe_p)}): mean={np.mean(safe_p):.4f}, median={np.median(safe_p):.4f}")

    # Standard classification metrics
    class_metrics = compute_classification_metrics(labels_list, p_vuln_list)
    auroc = class_metrics.get("auroc", float("nan"))
    print(f"\n  Calib Classification AUROC     : {auroc:.4f}")

    # Paired metrics
    pairs = reconstruct_pairs(results)
    pair_metrics = {}
    pairwise_acc = 0.0
    if pairs:
        pair_metrics = compute_paired_metrics(pairs)
        n_pairs = pair_metrics.get("n_pairs", len(pairs))
        pair_rank_correct = sum(1 for p in pairs if p["p_vuln_for_vuln"] > p["p_vuln_for_safe"])
        pair_rank_ties = sum(1 for p in pairs if p["p_vuln_for_vuln"] == p["p_vuln_for_safe"])
        pairwise_acc = (pair_rank_correct + 0.5 * pair_rank_ties) / len(pairs) * 100.0 if pairs else 0.0

        print(f"  Calib Paired Metrics ({n_pairs} pairs):")
        print(f"    Pairwise Ranking Acc (p_v > p_s): {pairwise_acc:.1f}% ({pair_rank_correct}/{n_pairs} strictly greater)")
        print(f"    P-C (Strict Pair Accuracy)     : {pair_metrics.get('P-C', 0.0):.1f}%")
        print(f"    P-V (Both Predicted Vuln)      : {pair_metrics.get('P-V', 0.0):.1f}%")
        print(f"    P-B (Both Predicted Benign)    : {pair_metrics.get('P-B', 0.0):.1f}%")
        print(f"    P-R (Reversed Classification)  : {pair_metrics.get('P-R', 0.0):.1f}%")

    # -------------------------------------------------------------------------
    # 4. Precision Maps & Weight Footprint Audit (NF4 and INT8)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("  PRECISION MAPS & MEMORY FOOTPRINT AUDIT")
    print("-" * 78)
    num_layers = 32 if "phi" in model_id.lower() else 36

    print(f"  [FP16 Baseline] Weights: {fp16_mem['weights_mb']:.2f} MB")

    nf4_weights_mb = None
    int8_weights_mb = None

    if device == "cuda" and torch.cuda.is_available():
        # NF4 Map
        print(f"\n  Loading Uniform NF4 precision map (all {num_layers} layers in NF4)...")
        reset_peak_memory()
        nf4_map = {i: "nf4" for i in range(num_layers)}
        model_nf4, _ = load_model(model_id=model_id, precision_map=nf4_map, device=device)
        nf4_mem = memory_report(model_nf4)
        nf4_weights_mb = nf4_mem["weights_mb"]
        print(f"  [NF4 Map]       Weights: {nf4_weights_mb:.2f} MB "
              f"(Compression: {fp16_mem['weights_mb']/nf4_weights_mb:.2f}x, Peak: {nf4_mem['peak_mb']:.2f} MB)")
        del model_nf4
        gc.collect()
        torch.cuda.empty_cache()

        # INT8 Map
        print(f"\n  Loading Uniform INT8 precision map (all {num_layers} layers in INT8)...")
        reset_peak_memory()
        int8_map = {i: "int8" for i in range(num_layers)}
        model_int8, _ = load_model(model_id=model_id, precision_map=int8_map, device=device)
        int8_mem = memory_report(model_int8)
        int8_weights_mb = int8_mem["weights_mb"]
        print(f"  [INT8 Map]      Weights: {int8_weights_mb:.2f} MB "
              f"(Compression: {fp16_mem['weights_mb']/int8_weights_mb:.2f}x, Peak: {int8_mem['peak_mb']:.2f} MB)")
        del model_int8
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("  [NOTE] Skipping CUDA-specific bitsandbytes materialization on CPU device.")

    summary = {
        "model": model_id,
        "device": device,
        "num_samples": len(ordered_samples),
        "pct_top1_in_yes_no": pct_top1_yn,
        "p_vuln_min": float(np.min(p_vuln_arr)),
        "p_vuln_mean": float(np.mean(p_vuln_arr)),
        "p_vuln_median": float(np.median(p_vuln_arr)),
        "p_vuln_max": float(np.max(p_vuln_arr)),
        "calib_auroc": auroc,
        "fp16_weights_mb": fp16_mem["weights_mb"],
        "nf4_weights_mb": nf4_weights_mb,
        "int8_weights_mb": int8_weights_mb,
    }
    if pairs:
        summary["pairwise_ranking_acc"] = pairwise_acc
        summary["pair_p_c"] = pair_metrics.get("P-C")
        summary["pair_p_v"] = pair_metrics.get("P-V")
        summary["pair_p_b"] = pair_metrics.get("P-B")
        summary["pair_p_r"] = pair_metrics.get("P-R")

    print("\n" + "=" * 78)
    print("  PRE-FLIGHT SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 78 + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="AQCAP Smoke Test")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct", help="Model ID")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_samples", type=int, default=20, help="Number of calibration samples (default: 20)")
    parser.add_argument("--splits_dir", default="data/splits", help="Splits directory")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    args = parser.parse_args()

    run_smoke_test(
        model_id=args.model,
        device=args.device,
        num_samples=args.max_samples,
        splits_dir=args.splits_dir,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
