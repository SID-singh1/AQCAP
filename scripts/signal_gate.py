"""
AQCAP Signal Gate Evaluator.

Evaluates unquantized FP16 models on calibration data to identify which models and datasets
have legitimate vulnerability detection signal before quantization experiments.

Evaluates:
  Models:
    - Qwen/Qwen2.5-Coder-7B-Instruct
    - Qwen/Qwen2.5-Coder-3B-Instruct
    - meta-llama/Llama-3.1-8B-Instruct
    - microsoft/Phi-3-mini-4k-instruct
  Datasets:
    - primevul_paired
    - primevul_unpaired
    - juliet
  Prompt Modes:
    - zero_shot (current frozen instruction prompt)
    - 2_shot (shortest calib pair demonstration, excluded from calib scoring)

Strict Constraints:
  - FP16 only.
  - Calibration split ONLY (Do NOT run test split).
  - Target token ID filtering (no overlapping or pure-whitespace tokens).
  - Metrics: AUROC with 95% bootstrap CI, mean p_vuln, P-C/P-V (for paired),
             FPR@90% recall, FPR@95% recall, AUPRC, PR curve data.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import gc
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from aqcap.quant import load_model, memory_report, reset_peak_memory
from aqcap.scoring import build_prompt, get_target_token_ids, score_prompt
from aqcap.data import (
    load_dataset_split,
    get_shortest_calib_demonstration,
    DEFAULT_DATA_DIR,
    DEFAULT_SPLITS_DIR,
)
from aqcap.metrics import (
    compute_classification_metrics,
    compute_paired_metrics,
    reconstruct_pairs,
    compute_pr_curve_data,
    bootstrap_paired_confidence_intervals,
    bootstrap_unpaired_confidence_intervals,
)

# Canonical model aliases and identifiers
MODEL_REGISTRY = {
    "qwen-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder-7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen/qwen2.5-coder-7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",

    "qwen-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen2.5-coder-3b-instruct": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen/qwen2.5-coder-3b-instruct": "Qwen/Qwen2.5-Coder-3B-Instruct",

    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",

    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
}

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
]

DEFAULT_DATASETS = [
    "primevul_paired",
    "primevul_unpaired",
    "juliet",
]

DEFAULT_MODES = [
    "zero_shot",
    "2_shot",
]


def resolve_model_id(name: str) -> str:
    """Resolves short names or aliases to canonical Hugging Face repository IDs."""
    key = name.strip().lower()
    return MODEL_REGISTRY.get(key, name.strip())


def sanitize_slug(name: str) -> str:
    """Creates a filesystem-safe identifier."""
    return name.replace("/", "__").replace("\\", "_").replace(":", "_").replace("-", "_").lower()


def evaluate_gate_cell(
    model,
    tokenizer,
    model_id: str,
    dataset_name: str,
    mode: str,
    output_dir: str,
    data_dir: str = DEFAULT_DATA_DIR,
    splits_dir: str = DEFAULT_SPLITS_DIR,
    max_tokens: int = 2048,
    yes_ids: Optional[List[int]] = None,
    no_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Evaluates one (model, dataset, mode) gate cell on the calibration split in FP16.
    """
    print(f"\n{'-'*65}")
    print(f"  SIGNAL GATE CELL: {model_id} | {dataset_name} | {mode}")
    print(f"{'-'*65}")

    # 1. Target Token IDs
    if yes_ids is None or no_ids is None:
        yes_ids, no_ids = get_target_token_ids(tokenizer)

    # 2. Shortest demonstration handling for 2-shot
    demonstration_info = None
    few_shot_examples = None
    excluded_sample_indices = set()
    excluded_pair_id = None

    if mode in ("2_shot", "few_shot", "2shot"):
        demo = get_shortest_calib_demonstration(dataset_name, data_dir=data_dir, splits_dir=splits_dir)
        demonstration_info = {
            "pair_id": demo.get("pair_id"),
            "vuln_idx": demo["vuln_idx"],
            "safe_idx": demo["safe_idx"],
            "vuln_code_len": len(demo["vuln_code"]),
            "safe_code_len": len(demo["safe_code"]),
            "cwe": demo.get("cwe", "Unknown"),
            "project": demo.get("project", "Unknown"),
        }
        few_shot_examples = [
            {"code": demo["vuln_code"], "label": "YES"},
            {"code": demo["safe_code"], "label": "NO"},
        ]
        excluded_sample_indices.add(demo["vuln_idx"])
        excluded_sample_indices.add(demo["safe_idx"])
        excluded_pair_id = demo.get("pair_id")
        print(f"  [2-SHOT] Fixed Demonstration: Vuln idx={demo['vuln_idx']}, Safe idx={demo['safe_idx']}")
        print(f"           Excluded from scoring to prevent demonstration leakage.")

    # 3. Load dataset split
    data_obj = load_dataset_split(
        dataset_name=dataset_name,
        split="calib",
        data_dir=data_dir,
        splits_dir=splits_dir,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )

    is_paired = data_obj["is_paired"]
    raw_samples = data_obj["samples"]

    prompt_mode = "2_shot" if mode in ("2_shot", "few_shot", "2shot") else "zero"

    from collections import defaultdict
    eval_items = []

    if is_paired:
        pair_dict = defaultdict(dict)
        for s in raw_samples:
            pid = s.get("pair_id")
            if not pid:
                continue
            if s.get("target") == 1:
                pair_dict[pid]["vuln"] = s
            elif s.get("target") == 0:
                pair_dict[pid]["safe"] = s

        dropped_pairs = 0
        for pid, pair in pair_dict.items():
            if "vuln" not in pair or "safe" not in pair:
                continue
            if mode in ("2_shot", "few_shot", "2shot"):
                if pid == excluded_pair_id or pair["vuln"]["idx"] in excluded_sample_indices or pair["safe"]["idx"] in excluded_sample_indices:
                    continue

            p_v = build_prompt(tokenizer, pair["vuln"]["func"], mode=prompt_mode, few_shot_examples=few_shot_examples)
            p_s = build_prompt(tokenizer, pair["safe"]["func"], mode=prompt_mode, few_shot_examples=few_shot_examples)
            tok_v = tokenizer(p_v, return_tensors="pt", truncation=False)["input_ids"].shape[1]
            tok_s = tokenizer(p_s, return_tensors="pt", truncation=False)["input_ids"].shape[1]

            if tok_v <= max_tokens and tok_s <= max_tokens:
                eval_items.append((pair["vuln"], p_v))
                eval_items.append((pair["safe"], p_s))
            else:
                dropped_pairs += 1

        if dropped_pairs > 0:
            print(f"  [GATE] Filtered {dropped_pairs} pairs exceeding max_tokens ({max_tokens}) with prompt template.")
    else:
        dropped_samples = 0
        for s in raw_samples:
            if mode in ("2_shot", "few_shot", "2shot"):
                if s["idx"] in excluded_sample_indices:
                    continue

            p = build_prompt(tokenizer, s["func"], mode=prompt_mode, few_shot_examples=few_shot_examples)
            tok = tokenizer(p, return_tensors="pt", truncation=False)["input_ids"].shape[1]
            if tok <= max_tokens:
                eval_items.append((s, p))
            else:
                dropped_samples += 1

        if dropped_samples > 0:
            print(f"  [GATE] Filtered {dropped_samples} samples exceeding max_tokens ({max_tokens}) with prompt template.")

    eval_samples = [it[0] for it in eval_items]
    print(f"  [GATE] Evaluating {len(eval_samples)} calibration samples (is_paired={is_paired})...")

    # 4. Scoring Loop
    t_start = time.time()
    scored_records = []
    y_true_list = []
    y_prob_list = []
    top1_correct = 0

    for i, (s, prompt_str) in enumerate(eval_items):
        code = s["func"]
        target = s["target"]
        pair_id = s.get("pair_id", f"sample_{s['idx']}")

        res = score_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_str,
            yes_ids=yes_ids,
            no_ids=no_ids,
            max_tokens=max_tokens,
        )

        p_vuln = res["p_vuln"]
        p_safe = res["p_safe"]
        pred_50 = res["pred_50"]
        is_in_yes_no = res["is_in_yes_no"]
        top1_token = res["top1_token"]

        if is_in_yes_no:
            top1_correct += 1

        y_true_list.append(target)
        y_prob_list.append(p_vuln)

        scored_records.append({
            "sample_index": i,
            "idx": s["idx"],
            "pair_id": pair_id,
            "target": target,
            "p_vuln": p_vuln,
            "p_safe": p_safe,
            "pred_50": pred_50,
            "top1_token": top1_token,
            "is_in_yes_no": is_in_yes_no,
            "n_input_tokens": res["n_input_tokens"],
            "latency_s": res["latency_s"],
            "cwe": s.get("cwe", "Unknown"),
            "project": s.get("project", "Unknown"),
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(eval_samples):
            status = "✔" if pred_50 == target else "✖"
            print(f"    [{i+1:>3}/{len(eval_samples)}] {status} True:{target} Pred:{pred_50} "
                  f"(p_vuln={p_vuln:.3f}, top1='{top1_token}') | {res['latency_s']:.2f}s")

    elapsed_time = time.time() - t_start
    y_true_arr = np.array(y_true_list, dtype=int)
    y_prob_arr = np.array(y_prob_list, dtype=float)

    # 5. Metrics Computation
    cls_metrics = compute_classification_metrics(y_true_arr, y_prob_arr, threshold=0.5)
    pr_curve_info = compute_pr_curve_data(y_true_arr, y_prob_arr)
    mean_p_vuln = float(np.mean(y_prob_arr))

    # Bootstrap 95% Confidence Intervals & Paired Metrics
    if is_paired:
        pairs_data = reconstruct_pairs(scored_records)
        boot_ci = bootstrap_paired_confidence_intervals(pairs_data, n_iterations=1000, seed=42)
        paired_m = compute_paired_metrics(pairs_data, threshold=0.5)
        pc = paired_m["P-C"]
        pv = paired_m["P-V"]
        pb = paired_m["P-B"]
        pr = paired_m["P-R"]
        n_pairs = paired_m["n_pairs"]
    else:
        boot_ci = bootstrap_unpaired_confidence_intervals(y_true_arr, y_prob_arr, n_iterations=1000, seed=42)
        pc = None
        pv = None
        pb = None
        pr = None
        n_pairs = None

    auroc = cls_metrics["auroc"]
    auroc_low, auroc_high = boot_ci["auroc"]
    auprc = cls_metrics["auprc"]
    fpr_90 = cls_metrics["fpr_at_90_recall"]
    fpr_95 = cls_metrics["fpr_at_95_recall"]
    top1_compliance = (top1_correct / len(eval_samples)) * 100.0 if eval_samples else 0.0

    print(f"  [RESULT] AUROC: {auroc:.4f} (95% CI: {auroc_low:.4f} - {auroc_high:.4f})")
    print(f"           AUPRC: {auprc:.4f} | Mean p_vuln: {mean_p_vuln:.4f}")
    if is_paired:
        print(f"           P-C: {pc:.1f}% | P-V: {pv:.1f}% | P-B: {pb:.1f}% | P-R: {pr:.1f}% ({n_pairs} pairs)")
    print(f"           FPR@90% Recall: {fpr_90:.4f} | FPR@95% Recall: {fpr_95:.4f}")
    print(f"           Top-1 YES/NO Match: {top1_compliance:.1f}%")

    result = {
        "model": model_id,
        "dataset": dataset_name,
        "mode": mode,
        "precision": "fp16",
        "split": "calib",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(eval_samples),
        "is_paired": is_paired,
        "n_pairs": n_pairs,
        "metrics": {
            "auroc": auroc,
            "auroc_ci_low": auroc_low,
            "auroc_ci_high": auroc_high,
            "auprc": auprc,
            "mean_p_vuln": mean_p_vuln,
            "balanced_accuracy": cls_metrics["balanced_accuracy"],
            "f1": cls_metrics["f1"],
            "mcc": cls_metrics["mcc"],
            "fpr_at_90_recall": fpr_90,
            "fpr_at_95_recall": fpr_95,
            "top1_compliance_pct": top1_compliance,
            "pc": pc,
            "pv": pv,
            "pb": pb,
            "pr": pr,
        },
        "target_token_ids": {
            "yes_ids": yes_ids,
            "no_ids": no_ids,
        },
        "demonstration_info": demonstration_info,
        "pr_curve": pr_curve_info,
        "elapsed_seconds": elapsed_time,
        "samples": scored_records,
    }

    # Save cell result to disk
    os.makedirs(output_dir, exist_ok=True)
    slug = f"{sanitize_slug(model_id)}_{dataset_name}_{mode}"
    cell_path = os.path.join(output_dir, f"{slug}.json")
    with open(cell_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  [SAVED] {cell_path}")

    return result


def render_gate_table(results: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Renders the consolidated gate table in Markdown and CSV formats.
    """
    headers = [
        "Model",
        "Dataset",
        "Prompt Mode",
        "AUROC (95% CI)",
        "AUPRC",
        "Mean p_vuln",
        "P-C (%)",
        "P-V (%)",
        "FPR@90% Rec",
        "FPR@95% Rec",
        "Top-1 (%)",
    ]

    rows = []
    csv_rows = [",".join(headers)]

    for r in results:
        m = r["metrics"]
        model_name = r["model"].split("/")[-1]
        ds = r["dataset"]
        mode = r["mode"]

        ci_low = m.get("auroc_ci_low")
        ci_high = m.get("auroc_ci_high")
        auroc = m.get("auroc", float("nan"))

        if not np.isnan(auroc) and ci_low is not None and not np.isnan(ci_low):
            auroc_str = f"{auroc:.4f} ({ci_low:.3f}, {ci_high:.3f})"
        elif not np.isnan(auroc):
            auroc_str = f"{auroc:.4f}"
        else:
            auroc_str = "N/A"

        auprc_str = f"{m.get('auprc', float('nan')):.4f}"
        mean_p_str = f"{m.get('mean_p_vuln', float('nan')):.4f}"
        pc_str = f"{m['pc']:.1f}" if m.get("pc") is not None else "—"
        pv_str = f"{m['pv']:.1f}" if m.get("pv") is not None else "—"
        fpr90_str = f"{m.get('fpr_at_90_recall', float('nan')):.4f}"
        fpr95_str = f"{m.get('fpr_at_95_recall', float('nan')):.4f}"
        top1_str = f"{m.get('top1_compliance_pct', 0.0):.1f}"

        row = [
            model_name,
            ds,
            mode,
            auroc_str,
            auprc_str,
            mean_p_str,
            pc_str,
            pv_str,
            fpr90_str,
            fpr95_str,
            top1_str,
        ]
        rows.append(row)
        # CSV escape
        csv_row = [f'"{val}"' if "," in val else val for val in row]
        csv_rows.append(",".join(csv_row))

    # Format Markdown
    col_widths = [max(len(h), max(len(r[i]) for r in rows) if rows else len(h)) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    body_lines = [
        "| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    ]
    md_table = "\n".join([header_line, sep_line] + body_lines)
    csv_table = "\n".join(csv_rows)

    return md_table, csv_table


def main():
    parser = argparse.ArgumentParser(description="AQCAP Signal Gate Evaluator")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of model names or aliases")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="List of datasets")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="List of prompt modes (zero_shot, 2_shot)")
    parser.add_argument("--output_dir", default="results/signal_gate", help="Output directory for gate results")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Path to raw datasets")
    parser.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR, help="Path to splits metadata")
    parser.add_argument("--device", default="auto", help="Device for execution (auto, cuda, cpu)")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max allowed prompt tokens")
    parser.add_argument("--force_rerun", action="store_true", help="Force re-evaluation even if results exist")
    parser.add_argument("--table_only", action="store_true", help="Render gate table from existing JSONs without running models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    resolved_models = [resolve_model_id(m) for m in args.models]

    print("\n" + "=" * 70)
    print("  AQCAP PRE-QUANTIZATION SIGNAL GATE (FP16 CALIBRATION ONLY)")
    print("=" * 70)
    print(f"  Models   : {resolved_models}")
    print(f"  Datasets : {args.datasets}")
    print(f"  Modes    : {args.modes}")
    print(f"  Split    : CALIB ONLY (Strictly no test split)")
    print(f"  Precision: FP16 unquantized reference")
    print(f"  Device   : {args.device}")
    print(f"  Output   : {args.output_dir}")
    print("=" * 70 + "\n")

    all_results = []

    # If table_only, load existing JSON files and render table
    if args.table_only:
        for model_id in resolved_models:
            for dataset_name in args.datasets:
                for mode in args.modes:
                    slug = f"{sanitize_slug(model_id)}_{dataset_name}_{mode}"
                    cell_path = os.path.join(args.output_dir, f"{slug}.json")
                    if os.path.isfile(cell_path):
                        with open(cell_path, "r", encoding="utf-8") as f:
                            all_results.append(json.load(f))
                    else:
                        print(f"[WARN] Result not found: {cell_path}")

        md_tbl, csv_tbl = render_gate_table(all_results)
        print("\n" + "=" * 70)
        print("  SIGNAL GATE EVALUATION TABLE")
        print("=" * 70)
        print(md_tbl)
        print("=" * 70 + "\n")
        return

    # Normal execution: group by model to minimize reloading overhead
    for model_id in resolved_models:
        # Check if all cells for this model already exist
        cells_needed = []
        for dataset_name in args.datasets:
            for mode in args.modes:
                slug = f"{sanitize_slug(model_id)}_{dataset_name}_{mode}"
                cell_path = os.path.join(args.output_dir, f"{slug}.json")
                if not args.force_rerun and os.path.isfile(cell_path):
                    try:
                        with open(cell_path, "r", encoding="utf-8") as f:
                            cached = json.load(f)
                        all_results.append(cached)
                        print(f"[CACHE] Loaded cached gate cell: {slug}")
                        continue
                    except Exception as e:
                        print(f"[WARN] Failed to read cached {cell_path}: {e}")
                cells_needed.append((dataset_name, mode))

        if not cells_needed:
            print(f"[SKIP] All gate cells for {model_id} already evaluated.")
            continue

        print(f"\n{'#'*70}")
        print(f"  LOADING MODEL: {model_id} (FP16 unquantized reference)")
        print(f"{'#'*70}")

        reset_peak_memory()
        # Empty precision_map means pure FP16 unquantized base model
        model, tokenizer = load_model(
            model_id=model_id,
            precision_map={},
            device=args.device,
        )

        yes_ids, no_ids = get_target_token_ids(tokenizer)
        print(f"[GATE] Target Token IDs for {model_id}:")
        print(f"       YES ids: {yes_ids} -> {[tokenizer.decode([i]) for i in yes_ids]}")
        print(f"       NO  ids: {no_ids} -> {[tokenizer.decode([i]) for i in no_ids]}")

        # Run each required gate cell
        for dataset_name, mode in cells_needed:
            res = evaluate_gate_cell(
                model=model,
                tokenizer=tokenizer,
                model_id=model_id,
                dataset_name=dataset_name,
                mode=mode,
                output_dir=args.output_dir,
                data_dir=args.data_dir,
                splits_dir=args.splits_dir,
                max_tokens=args.max_tokens,
                yes_ids=yes_ids,
                no_ids=no_ids,
            )
            all_results.append(res)

        # Cleanup model to free VRAM for next model
        print(f"\n[CLEANUP] Unloading {model_id} and freeing memory...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Render and save gate table
    md_tbl, csv_tbl = render_gate_table(all_results)

    table_md_path = os.path.join(args.output_dir, "gate_table.md")
    table_csv_path = os.path.join(args.output_dir, "gate_table.csv")

    with open(table_md_path, "w", encoding="utf-8") as f:
        f.write(md_tbl + "\n")

    with open(table_csv_path, "w", encoding="utf-8") as f:
        f.write(csv_tbl + "\n")

    print("\n" + "=" * 70)
    print("  SIGNAL GATE EVALUATION TABLE")
    print("=" * 70)
    print(md_tbl)
    print(f"\nSaved Markdown table -> {table_md_path}")
    print(f"Saved CSV table      -> {table_csv_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
