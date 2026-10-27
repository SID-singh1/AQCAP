"""
AQCAP Layer Sensitivity Profiling Runner (RQ2).

Recomputes S_task, S_gen, and S_l2 on the calibration split of the target dataset (Juliet by default).
Dynamically outputs:
- results/sensitivity_{model_tag}_{dataset}.csv
- configs/{prefix}top_task.json
- configs/{prefix}top_gen.json
- configs/{prefix}bottom_task.json

Usage:
    python scripts/run_sensitivity.py --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset juliet
    python scripts/run_sensitivity.py --model microsoft/Phi-3-mini-4k-instruct --dataset juliet
    python scripts/run_sensitivity.py --model Qwen/Qwen2.5-Coder-3B-Instruct --dataset juliet
"""

import os
import sys
import json
import argparse

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aqcap.sensitivity import run_full_sensitivity_profile, generate_sensitivity_configs
from aqcap.data import load_dataset_split


def get_model_tag_and_prefix(model_name: str):
    name = model_name.lower()
    if "qwen" in name and "7b" in name:
        return "qwen7b", "qwen7b_"
    elif "qwen" in name and "3b" in name:
        return "qwen3b", "qwen3b_"
    elif "phi" in name:
        return "phi3", "phi3_"
    elif "llama" in name:
        return "llama8b", "llama8b_"
    else:
        clean = name.replace("/", "_").replace("-", "_")
        return clean, f"{clean}_"


def main():
    parser = argparse.ArgumentParser(description="AQCAP Layer Sensitivity Profiling CLI (RQ2)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="HuggingFace model ID or path")
    parser.add_argument("--dataset", default="juliet", choices=["juliet", "primevul_unpaired", "primevul_paired"],
                        help="Calibration dataset to profile sensitivity on")
    parser.add_argument("--output_csv", default=None, help="Output CSV path for sensitivity scores")
    parser.add_argument("--configs_dir", default="configs", help="Directory where dynamic configs are saved")
    parser.add_argument("--prefix", default=None, help="Custom prefix for dynamic configs (e.g. 'qwen3b_pvu_', 'qwen7b_hqq_2bit_')")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resumption and compute all layers from scratch")
    parser.add_argument("--perturbation", default="nf4", choices=["nf4", "hqq2", "2bit", "hqq3", "hqq4"],
                        help="Single-layer perturbation type: 'nf4' or 'hqq2'")
    parser.add_argument("--budget", type=int, default=None, help="Number of sensitive layers to preserve (defaults to 50% for INT8/NF4, 25% for HQQ)")
    parser.add_argument("--baseline_prec", default=None, help="Baseline precision for non-sensitive layers ('nf4' or 'hqq2')")
    parser.add_argument("--sensitive_prec", default=None, help="Precision for sensitive layers ('int8' or 'hqq4')")
    parser.add_argument("--device", default="cuda", help="Execution device (cuda, auto, or cpu)")
    parser.add_argument("--blocksize", type=int, default=64, help="NF4 blocksize for perturbation")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per prompt")
    args = parser.parse_args()

    # 1. Load calibration split
    print(f"\n[RUN_SENSITIVITY] Loading calibration data for dataset: {args.dataset}...")
    data_obj = load_dataset_split(dataset_name=args.dataset, split="calib", max_tokens=args.max_tokens)
    calib_samples = data_obj["samples"]
    print(f"[RUN_SENSITIVITY] Loaded {len(calib_samples)} calibration functions for {args.dataset}.")

    # 2. Determine model tag, prefix, and output CSV
    model_tag, default_prefix = get_model_tag_and_prefix(args.model)
    prefix = args.prefix if args.prefix is not None else default_prefix
    if args.output_csv is None:
        if args.perturbation in ("hqq2", "2bit"):
            args.output_csv = os.path.join("results", f"sensitivity_{model_tag}_hqq2_{args.dataset}.csv")
        else:
            args.output_csv = os.path.join("results", f"sensitivity_{model_tag}_{args.dataset}.csv")

    resume = not args.no_resume

    # 3. Run full sensitivity profile
    sensitivity_df = run_full_sensitivity_profile(
        model_id=args.model,
        calib_samples=calib_samples,
        output_csv_path=args.output_csv,
        device=args.device,
        nf4_blocksize=args.blocksize,
        max_tokens=args.max_tokens,
        resume=resume,
        perturbation=args.perturbation,
    )

    # 4. Generate dynamic configs (top_task, top_gen, bottom_task)
    num_layers = len(sensitivity_df)
    is_hqq = args.perturbation in ("hqq2", "2bit")
    default_budget = round(num_layers * 0.25) if is_hqq else (num_layers // 2)
    budget = args.budget if args.budget is not None else default_budget
    baseline_prec = args.baseline_prec or ("hqq2" if is_hqq else "nf4")
    sensitive_prec = args.sensitive_prec or ("hqq4" if is_hqq else "int8")

    generate_sensitivity_configs(
        sensitivity_df=sensitivity_df,
        configs_dir=args.configs_dir,
        num_layers=num_layers,
        int8_budget=budget,
        prefix=prefix,
        baseline_prec=baseline_prec,
        sensitive_prec=sensitive_prec,
    )

    # If Phi-3, also save without prefix for canonical ROADMAP names
    if model_tag == "phi3" and args.prefix is None:
        generate_sensitivity_configs(
            sensitivity_df=sensitivity_df,
            configs_dir=args.configs_dir,
            num_layers=num_layers,
            int8_budget=budget,
            prefix="",
            baseline_prec=baseline_prec,
            sensitive_prec=sensitive_prec,
        )

    print(f"\n[RUN_SENSITIVITY] SENSITIVITY PROFILING COMPLETE.")
    print(f"Results saved to: {args.output_csv}")
    print(f"Generated configs in: {args.configs_dir} (prefix='{prefix}', budget={budget} of {num_layers} layers in {sensitive_prec.upper()})")


if __name__ == "__main__":
    main()
