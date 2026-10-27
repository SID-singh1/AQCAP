"""
AQCAP Kaggle GPU Execution Launcher (Main Study & Leakage Audit).

Automates the empirical evaluation pipeline across target models and datasets:
1. Stage 'rerun_audit': Rerun FP16 Qwen2.5-Coder-7B on audited Juliet calib & report AUROC.
2. Stage 'sensitivity': Recomputes S_task, S_gen, and S_l2 on Juliet calib per model.
3. Stage 'qwen7b_juliet': Main study primary evaluations on Juliet test (1000 funcs, budget=14 INT8/14 NF4).
4. Stage 'phi3_juliet': Main study secondary evaluations on Juliet test (1000 funcs, budget=16 INT8/16 NF4).
5. Stage 'qwen3b_juliet': Main study tertiary evaluations on Juliet test (1000 funcs, budget=18 INT8/18 NF4).
6. Stage 'qwen3b_primevul': Secondary evaluations on PrimeVul unpaired test (1000 funcs, fp16/nf4/int8/top_task).
7. Stage 'analyze': Statistical analysis, bootstrap CIs, McNemar tests vs FP16, LaTeX tables.
8. Stage 'package': Zip results, paper, and configs into aqcap_results.zip.

Usage:
    python scripts/kaggle_eval_launcher.py --stage rerun_audit
    python scripts/kaggle_eval_launcher.py --stage qwen7b_sensitivity
    python scripts/kaggle_eval_launcher.py --stage qwen7b_juliet
    python scripts/kaggle_eval_launcher.py --stage all_main
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import shutil
import zipfile
import argparse
import subprocess
from typing import List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QWEN_7B_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
PHI3_MODEL = "microsoft/Phi-3-mini-4k-instruct"
QWEN_3B_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
LLAMA_8B_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Cross-family generalisation: Llama-3.1-8B on Juliet (uniform bit-width sweep only)
LLAMA8B_CONFIGS = [
    "llama8b_fp16", "llama8b_nf4", "llama8b_int8",
    "llama8b_hqq_4bit", "llama8b_hqq_3bit", "llama8b_hqq_2bit"
]

# Primary configs for Qwen-7B on Juliet (28 layers, 14 INT8 / 14 NF4)
QWEN7B_CONFIGS = [
    "qwen7b_fp16", "qwen7b_nf4", "qwen7b_int8",
    "qwen7b_top_task", "qwen7b_top_gen", "qwen7b_bottom_task",
    "qwen7b_first14", "qwen7b_last14",
    "qwen7b_rand_s0", "qwen7b_rand_s1", "qwen7b_rand_s2"
]

# Secondary configs for Phi-3-mini on Juliet (32 layers, 16 INT8 / 16 NF4)
PHI3_CONFIGS = [
    "fp16", "nf4", "int8",
    "top_task", "top_gen", "bottom_task",
    "first16", "last16",
    "rand_s0", "rand_s1", "rand_s2"
]

# Tertiary configs for Qwen-3B on Juliet (36 layers, 18 INT8 / 18 NF4)
QWEN3B_JULIET_CONFIGS = [
    "qwen3b_fp16", "qwen3b_nf4", "qwen3b_int8",
    "qwen3b_top_task", "qwen3b_top_gen", "qwen3b_bottom_task",
    "qwen3b_first18", "qwen3b_last18",
    "qwen3b_rand_s0", "qwen3b_rand_s1", "qwen3b_rand_s2"
]

# Secondary study configs for Qwen-3B on PrimeVul unpaired (1000 funcs)
QWEN3B_PRIMEVUL_CONFIGS = [
    "qwen3b_fp16", "qwen3b_nf4", "qwen3b_int8", "qwen3b_pvu_top_task"
]

# Natural code generalization configs for Qwen-7B on PrimeVul unpaired (Item D)
QWEN7B_PRIMEVUL_CONFIGS = [
    "qwen7b_fp16", "qwen7b_nf4", "qwen7b_int8"
]

# Low-bit HQQ sweep on Qwen-7B Juliet test (Item C)
QWEN7B_HQQ_CONFIGS = [
    "qwen7b_hqq_4bit", "qwen7b_hqq_3bit", "qwen7b_hqq_2bit",
    "qwen7b_hqq_2bit_top_task", "qwen7b_hqq_2bit_top_gen",
    "qwen7b_hqq_2bit_first7", "qwen7b_hqq_2bit_last7",
    "qwen7b_hqq_2bit_rand_s0", "qwen7b_hqq_2bit_rand_s1", "qwen7b_hqq_2bit_rand_s2"
]


def run_cmd(cmd: List[str]):
    """Executes a command and ensures it succeeds."""
    resolved_cmd = list(cmd)
    if resolved_cmd and resolved_cmd[0] == "python":
        resolved_cmd[0] = sys.executable
    cmd_str = " ".join(resolved_cmd)
    print(f"\n[KAGGLE_EXEC] >>> {cmd_str}")
    subprocess.check_call(resolved_cmd)


def ensure_dependencies():
    """Ensures critical packages like bitsandbytes and hqq are installed and importable."""
    print("\n" + "=" * 70)
    print("  VERIFYING RUNTIME DEPENDENCIES")
    print("=" * 70)
    try:
        import bitsandbytes as bnb
        print(f"[SUCCESS] bitsandbytes is already installed (version {getattr(bnb, '__version__', 'unknown')}).")
    except Exception as e:
        print(f"[LAUNCHER] bitsandbytes not available ({e}). Installing via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "bitsandbytes"])
            import importlib
            importlib.invalidate_caches()
            import bitsandbytes as bnb
            print(f"[SUCCESS] bitsandbytes installed and verified (version {getattr(bnb, '__version__', 'unknown')}).")
        except Exception as install_err:
            print(f"[WARNING] Auto-install of bitsandbytes failed: {install_err}")

    try:
        import hqq
        print(f"[SUCCESS] hqq is already installed.")
    except Exception as e:
        print(f"[LAUNCHER] hqq not available ({e}). Installing via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "hqq"])
            import importlib
            importlib.invalidate_caches()
            import hqq
            print(f"[SUCCESS] hqq installed and verified.")
        except Exception as install_err:
            print(f"[WARNING] Auto-install of hqq failed: {install_err}")


def setup_environment():
    """Checks GPU availability and authenticates with Hugging Face via Kaggle Secrets."""
    print("\n" + "=" * 70)
    print("  AQCAP KAGGLE RUNTIME ENVIRONMENT DIAGNOSTICS")
    print("=" * 70)

    try:
        gpu_info = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
        print(gpu_info)
    except Exception as e:
        print(f"[WARNING] nvidia-smi failed: {e}. GPU might not be attached.")

    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        hf_token = secrets.get_secret("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            print("[SUCCESS] Authenticated with Hugging Face Hub using Kaggle Secrets.")
        else:
            print("[INFO] No HF_TOKEN secret found in Kaggle Secrets.")
    except Exception:
        print("[INFO] Not running in Kaggle notebook with Kaggle Secrets or token already set.")


def run_rerun_audit():
    """Reruns FP16 Qwen2.5-Coder-7B zero-shot on the audited Juliet calib split and reports AUROC."""
    print("\n" + "=" * 70)
    print("  STAGE 1: LEAKAGE AUDIT RERUN (FP16 Qwen2.5-Coder-7B on Audited Juliet)")
    print("=" * 70)
    run_cmd([
        "python", "scripts/rerun_qwen7b_juliet_audit.py",
        "--device", "auto",
        "--split", "calib",
    ])


def run_sensitivity_profiling(model_id: str, dataset: str = "juliet"):
    """Profiles layer sensitivities (S_task, S_gen, S_l2) on calibration split."""
    print("\n" + "=" * 70)
    print(f"  STAGE: SENSITIVITY PROFILING (RQ2) — {model_id} on {dataset} calib")
    print("=" * 70)
    run_cmd([
        "python", "scripts/run_sensitivity.py",
        "--model", model_id,
        "--dataset", dataset,
        "--device", "auto",
    ])


def ensure_sensitivity_configs(model_id: str, prefix: str, dataset: str = "juliet"):
    """Profiles sensitivity if the {prefix}top_task/top_gen/bottom_task configs are absent."""
    needed = [f"{prefix}{k}" for k in ("top_task", "top_gen", "bottom_task")]
    if all(os.path.isfile(os.path.join("configs", f"{n}.json")) for n in needed):
        print(f"[LAUNCHER] Sensitivity configs for prefix '{prefix}' found. Skipping profiling.")
        return
    print(f"\n[LAUNCHER] Sensitivity configs for prefix '{prefix}' missing. Profiling first...")
    run_sensitivity_profiling(model_id, dataset=dataset)


def is_run_complete(filepath: str, min_samples: int = 50) -> bool:
    """Checks if a JSONL result file exists and has completed scoring."""
    if not os.path.isfile(filepath):
        return False
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        return lines >= min_samples
    except Exception:
        return False


def run_eval_suite(
    model_id: str,
    dataset: str,
    configs: List[str],
    include_cap: bool = True,
    cap_target_config: str = None,
):
    """Executes evaluation across specified configurations for a given model and dataset."""
    mid = model_id.lower()
    if "llama" in mid:
        tag = "llama8b"
    elif "7b" in mid:
        tag = "qwen7b"
    elif "phi" in mid:
        tag = "phi3"
    else:
        tag = "qwen3b"

    missing = [c for c in configs if not os.path.isfile(os.path.join("configs", f"{c}.json"))]
    if missing:
        raise FileNotFoundError(
            f"[LAUNCHER] Missing configs {missing}. Run the matching sensitivity stage first."
        )

    print("\n" + "=" * 70)
    print(f"  STAGE: EVALUATION SUITE — {model_id} on {dataset} (test split, >=1000 funcs)")
    print("=" * 70)

    for cfg in configs:
        out_jsonl = f"results/{tag}_{dataset}_{cfg}_test_cap_off.jsonl"
        if is_run_complete(out_jsonl):
            print(f"[INFO] Run {out_jsonl} already completed. Skipping.")
        else:
            if os.path.isfile(out_jsonl):
                print(f"[INFO] Removing incomplete/corrupted run {out_jsonl} to recompute cleanly...")
                os.remove(out_jsonl)
            print(f"\n>>> Running {tag} on {dataset}: config={cfg}, cap=off")
            run_cmd([
                "python", "scripts/run_eval.py",
                "--model", model_id,
                "--dataset", dataset,
                "--config", cfg,
                "--split", "test",
                "--cap", "off",
                "--device", "auto",
            ])

    if include_cap:
        target_cap_cfg = cap_target_config or (f"{tag}_top_task" if f"{tag}_top_task" in configs else "top_task")
        out_cap_jsonl = f"results/{tag}_{dataset}_{target_cap_cfg}_test_cap_on.jsonl"
        if is_run_complete(out_cap_jsonl):
            print(f"[INFO] Run {out_cap_jsonl} already completed. Skipping.")
        else:
            if os.path.isfile(out_cap_jsonl):
                print(f"[INFO] Removing incomplete/corrupted run {out_cap_jsonl} to recompute cleanly...")
                os.remove(out_cap_jsonl)
            print(f"\n>>> Running {tag} CAP ablation on {dataset}: config={target_cap_cfg}, cap=on")
            run_cmd([
                "python", "scripts/run_eval.py",
                "--model", model_id,
                "--dataset", dataset,
                "--config", target_cap_cfg,
                "--split", "test",
                "--cap", "on",
                "--device", "auto",
            ])


def run_analysis():
    """Generates all markdown tables, IEEE LaTeX tables, bootstrap CIs, and McNemar vs FP16."""
    print("\n" + "=" * 70)
    print("  STAGE: STATISTICAL ANALYSIS & LATEX GENERATION")
    print("=" * 70)
    run_cmd(["python", "scripts/analyze.py", "--results_dir", "results", "--output_dir", "paper"])


def package_artifacts(zip_name: str = "aqcap_results.zip"):
    """Zips results/ and paper/ and configs/ into a single downloadable archive for Kaggle."""
    print("\n" + "=" * 70)
    print("  PACKAGING ARTIFACTS ARCHIVE")
    print("=" * 70)
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ["results", "paper", "configs"]:
            if os.path.isdir(folder):
                for root, _, files in os.walk(folder):
                    for f in files:
                        full_path = os.path.join(root, f)
                        arc_name = os.path.relpath(full_path, start=".")
                        z.write(full_path, arc_name)
    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"[SUCCESS] Packaged {zip_name} ({size_mb:.2f} MB). Ready for download!")


def main():
    parser = argparse.ArgumentParser(description="AQCAP Kaggle Execution Launcher")
    parser.add_argument(
        "--stage",
        default="all_main",
        choices=[
            "setup",
            "rerun_audit",
            "qwen7b_sensitivity",
            "qwen7b_juliet",
            "qwen7b_primevul",
            "qwen7b_hqq_sensitivity",
            "qwen7b_hqq_sweep",
            "phi3_sensitivity",
            "phi3_juliet",
            "qwen3b_sensitivity",
            "qwen3b_juliet",
            "qwen3b_primevul",
            "llama8b_juliet",
            "qwen7b_hqq_upgrade",
            "qwen3b_pvu_hqq",
            "analyze",
            "package",
            "all_main",
        ],
        help="Stage to execute"
    )
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("paper", exist_ok=True)

    ensure_dependencies()

    if args.stage in ["setup", "all_main"]:
        setup_environment()

    # 1. Leakage Audit Rerun
    if args.stage in ["rerun_audit", "all_main"]:
        run_rerun_audit()

    # 2. Qwen-7B Juliet (Primary Model)
    if args.stage in ["qwen7b_sensitivity", "all_main"]:
        run_sensitivity_profiling(QWEN_7B_MODEL, dataset="juliet")

    if args.stage in ["qwen7b_juliet", "all_main"]:
        ensure_sensitivity_configs(QWEN_7B_MODEL, "qwen7b_", dataset="juliet")
        run_eval_suite(QWEN_7B_MODEL, dataset="juliet", configs=QWEN7B_CONFIGS)

    # 2b. Qwen-7B Natural Code Generalization on PrimeVul Unpaired (Item D)
    if args.stage in ["qwen7b_primevul", "all_main"]:
        print("\n[LAUNCHER] Running Qwen-7B natural code generalization on primevul_unpaired (Item D)...")
        run_eval_suite(QWEN_7B_MODEL, dataset="primevul_unpaired", configs=QWEN7B_PRIMEVUL_CONFIGS, include_cap=False)

    # 2c. Qwen-7B Sub-4-Bit HQQ 2-Bit Sensitivity Profiling (Item C)
    if args.stage in ["qwen7b_hqq_sensitivity", "all_main"]:
        hqq_tt_cfg = "configs/qwen7b_hqq_2bit_top_task.json"
        hqq_csv = "results/sensitivity_qwen7b_hqq2_juliet.csv"
        if os.path.isfile(hqq_tt_cfg) and os.path.isfile(hqq_csv):
            print(f"\n[LAUNCHER] HQQ 2-bit sensitivity profiling already completed ({hqq_csv}). Skipping.")
        else:
            print("\n[LAUNCHER] Running 2-bit single-layer HQQ sensitivity profiling for Qwen-7B on Juliet calib (Item C)...")
            run_cmd([
                "python", "scripts/run_sensitivity.py",
                "--model", QWEN_7B_MODEL,
                "--dataset", "juliet",
                "--output_csv", hqq_csv,
                "--prefix", "qwen7b_hqq_2bit_",
                "--perturbation", "hqq2",
                "--budget", "7",
                "--baseline_prec", "hqq2",
                "--sensitive_prec", "hqq4",
                "--no-resume",
                "--device", "auto",
            ])

    # 2d. Qwen-7B Sub-4-Bit HQQ Sweep on Juliet Test (Item C)
    if args.stage in ["qwen7b_hqq_sweep", "all_main"]:
        hqq_tt_cfg = "configs/qwen7b_hqq_2bit_top_task.json"
        if not os.path.isfile(hqq_tt_cfg):
            print("\n[LAUNCHER] HQQ 2-bit sensitivity config not found. Profiling first...")
            run_cmd([
                "python", "scripts/run_sensitivity.py",
                "--model", QWEN_7B_MODEL,
                "--dataset", "juliet",
                "--output_csv", "results/sensitivity_qwen7b_hqq2_juliet.csv",
                "--prefix", "qwen7b_hqq_2bit_",
                "--perturbation", "hqq2",
                "--budget", "7",
                "--baseline_prec", "hqq2",
                "--sensitive_prec", "hqq4",
                "--no-resume",
                "--device", "auto",
            ])
        print("\n[LAUNCHER] Running Low-Bit HQQ Sweep on Qwen-7B Juliet test (Item C)...")
        run_eval_suite(QWEN_7B_MODEL, dataset="juliet", configs=QWEN7B_HQQ_CONFIGS, include_cap=False)

    # 3. Phi-3 Juliet (Secondary Model)
    if args.stage in ["phi3_sensitivity", "all_main"]:
        run_sensitivity_profiling(PHI3_MODEL, dataset="juliet")

    if args.stage in ["phi3_juliet", "all_main"]:
        if not os.path.isfile("configs/top_task.json") and not os.path.isfile("configs/phi3_top_task.json"):
            print("\n[LAUNCHER] Phi-3 sensitivity configs not found. Profiling first...")
            run_sensitivity_profiling(PHI3_MODEL, dataset="juliet")
        run_eval_suite(PHI3_MODEL, dataset="juliet", configs=PHI3_CONFIGS)

    # 4. Qwen-3B Juliet (Tertiary Model, if time)
    if args.stage == "qwen3b_sensitivity":
        run_sensitivity_profiling(QWEN_3B_MODEL, dataset="juliet")

    if args.stage == "qwen3b_juliet":
        ensure_sensitivity_configs(QWEN_3B_MODEL, "qwen3b_", dataset="juliet")
        run_eval_suite(QWEN_3B_MODEL, dataset="juliet", configs=QWEN3B_JULIET_CONFIGS)

    # 5. Qwen-3B PrimeVul Unpaired (Secondary Study, Item A FIX)
    if args.stage in ["qwen3b_primevul", "all_main"]:
        pvu_tt_cfg = "configs/qwen3b_pvu_top_task.json"
        pvu_sens_csv = "results/sensitivity_qwen3b_primevul_unpaired.csv"
        if not os.path.isfile(pvu_tt_cfg) or not os.path.isfile(pvu_sens_csv):
            print("\n[LAUNCHER] Recomputing sensitivity for Qwen-3B on primevul_unpaired calib (no resume)...")
            run_cmd([
                "python", "scripts/run_sensitivity.py",
                "--model", QWEN_3B_MODEL,
                "--dataset", "primevul_unpaired",
                "--output_csv", pvu_sens_csv,
                "--prefix", "qwen3b_pvu_",
                "--no-resume",
                "--device", "auto",
            ])
        run_eval_suite(QWEN_3B_MODEL, dataset="primevul_unpaired", configs=QWEN3B_PRIMEVUL_CONFIGS, include_cap=False)

    # 5c. Upgrade-sensitivity mixed precision at 2-bit + extra random seeds (Qwen-7B, Juliet)
    if args.stage == "qwen7b_hqq_upgrade":
        if not os.path.isfile("configs/qwen7b_hqq_2bit_top_up.json"):
            run_cmd([
                "python", "scripts/run_upgrade_sensitivity.py",
                "--model", QWEN_7B_MODEL, "--dataset", "juliet",
                "--prefix", "qwen7b_hqq_2bit_", "--budget", "7",
                "--output_csv", "results/upgrade_sensitivity_qwen7b_hqq2_juliet.csv",
                "--device", "auto",
            ])
        cfgs = ["qwen7b_hqq_2bit_top_up"] + [f"qwen7b_hqq_2bit_rand_s{s}" for s in range(3, 10)]
        run_eval_suite(QWEN_7B_MODEL, dataset="juliet", configs=cfgs, include_cap=False)

    # 5d. Low-bit replication on Qwen-3B / PrimeVul unpaired
    if args.stage == "qwen3b_pvu_hqq":
        if not os.path.isfile("configs/qwen3b_hqq_2bit_top_up.json"):
            run_cmd([
                "python", "scripts/run_upgrade_sensitivity.py",
                "--model", QWEN_3B_MODEL, "--dataset", "primevul_unpaired",
                "--prefix", "qwen3b_hqq_2bit_", "--budget", "9",
                "--output_csv", "results/upgrade_sensitivity_qwen3b_hqq2_primevul_unpaired.csv",
                "--device", "auto",
            ])
        cfgs = (["qwen3b_hqq_4bit", "qwen3b_hqq_3bit", "qwen3b_hqq_2bit", "qwen3b_hqq_2bit_top_up"]
                + [f"qwen3b_hqq_2bit_rand_s{s}" for s in range(10)])
        run_eval_suite(QWEN_3B_MODEL, dataset="primevul_unpaired", configs=cfgs, include_cap=False)

    # 5b. Llama-3.1-8B Juliet uniform bit-width sweep (cross-family generalisation)
    if args.stage == "llama8b_juliet":
        run_eval_suite(LLAMA_8B_MODEL, dataset="juliet", configs=LLAMA8B_CONFIGS, include_cap=False)

    # 6. Analysis & Packaging
    if args.stage in ["analyze", "all_main"]:
        run_analysis()

    if args.stage in ["package", "all_main"]:
        package_artifacts()


if __name__ == "__main__":
    main()
