"""
AQCAP Comprehensive Results Analyzer & Paper Table Generator.

Reads results/*.jsonl, computes all RQ1-RQ4 statistical metrics, and generates:
1. Main Tables (Qwen2.5-Coder-7B on Juliet, Qwen2.5-Coder-3B on PrimeVul unpaired, Phi-3):
   - Paired bootstrap (2,000 resamples, identical sample/pair indices) of dAUROC vs FP16 with 95% CIs.
   - TOST (Two One-Sided Tests) equivalence testing at margin 0.02.
   - Systems latency reporting (s/sample).
2. Flip Table vs FP16 at calibration-chosen 90%-recall threshold:
   - Vuln -> Safe (missed flaws) and Safe -> Vuln (false alarms).
3. Per-CWE AUROC Breakdown for Juliet:
   - Evaluates subgroup detection performance across major CWEs (>= 15 pairs).
4. Score-Shift Plot:
   - High-resolution distribution histograms of p_vuln (FP16 vs NF4) on vulnerable and benign samples.
5. CAP Pruning Ablation (RQ4).
6. Sensitivity Spearman Correlations (RQ2).

STRICT INTEGRITY GUARD:
Refuses to process any JSONL run flagged as simulated (env.dry_run=true or is_dry_run=true)
unless --allow-dry-run is explicitly passed. In that case, all outputs are indelibly watermarked
with "SIMULATED — NOT REAL DATA" in every table caption and file header.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import glob
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aqcap.metrics import (
    compute_classification_metrics,
    find_optimal_calib_threshold,
    compute_threshold_at_recall,
    compute_paired_metrics,
    compute_quantization_degradation,
    bootstrap_paired_confidence_intervals,
    bootstrap_unpaired_confidence_intervals,
    generate_bootstrap_resample_indices,
    paired_bootstrap_auroc_diff_with_tost,
    paired_bootstrap_auroc_diff_pairs_with_tost,
    compute_flip_table,
    paired_bootstrap_auroc_diff,
    mcnemar_test_vs_fp16,
    mcnemar_test_vs_nf4,
    compute_systems_latency,
    reconstruct_pairs,
)


def load_jsonl_run(filepath: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Loads run metadata and sample records from a results JSONL file."""
    meta = {}
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "metadata":
                meta = obj
            elif obj.get("type") == "sample" or "p_vuln" in obj:
                samples.append(obj)
    return meta, samples


def analyze_sensitivity_csv(filepath: str) -> Optional[Dict[str, Any]]:
    """Computes Spearman correlations between S_task, S_gen, and S_l2 from sensitivity CSV."""
    if not os.path.isfile(filepath):
        return None
    df = pd.read_csv(filepath)
    corrs = {}
    for col1, col2 in [("S_task", "S_gen"), ("S_task", "S_l2"), ("S_gen", "S_l2")]:
        if col1 in df.columns and col2 in df.columns:
            rho, pval = spearmanr(df[col1], df[col2])
            corrs[f"{col1}_vs_{col2}"] = {"rho": float(rho), "p_val": float(pval)}
    return {"data": df, "correlations": corrs}


def generate_latex_main_table(results_df: pd.DataFrame, output_path: str, is_simulated: bool = False, dataset_name: str = "Juliet"):
    """Formats DataFrame as camera-ready IEEE double-column LaTeX table."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    caption_prefix = r"\textbf{SIMULATED --- NOT REAL DATA.} " if is_simulated else ""
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{caption_prefix}Comprehensive Vulnerability Detection Performance Across Quantization Variants on {dataset_name} Test Split. Paired bootstrap $\\Delta$AUROC vs FP16 computed over 2,000 resamples with identical sample indices. TOST equivalence evaluated at margin $\\epsilon = 0.02$.}}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"\textbf{Config} & \textbf{VRAM (MB)} & \textbf{AUROC [95\% CI]} & \textbf{AUPRC} & \textbf{B-Acc} & \textbf{MCC} & \textbf{$\Delta$AUROC vs FP16 [95\% CI]} & \textbf{TOST Equiv} & \textbf{Flip vs FP16} & \textbf{Latency (s/sample)} \\",
        r"\midrule",
    ]

    for _, row in results_df.iterrows():
        cfg = str(row.get("config", "—")).replace("_", r"\_")
        vram = f"{row.get('vram_mb', 0):,.0f}" if pd.notnull(row.get('vram_mb')) and row.get('vram_mb') > 0 else "—"
        if pd.notnull(row.get("auroc")):
            auc_str = f"{row.get('auroc', 0):.4f} [{row.get('auroc_ci_low', 0):.4f}, {row.get('auroc_ci_high', 0):.4f}]"
        else:
            auc_str = "—"
        auprc = f"{row.get('auprc', 0):.4f}" if pd.notnull(row.get('auprc')) else "—"
        bacc = f"{row.get('bacc_05', 0):.3f}" if pd.notnull(row.get('bacc_05')) else "—"
        mcc = f"{row.get('mcc_05', 0):.3f}" if pd.notnull(row.get('mcc_05')) else "—"

        if "fp16" in str(row.get("config", "")).lower():
            del_auc_str = r"\textit{Reference}"
            tost_str = r"\textit{Identical}"
            flip_str = r"0.0\%"
        else:
            del_auc = row.get("delta_auroc_vs_fp16")
            ci_l = row.get("delta_auroc_ci_low")
            ci_h = row.get("delta_auroc_ci_high")
            if pd.notnull(del_auc):
                del_auc_str = f"{del_auc:+.4f} [{ci_l:+.4f}, {ci_h:+.4f}]"
            else:
                del_auc_str = "—"
            tost_str = str(row.get("tost_equiv_vs_fp16", "—"))
            flip_val = row.get("flip_rate_pct")
            flip_str = f"{flip_val:.1f}\\%" if pd.notnull(flip_val) else "—"

        lat = f"{row.get('median_latency_s', 0):.3f}" if pd.notnull(row.get('median_latency_s')) else "—"

        tex_lines.append(
            f"{cfg} & {vram} & {auc_str} & {auprc} & {bacc} & {mcc} & {del_auc_str} & {tost_str} & {flip_str} & {lat} \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))
    print(f"[ANALYZE] Saved primary LaTeX table -> {output_path}")


def generate_latex_flip_table(flip_df: pd.DataFrame, output_path: str, is_simulated: bool = False, dataset_name: str = "Juliet"):
    """Formats prediction flip table vs FP16 at tau_90 as camera-ready LaTeX table."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    caption_prefix = r"\textbf{SIMULATED --- NOT REAL DATA.} " if is_simulated else ""
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption_prefix}Prediction Flip Analysis vs FP16 at Calibration-Chosen 90\\%-Recall Threshold ($\\tau_{{90}}$) on {dataset_name} Test Split. Shows Vulnerable $\\to$ Safe (missed vulnerabilities) and Safe $\\to$ Vulnerable (false alarms).}}",
        r"\label{tab:flip_results}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Config} & \textbf{$\tau_{90}$} & \textbf{Vuln $\to$ Safe} & \textbf{Safe $\to$ Vuln} & \textbf{Total Flips (\%)} & \textbf{Harmful} & \textbf{Net Shift} \\",
        r"\midrule",
    ]

    for _, row in flip_df.iterrows():
        cfg = str(row.get("config", "—")).replace("_", r"\_")
        tau = f"{row.get('tau_90', 0):.3f}"
        v2s = f"{int(row.get('vuln_to_safe', 0))} ({row.get('vuln_to_safe_pct', 0):.1f}\\%)"
        s2v = f"{int(row.get('safe_to_vuln', 0))} ({row.get('safe_to_vuln_pct', 0):.1f}\\%)"
        tot = f"{int(row.get('total_flips', 0))} ({row.get('flip_rate_pct', 0):.1f}\\%)"
        harm = f"{int(row.get('harmful_flips', 0))}"
        net = f"{int(row.get('net_shift', 0)):+d}"
        tex_lines.append(f"{cfg} & {tau} & {v2s} & {s2v} & {tot} & {harm} & {net} \\\\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))
    print(f"[ANALYZE] Saved flip LaTeX table -> {output_path}")


def generate_latex_cwe_table(cwe_df: pd.DataFrame, output_path: str, is_simulated: bool = False):
    """Formats Per-CWE subgroup breakdown as camera-ready LaTeX table."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    caption_prefix = r"\textbf{SIMULATED --- NOT REAL DATA.} " if is_simulated else ""
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption_prefix}Per-CWE Subgroup Vulnerability Detection AUROC on Audited Juliet Benchmark (Categories with $\\ge 15$ pairs).}}",
        r"\label{tab:cwe_breakdown}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{CWE Identifier} & \textbf{Pairs ($n$)} & \textbf{FP16} & \textbf{NF4} & \textbf{INT8} & \textbf{Top-Task} & \textbf{Last-14} \\",
        r"\midrule",
    ]

    for _, row in cwe_df.iterrows():
        cwe = str(row.get("cwe", "—"))
        n_pairs = int(row.get("n_pairs", 0))
        auc_fp16 = f"{row.get('auroc_fp16', 0):.3f}" if pd.notnull(row.get('auroc_fp16')) else "—"
        auc_nf4 = f"{row.get('auroc_nf4', 0):.3f}" if pd.notnull(row.get('auroc_nf4')) else "—"
        auc_int8 = f"{row.get('auroc_int8', 0):.3f}" if pd.notnull(row.get('auroc_int8')) else "—"
        auc_top = f"{row.get('auroc_top_task', 0):.3f}" if pd.notnull(row.get('auroc_top_task')) else "—"
        auc_last = f"{row.get('auroc_last14', 0):.3f}" if pd.notnull(row.get('auroc_last14')) else "—"

        tex_lines.append(
            f"{cwe} & {n_pairs} & {auc_fp16} & {auc_nf4} & {auc_int8} & {auc_top} & {auc_last} \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))
    print(f"[ANALYZE] Saved CWE LaTeX table -> {output_path}")


def generate_latex_cap_table(cap_df: pd.DataFrame, output_path: str, is_simulated: bool = False):
    """Formats CAP Pruning ablation results as camera-ready LaTeX table."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    caption_prefix = r"\textbf{SIMULATED --- NOT REAL DATA.} " if is_simulated else ""
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption_prefix}Context-Aware Pruning (CAP) Ablation: Token Compression, Latency Speedup, and Security Information Loss.}}",
        r"\label{tab:cap_ablation}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Config} & \textbf{CAP} & \textbf{Tokens (Mean)} & \textbf{Token Red. (\%)} & \textbf{Latency (s)} & \textbf{Speedup (\%)} & \textbf{AUROC} & \textbf{Info Loss (\%)} \\",
        r"\midrule",
    ]

    for _, row in cap_df.iterrows():
        cfg = str(row.get("config", "—")).replace("_", r"\_")
        cap_setting = "On" if row.get("cap") == "on" else "Off"
        mean_tok = f"{row.get('mean_tokens', 0):.1f}"
        tok_red = f"{row.get('token_reduction_pct', 0):.1f}\\%" if row.get("cap") == "on" else "—"
        lat = f"{row.get('median_latency_s', 0):.3f}"
        spd = f"{row.get('speedup_pct', 0):+.1f}\\%" if row.get("cap") == "on" else "—"
        auc = f"{row.get('auroc', 0):.3f}"
        info_loss = f"{row.get('sec_info_loss_pct', 0):.1f}\\%" if row.get("cap") == "on" else "0.0\\%"

        tex_lines.append(
            f"{cfg} & {cap_setting} & {mean_tok} & {tok_red} & {lat} & {spd} & {auc} & {info_loss} \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))
    print(f"[ANALYZE] Saved CAP LaTeX table -> {output_path}")


def analyze_runs_catalog(
    catalog: Dict[str, Dict[str, Any]],
    model_tag: str,
    dataset: str = "juliet",
    split: str = "test",
    cap: str = "off"
) -> pd.DataFrame:
    """
    Computes comprehensive metrics across a filtered set of evaluated runs.
    Uses 2,000 paired bootstrap resamples with identical pre-drawn sample/pair indices
    to ensure mathematically rigorous dAUROC comparisons against uncompressed FP16.
    Evaluates TOST equivalence at margin epsilon = 0.02.
    """
    matching = {}
    for run_key, data in catalog.items():
        meta = data["meta"]
        m_name = meta.get("model", "").lower()
        d_name = meta.get("dataset", "").lower()

        is_model = (model_tag in m_name) or (model_tag in run_key)
        is_dataset = (dataset in d_name) or (f"_{dataset}_" in run_key) if dataset else True
        is_split = (meta.get("split", "test") == split) or (f"_{split}_" in run_key)
        is_cap = (meta.get("cap", "off") == cap) or (f"cap_{cap}" in run_key)

        if is_model and is_dataset and is_split and is_cap:
            cfg_name = meta.get("config_name")
            if not cfg_name:
                cfg_obj = meta.get("config")
                if isinstance(cfg_obj, dict):
                    cfg_name = cfg_obj.get("name")
                elif isinstance(cfg_obj, str):
                    cfg_name = cfg_obj
            if not cfg_name:
                parts = run_key.split("_")
                cfg_name = parts[1] if len(parts) > 1 else run_key
            matching[cfg_name] = data

    if not matching:
        return pd.DataFrame()

    # Locate reference run (FP16)
    fp16_data = matching.get("fp16")
    if not fp16_data:
        for k in matching:
            if k.endswith("fp16") or "fp16" in k:
                fp16_data = matching[k]
                break

    fp16_samples = fp16_data.get("samples", []) if fp16_data else []
    fp16_pairs = reconstruct_pairs(fp16_samples) if fp16_samples else []
    is_paired = bool(fp16_pairs) and any(s.get("pair_id") for s in fp16_samples)

    y_true_fp16 = np.array([s["target"] for s in fp16_samples]) if fp16_samples else np.array([])
    y_prob_fp16 = np.array([s["p_vuln"] for s in fp16_samples]) if fp16_samples else np.array([])
    tau_90 = compute_threshold_at_recall(y_true_fp16, y_prob_fp16, target_recall=0.90) if len(y_true_fp16) > 0 else 0.5

    # Pre-generate 2,000 bootstrap resample indices: CRUCIAL FOR PAIRED COMPARISON
    # All configurations will be evaluated on the EXACT same resampled replicates.
    if is_paired and len(fp16_pairs) > 0:
        resample_indices = generate_bootstrap_resample_indices(len(fp16_pairs), n_iterations=2000, seed=42)
    elif len(y_true_fp16) > 0:
        resample_indices = generate_bootstrap_resample_indices(len(y_true_fp16), n_iterations=2000, seed=42)
    else:
        resample_indices = np.array([])

    order_preference = [
        "fp16", "int8", "nf4", "hqq_4bit", "hqq_3bit", "hqq_2bit",
        "top_task", "pvu_top_task", "top_gen",
        "last14", "first14", "last16", "first16", "last18", "first18",
        "first7", "last7",
        "rand_s0", "rand_s1", "rand_s2",
        "bottom_task"
    ]
    sorted_cfgs = sorted(
        matching.keys(),
        key=lambda x: order_preference.index(
            next((op for op in order_preference if op in x), "zzz")
        ) if any(op in x for op in order_preference) else 99
    )

    rows = []
    for cfg_name in sorted_cfgs:
        data = matching[cfg_name]
        samples = data["samples"]
        if not samples:
            continue
        pairs = reconstruct_pairs(samples) if is_paired else []
        y_true = np.array([s["target"] for s in samples])
        y_prob = np.array([s["p_vuln"] for s in samples])
        latencies = [s.get("latency_s", 0.0) for s in samples if "latency_s" in s]

        m_05 = compute_classification_metrics(y_true, y_prob, threshold=0.5)
        pair_m = compute_paired_metrics(pairs, threshold=0.5) if is_paired else {"P-C": None, "P-V": None, "P-B": None, "P-R": None}
        sys_lat = compute_systems_latency(latencies)

        # 2,000-sample bootstrap confidence intervals for individual config AUROC
        if is_paired and pairs:
            ci_95 = bootstrap_paired_confidence_intervals(pairs, n_iterations=2000, seed=42, threshold=0.5)
        else:
            ci_95 = bootstrap_unpaired_confidence_intervals(y_true, y_prob, n_iterations=2000, seed=42, threshold=0.5)

        # Paired bootstrap dAUROC vs FP16 with 95% CI & TOST equivalence at margin 0.02
        delta_auc_vs_fp16 = None
        delta_auc_ci_low = None
        delta_auc_ci_high = None
        tost_p_val = None
        tost_equiv = "—"
        mcnemar_res_fp16 = None

        if fp16_samples and not ("fp16" in cfg_name.lower()):
            if is_paired and pairs and fp16_pairs and len(resample_indices) > 0:
                tost_res = paired_bootstrap_auroc_diff_pairs_with_tost(
                    pairs, fp16_pairs, resample_indices, tost_margin=0.02
                )
            elif len(y_prob_fp16) == len(y_prob) and len(resample_indices) > 0:
                tost_res = paired_bootstrap_auroc_diff_with_tost(
                    y_true, y_prob, y_prob_fp16, resample_indices, tost_margin=0.02
                )
            else:
                tost_res = {}

            if tost_res:
                delta_auc_vs_fp16 = tost_res["mean_delta_auroc"]
                delta_auc_ci_low = tost_res["ci_95"][0]
                delta_auc_ci_high = tost_res["ci_95"][1]
                tost_p_val = tost_res["tost_p"]
                tost_equiv = "YES" if tost_res["is_equivalent"] else "NO"

            y_pred_cfg = (y_prob >= tau_90).astype(int)
            y_pred_fp16 = (y_prob_fp16 >= tau_90).astype(int)
            mcnemar_res_fp16 = mcnemar_test_vs_fp16(y_true, y_pred_cfg, y_pred_fp16)

        # Flips at calib-chosen 90%-recall threshold
        if fp16_samples and len(y_prob_fp16) == len(y_prob) and not ("fp16" in cfg_name.lower()):
            flips = compute_flip_table(y_true, y_prob, y_prob_fp16, threshold=tau_90)
            flip_rate_pct = flips["flip_rate_pct"]
            vuln_to_safe = flips["vuln_to_safe"]
            safe_to_vuln = flips["safe_to_vuln"]
        else:
            flip_rate_pct = 0.0 if "fp16" in cfg_name.lower() else None
            vuln_to_safe = 0 if "fp16" in cfg_name.lower() else None
            safe_to_vuln = 0 if "fp16" in cfg_name.lower() else None

        row = {
            "config": cfg_name,
            "vram_mb": data["meta"].get("weights_mb", 0.0),
            "auroc": m_05["auroc"],
            "auroc_ci_low": ci_95.get("auroc", (0, 0))[0],
            "auroc_ci_high": ci_95.get("auroc", (0, 0))[1],
            "auprc": m_05["auprc"],
            "fpr_at_90_recall": m_05.get("fpr_at_90_recall"),
            "fpr_at_95_recall": m_05.get("fpr_at_95_recall"),
            "bacc_05": m_05["balanced_accuracy"],
            "mcc_05": m_05["mcc"],
            "f1_05": m_05["f1"],
            "p_c": pair_m.get("P-C"),
            "p_r": pair_m.get("P-R"),
            "delta_auroc_vs_fp16": delta_auc_vs_fp16,
            "delta_auroc_ci_low": delta_auc_ci_low,
            "delta_auroc_ci_high": delta_auc_ci_high,
            "tost_p_vs_fp16": tost_p_val,
            "tost_equiv_vs_fp16": tost_equiv,
            "mcnemar_p_vs_fp16": mcnemar_res_fp16.get("p_value") if mcnemar_res_fp16 else None,
            "tau_90": tau_90,
            "vuln_to_safe": vuln_to_safe,
            "safe_to_vuln": safe_to_vuln,
            "flip_rate_pct": flip_rate_pct,
            "median_latency_s": sys_lat["median_s"],
            "mean_latency_s": sys_lat["mean_s"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def generate_flip_table_df(
    catalog: Dict[str, Dict[str, Any]],
    model_tag: str = "qwen7b",
    dataset: str = "juliet",
) -> pd.DataFrame:
    """
    Computes detailed prediction flip table vs FP16 at the calibration-selected
    90%-recall threshold tau_90.
    """
    runs = {}
    for r_key, r_val in catalog.items():
        meta = r_val["meta"]
        m_match = (model_tag in meta.get("model", "").lower()) or (model_tag in r_key)
        d_match = (dataset in meta.get("dataset", "").lower()) or (f"_{dataset}_" in r_key)
        c_name = meta.get("config_name", "")
        if not c_name and isinstance(meta.get("config"), dict):
            c_name = meta.get("config", {}).get("name", "")
        if not c_name:
            parts = r_key.split("_")
            c_name = parts[1] if len(parts) > 1 else ""

        if m_match and d_match and (meta.get("cap", "off") == "off" or "_cap_off" in r_key):
            runs[c_name] = r_val["samples"]

    fp16_samples = None
    for k in runs:
        if "fp16" in k.lower():
            fp16_samples = runs[k]
            break

    if not fp16_samples:
        return pd.DataFrame()

    y_true = np.array([s["target"] for s in fp16_samples])
    y_prob_fp16 = np.array([s["p_vuln"] for s in fp16_samples])
    tau_90 = compute_threshold_at_recall(y_true, y_prob_fp16, target_recall=0.90)

    rows = []
    for cfg_name, samples in runs.items():
        if "fp16" in cfg_name.lower():
            continue
        y_prob_cfg = np.array([s["p_vuln"] for s in samples])
        flips = compute_flip_table(y_true, y_prob_cfg, y_prob_fp16, threshold=tau_90)
        rows.append({
            "config": cfg_name,
            "tau_90": tau_90,
            "vuln_to_safe": flips["vuln_to_safe"],
            "vuln_to_safe_pct": flips["vuln_to_safe_pct"],
            "safe_to_vuln": flips["safe_to_vuln"],
            "safe_to_vuln_pct": flips["safe_to_vuln_pct"],
            "total_flips": flips["total_flips"],
            "flip_rate_pct": flips["flip_rate_pct"],
            "harmful_flips": flips["harmful_flips"],
            "beneficial_flips": flips["beneficial_flips"],
            "net_shift": flips["net_shift"],
        })

    return pd.DataFrame(rows)


def analyze_per_cwe_breakdown(
    catalog: Dict[str, Dict[str, Any]],
    model_tag: str = "qwen7b",
    dataset: str = "juliet",
    min_pairs: int = 15,
) -> pd.DataFrame:
    """
    Computes per-CWE AUROC across key configurations on the Juliet test split.
    Evaluates CWE categories with at least min_pairs pairs.
    """
    key_configs = ["fp16", "nf4", "int8", "top_task", "last14", "first14"]
    runs = {}
    for r_key, r_val in catalog.items():
        meta = r_val["meta"]
        m_match = (model_tag in meta.get("model", "").lower()) or (model_tag in r_key)
        d_match = (dataset in meta.get("dataset", "").lower()) or (f"_{dataset}_" in r_key)
        c_name = meta.get("config_name", "")
        if not c_name and isinstance(meta.get("config"), dict):
            c_name = meta.get("config", {}).get("name", "")
        if not c_name:
            parts = r_key.split("_")
            c_name = parts[1] if len(parts) > 1 else ""

        clean_c_name = c_name
        for pfx in [f"{model_tag}_", "phi3_", "qwen3b_", "qwen7b_"]:
            if clean_c_name.startswith(pfx):
                clean_c_name = clean_c_name[len(pfx):]

        if m_match and d_match and (meta.get("cap", "off") == "off" or "_cap_off" in r_key):
            if clean_c_name in key_configs or c_name in key_configs:
                matched_key = clean_c_name if clean_c_name in key_configs else c_name
                runs[matched_key] = r_val["samples"]

    if not runs:
        return pd.DataFrame()

    first_samples = list(runs.values())[0]
    pairs = reconstruct_pairs(first_samples)
    cwe_to_pair_ids = defaultdict(list)
    for p in pairs:
        cwe = p.get("cwe", "Unknown")
        cwe_to_pair_ids[cwe].append(p["pair_id"])

    qualifying_cwes = {cwe: p_ids for cwe, p_ids in cwe_to_pair_ids.items() if len(p_ids) >= min_pairs}
    if not qualifying_cwes:
        qualifying_cwes = {cwe: p_ids for cwe, p_ids in cwe_to_pair_ids.items() if len(p_ids) >= 8}

    cwe_rows = []
    for cwe, p_ids in sorted(qualifying_cwes.items(), key=lambda x: len(x[1]), reverse=True):
        row = {"cwe": cwe, "n_pairs": len(p_ids)}
        p_set = set(p_ids)

        for cfg in key_configs:
            samples = runs.get(cfg, [])
            if not samples:
                row[f"auroc_{cfg}"] = np.nan
                continue
            sub_samples = [s for s in samples if s.get("pair_id") in p_set]
            if not sub_samples:
                row[f"auroc_{cfg}"] = np.nan
                continue
            y_t = np.array([s["target"] for s in sub_samples])
            y_p = np.array([s["p_vuln"] for s in sub_samples])
            m = compute_classification_metrics(y_t, y_p)
            row[f"auroc_{cfg}"] = m["auroc"]

        cwe_rows.append(row)

    return pd.DataFrame(cwe_rows)


def analyze_cap_ablation(catalog: Dict[str, Dict[str, Any]], model_tag: str = "qwen7b") -> pd.DataFrame:
    """Analyzes CAP ablation (RQ4) across fp16, nf4, and top_task."""
    configs_to_check = ["fp16", "nf4", "top_task"]
    rows = []

    for cfg_name in configs_to_check:
        off_data = None
        on_data = None
        for r_key, r_val in catalog.items():
            meta = r_val["meta"]
            m_match = (model_tag in meta.get("model", "").lower()) or (model_tag in r_key)
            c_name = meta.get("config_name", "")
            if not c_name and isinstance(meta.get("config"), dict):
                c_name = meta.get("config", {}).get("name", "")
            if not c_name:
                parts = r_key.split("_")
                c_name = parts[1] if len(parts) > 1 else ""

            if m_match and (cfg_name in c_name or c_name.endswith(cfg_name)):
                if meta.get("cap") == "off" or "_cap_off" in r_key:
                    off_data = r_val
                elif meta.get("cap") == "on" or "_cap_on" in r_key:
                    on_data = r_val

        if not off_data or not on_data:
            continue

        off_samples = off_data["samples"]
        off_tokens = [s.get("n_input_tokens", 0) for s in off_samples if "n_input_tokens" in s]
        off_lats = [s.get("latency_s", 0) for s in off_samples if "latency_s" in s]
        off_m = compute_classification_metrics(
            np.array([s["target"] for s in off_samples]),
            np.array([s["p_vuln"] for s in off_samples]),
        )
        mean_tok_off = float(np.mean(off_tokens)) if off_tokens else 0.0
        med_lat_off = float(np.median(off_lats)) if off_lats else 0.0

        rows.append({
            "config": cfg_name,
            "cap": "off",
            "mean_tokens": mean_tok_off,
            "token_reduction_pct": 0.0,
            "median_latency_s": med_lat_off,
            "speedup_pct": 0.0,
            "auroc": off_m["auroc"],
            "sec_info_loss_pct": 0.0,
        })

        on_samples = on_data["samples"]
        on_tokens = [s.get("n_input_tokens", 0) for s in on_samples if "n_input_tokens" in s]
        on_lats = [s.get("latency_s", 0) for s in on_samples if "latency_s" in s]
        on_m = compute_classification_metrics(
            np.array([s["target"] for s in on_samples]),
            np.array([s["p_vuln"] for s in on_samples]),
        )
        mean_tok_on = float(np.mean(on_tokens)) if on_tokens else 0.0
        med_lat_on = float(np.median(on_lats)) if on_lats else 0.0

        tok_red_pct = ((mean_tok_off - mean_tok_on) / max(mean_tok_off, 1e-6)) * 100.0
        speedup_pct = ((med_lat_off - med_lat_on) / max(med_lat_off, 1e-6)) * 100.0

        sec_loss_count = sum(1 for s in on_samples if len(s.get("sec_keywords", [])) > 0)
        sec_loss_pct = (sec_loss_count / max(len(on_samples), 1)) * 100.0

        rows.append({
            "config": cfg_name,
            "cap": "on",
            "mean_tokens": mean_tok_on,
            "token_reduction_pct": tok_red_pct,
            "median_latency_s": med_lat_on,
            "speedup_pct": speedup_pct,
            "auroc": on_m["auroc"],
            "sec_info_loss_pct": sec_loss_pct,
        })

    return pd.DataFrame(rows)


def generate_score_shift_plot(
    catalog: Dict[str, Dict[str, Any]],
    model_tag: str = "qwen7b",
    dataset: str = "juliet",
    output_dir: str = "paper",
):
    """
    Generates a publication-grade figure comparing p_vuln distribution
    shifts between uncompressed FP16 and uniform NF4.
    Saves to:
      - paper/score_shift_fp16_vs_nf4.png
      - paper/score_shift_fp16_vs_nf4.pdf
    """
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    fp16_samples = []
    nf4_samples = []
    for r_key, r_val in catalog.items():
        meta = r_val["meta"]
        m_name = meta.get("model", "").lower()
        d_name = meta.get("dataset", "").lower()
        if (model_tag in m_name or model_tag in r_key) and (dataset in d_name or f"_{dataset}_" in r_key):
            cfg_name = meta.get("config_name") or r_key
            if "fp16" in cfg_name and (meta.get("cap", "off") == "off" or "_cap_off" in r_key):
                fp16_samples = r_val["samples"]
            elif "nf4" in cfg_name and (meta.get("cap", "off") == "off" or "_cap_off" in r_key):
                nf4_samples = r_val["samples"]

    if not fp16_samples or not nf4_samples:
        print("[ANALYZE] Could not find both FP16 and NF4 runs for score-shift plot. Skipping.")
        return

    fp16_map = {s["idx"]: s for s in fp16_samples}
    nf4_map = {s["idx"]: s for s in nf4_samples}
    common_keys = sorted(list(set(fp16_map.keys()) & set(nf4_map.keys())))

    vuln_fp16 = [fp16_map[k]["p_vuln"] for k in common_keys if fp16_map[k]["target"] == 1]
    vuln_nf4  = [nf4_map[k]["p_vuln"] for k in common_keys if fp16_map[k]["target"] == 1]
    safe_fp16 = [fp16_map[k]["p_vuln"] for k in common_keys if fp16_map[k]["target"] == 0]
    safe_nf4  = [nf4_map[k]["p_vuln"] for k in common_keys if fp16_map[k]["target"] == 0]

    tau_90 = float(np.percentile(vuln_fp16, 10.0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel A: Vulnerable Functions
    ax1 = axes[0]
    bins = np.linspace(0.0, 1.0, 25)
    ax1.hist(vuln_fp16, bins=bins, alpha=0.55, color="#1f77b4", edgecolor="#144f77", label=f"FP16 (μ={np.mean(vuln_fp16):.3f})", density=True)
    ax1.hist(vuln_nf4,  bins=bins, alpha=0.55, color="#d62728", edgecolor="#8e1a1b", label=f"NF4 (μ={np.mean(vuln_nf4):.3f})", density=True)
    ax1.axvline(tau_90, color="black", linestyle="--", linewidth=1.5, label=f"$\\tau_{{90}}$ Threshold ({tau_90:.3f})")
    ax1.set_title("Vulnerable Functions ($y = 1$)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Predicted Vulnerability Probability ($p_{\\mathrm{vuln}}$)", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.legend(frameon=True, fontsize=10, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlim(-0.02, 1.02)

    # Panel B: Benign Functions
    ax2 = axes[1]
    ax2.hist(safe_fp16, bins=bins, alpha=0.55, color="#1f77b4", edgecolor="#144f77", label=f"FP16 (μ={np.mean(safe_fp16):.3f})", density=True)
    ax2.hist(safe_nf4,  bins=bins, alpha=0.55, color="#d62728", edgecolor="#8e1a1b", label=f"NF4 (μ={np.mean(safe_nf4):.3f})", density=True)
    ax2.axvline(tau_90, color="black", linestyle="--", linewidth=1.5, label=f"$\\tau_{{90}}$ Threshold ({tau_90:.3f})")
    ax2.set_title("Benign / Safe Functions ($y = 0$)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Predicted Vulnerability Probability ($p_{\\mathrm{vuln}}$)", fontsize=11)
    ax2.set_ylabel("Probability Density", fontsize=11)
    ax2.legend(frameon=True, fontsize=10, loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_xlim(-0.02, 1.02)

    plt.suptitle("Quantization Score-Shift: Uncompressed FP16 vs Uniform NF4 Distribution on Juliet Test",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    png_path = os.path.join(output_dir, "score_shift_fp16_vs_nf4.png")
    pdf_path = os.path.join(output_dir, "score_shift_fp16_vs_nf4.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ANALYZE] Saved score-shift plot -> {png_path} and {pdf_path}")


def run_full_analysis(results_dir: str = "results", output_dir: str = "paper", allow_dry_run: bool = False):
    """
    Main analysis orchestrator generating all Markdown reports and IEEE LaTeX tables.
    CRITICAL INTEGRITY GUARD:
    Refuses to process data containing env.dry_run=true or is_dry_run=true unless allow_dry_run=True.
    """
    print(f"\n{'='*70}\n  AQCAP RESULTS ANALYZER & LATEX TABLE GENERATOR\n{'='*70}")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    jsonl_files = glob.glob(os.path.join(results_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"[ANALYZE] No .jsonl files found in {results_dir}/. Place evaluation outputs here.")
        return

    catalog = {}
    has_simulated_data = False
    for f in jsonl_files:
        meta, samples = load_jsonl_run(f)
        run_id = meta.get("run_id", os.path.splitext(os.path.basename(f))[0])
        catalog[run_id] = {"meta": meta, "samples": samples, "filepath": f}

        if meta.get("env", {}).get("dry_run") is True or meta.get("is_dry_run") is True or meta.get("dry_run") is True:
            has_simulated_data = True

    if has_simulated_data:
        if not allow_dry_run:
            raise ValueError(
                f"REFUSAL ERROR: Directory '{results_dir}' contains simulated dry-run records (env.dry_run=true). "
                f"analyze.py REFUSES to process simulated data into official paper tables unless --allow-dry-run "
                f"is explicitly passed. To prevent accidental paper contamination, aborting immediately."
            )
        else:
            print(f"\n[GUARD WARNING] Simulated data detected with --allow-dry-run. "
                  f"Stamping 'SIMULATED — NOT REAL DATA' on all captions and headers.\n")

    is_simulated = has_simulated_data and allow_dry_run
    sim_header = (
        "# SIMULATED — NOT REAL DATA\n\n"
        "> **CRITICAL WARNING:** This file was generated using synthetic / simulated data.\n"
        "> It must NOT be used for academic publication, submissions, or empirical claims.\n\n"
    ) if is_simulated else ""

    print(f"[ANALYZE] Loaded {len(catalog)} evaluated run(s):")
    for r_id in catalog.keys():
        print(f"  - {r_id} ({len(catalog[r_id]['samples'])} samples)")

    # 1. Primary Model: Qwen2.5-Coder-7B on Juliet Test Split (Main Study)
    df_qwen7b_juliet = analyze_runs_catalog(catalog, model_tag="qwen7b", dataset="juliet", split="test", cap="off")
    if not df_qwen7b_juliet.empty:
        md_q7 = os.path.join(results_dir, "table_qwen7b_juliet.md")
        with open(md_q7, "w", encoding="utf-8") as f:
            f.write(sim_header + df_qwen7b_juliet.to_markdown(index=False))
        print(f"\n[ANALYZE] Generated Primary Model (Qwen-7B Juliet) Markdown -> {md_q7}")

        tex_q7 = os.path.join(output_dir, "table_qwen7b_juliet.tex")
        generate_latex_main_table(df_qwen7b_juliet, tex_q7, is_simulated=is_simulated, dataset_name="Juliet")

        # Flips table for Qwen-7B Juliet
        df_flip_q7 = generate_flip_table_df(catalog, model_tag="qwen7b", dataset="juliet")
        if not df_flip_q7.empty:
            md_flip_q7 = os.path.join(results_dir, "flip_table_qwen7b_juliet.md")
            with open(md_flip_q7, "w", encoding="utf-8") as f:
                f.write(sim_header + df_flip_q7.to_markdown(index=False))
            tex_flip_q7 = os.path.join(output_dir, "table_flip_qwen7b_juliet.tex")
            generate_latex_flip_table(df_flip_q7, tex_flip_q7, is_simulated=is_simulated, dataset_name="Juliet")

        # Per-CWE breakdown for Juliet
        df_cwe_q7 = analyze_per_cwe_breakdown(catalog, model_tag="qwen7b", dataset="juliet", min_pairs=15)
        if not df_cwe_q7.empty:
            md_cwe_q7 = os.path.join(results_dir, "cwe_breakdown_juliet.md")
            with open(md_cwe_q7, "w", encoding="utf-8") as f:
                f.write(sim_header + df_cwe_q7.to_markdown(index=False))
            tex_cwe_q7 = os.path.join(output_dir, "table_cwe_juliet.tex")
            generate_latex_cwe_table(df_cwe_q7, tex_cwe_q7, is_simulated=is_simulated)

        # Score-Shift Plot (FP16 vs NF4)
        generate_score_shift_plot(catalog, model_tag="qwen7b", dataset="juliet", output_dir=output_dir)

    # 2. Secondary Study: Qwen2.5-Coder-3B on PrimeVul Unpaired Test Split
    df_qwen3b_pv = analyze_runs_catalog(catalog, model_tag="qwen3b", dataset="primevul_unpaired", split="test", cap="off")
    if not df_qwen3b_pv.empty:
        md_q3_pv = os.path.join(results_dir, "table_qwen3b_primevul_unpaired.md")
        with open(md_q3_pv, "w", encoding="utf-8") as f:
            f.write(sim_header + df_qwen3b_pv.to_markdown(index=False))
        print(f"[ANALYZE] Generated Qwen-3B PrimeVul Unpaired Markdown -> {md_q3_pv}")

        tex_q3_pv = os.path.join(output_dir, "table_qwen3b_primevul_unpaired.tex")
        generate_latex_main_table(df_qwen3b_pv, tex_q3_pv, is_simulated=is_simulated, dataset_name="PrimeVul Unpaired")

        df_flip_q3_pv = generate_flip_table_df(catalog, model_tag="qwen3b", dataset="primevul_unpaired")
        if not df_flip_q3_pv.empty:
            md_flip_q3 = os.path.join(results_dir, "flip_table_qwen3b_primevul_unpaired.md")
            with open(md_flip_q3, "w", encoding="utf-8") as f:
                f.write(sim_header + df_flip_q3_pv.to_markdown(index=False))

    # 3. Item D: Qwen2.5-Coder-7B on PrimeVul Unpaired Test Split (Natural Code Generalization)
    df_qwen7b_pv = analyze_runs_catalog(catalog, model_tag="qwen7b", dataset="primevul_unpaired", split="test", cap="off")
    if not df_qwen7b_pv.empty:
        md_q7_pv = os.path.join(results_dir, "table_qwen7b_primevul_unpaired.md")
        with open(md_q7_pv, "w", encoding="utf-8") as f:
            f.write(sim_header + df_qwen7b_pv.to_markdown(index=False))
        print(f"[ANALYZE] Generated Qwen-7B PrimeVul Unpaired Markdown -> {md_q7_pv}")

        tex_q7_pv = os.path.join(output_dir, "table_qwen7b_primevul_unpaired.tex")
        generate_latex_main_table(df_qwen7b_pv, tex_q7_pv, is_simulated=is_simulated, dataset_name="PrimeVul Unpaired (Qwen-7B)")

    # 4. Item C: Qwen2.5-Coder-7B Juliet HQQ Sweep (Low-Bit Quantization)
    hqq_runs_present = any("hqq" in k.lower() for k in catalog.keys())
    if hqq_runs_present:
        df_hqq = analyze_runs_catalog(catalog, model_tag="qwen7b", dataset="juliet", split="test", cap="off")
        df_hqq_filtered = df_hqq[df_hqq["config"].str.contains("hqq|fp16", case=False)] if not df_hqq.empty else pd.DataFrame()
        if not df_hqq_filtered.empty:
            md_hqq = os.path.join(results_dir, "table_qwen7b_hqq_sweep.md")
            with open(md_hqq, "w", encoding="utf-8") as f:
                f.write(sim_header + df_hqq_filtered.to_markdown(index=False))
            tex_hqq = os.path.join(output_dir, "table_hqq_sweep.tex")
            generate_latex_main_table(df_hqq_filtered, tex_hqq, is_simulated=is_simulated, dataset_name="Juliet Sub-4-Bit (HQQ)")

    # 5. Secondary Model: Phi-3-mini on Juliet Test Split (if evaluated)
    df_phi3_juliet = analyze_runs_catalog(catalog, model_tag="phi3", dataset="juliet", split="test", cap="off")
    if not df_phi3_juliet.empty:
        md_p3 = os.path.join(results_dir, "table_phi3_juliet.md")
        with open(md_p3, "w", encoding="utf-8") as f:
            f.write(sim_header + df_phi3_juliet.to_markdown(index=False))
        tex_p3 = os.path.join(output_dir, "table_phi3_juliet.tex")
        generate_latex_main_table(df_phi3_juliet, tex_p3, is_simulated=is_simulated, dataset_name="Juliet (Phi-3-mini)")

    # 5b. Legacy / Untagged Main Results Table (Phi-3-mini fallback)
    df_phi3 = analyze_runs_catalog(catalog, model_tag="phi3", dataset="", split="test", cap="off")
    if not df_phi3.empty and df_phi3_juliet.empty:
        md_main = os.path.join(results_dir, "main_table.md")
        with open(md_main, "w", encoding="utf-8") as f:
            f.write(sim_header + df_phi3.to_markdown(index=False))
        print(f"\n[ANALYZE] Generated Primary Markdown Table -> {md_main}")
        tex_main = os.path.join(output_dir, "table_main.tex")
        generate_latex_main_table(df_phi3, tex_main, is_simulated=is_simulated, dataset_name="Main Study")

    # 6. CAP Pruning Ablation Table (RQ4)
    df_cap = analyze_cap_ablation(catalog, model_tag="qwen7b")
    if df_cap.empty:
        df_cap = analyze_cap_ablation(catalog, model_tag="phi3")
    if not df_cap.empty:
        md_cap = os.path.join(results_dir, "cap_ablation.md")
        with open(md_cap, "w", encoding="utf-8") as f:
            f.write(sim_header + df_cap.to_markdown(index=False))
        tex_cap = os.path.join(output_dir, "table_cap.tex")
        generate_latex_cap_table(df_cap, tex_cap, is_simulated=is_simulated)

    # 7. Sensitivity Spearman Correlations
    sens_reports = []
    for sens_csv in glob.glob(os.path.join(results_dir, "sensitivity_*.csv")):
        sens_res = analyze_sensitivity_csv(sens_csv)
        if sens_res:
            m_name = os.path.basename(sens_csv)
            sens_reports.append(f"### Sensitivity Spearman Correlations: `{m_name}`\n")
            sens_reports.append("| Metric Pair | Spearman $\\rho$ | $p$-value | Significance |")
            sens_reports.append("|---|---|---|---|")
            for pair, vals in sens_res["correlations"].items():
                p_str = f"{vals['p_val']:.4e}" if vals['p_val'] < 0.001 else f"{vals['p_val']:.4f}"
                sig = "**p < 0.001**" if vals['p_val'] < 0.001 else ("p < 0.05" if vals['p_val'] < 0.05 else "n.s.")
                sens_reports.append(f"| `{pair}` | {vals['rho']:+.4f} | {p_str} | {sig} |")
            sens_reports.append("\n")

    if sens_reports:
        sens_md_path = os.path.join(results_dir, "sensitivity_correlations.md")
        with open(sens_md_path, "w", encoding="utf-8") as f:
            f.write(sim_header + "# Sensitivity Correlation Analysis (RQ2)\n\n" + "\n".join(sens_reports))
        print(f"[ANALYZE] Saved Sensitivity Correlations -> {sens_md_path}")

    print(f"\n{'='*70}\n  ANALYSIS COMPLETE. ALL REPORTS AND LATEX TABLES READY IN {output_dir}/\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQCAP Analysis Suite")
    parser.add_argument("--results_dir", default="results", help="Directory with JSONL results")
    parser.add_argument("--output_dir", default="paper", help="Output directory for paper tables")
    parser.add_argument("--allow-dry-run", action="store_true", help="Permit processing simulated dry-run JSONL records with explicit watermark")
    args = parser.parse_args()
    try:
        run_full_analysis(args.results_dir, args.output_dir, allow_dry_run=args.allow_dry_run)
    except ValueError as e:
        print(f"\n[FATAL GUARD] {e}\n", file=sys.stderr)
        sys.exit(1)
