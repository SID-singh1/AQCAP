"""
AQCAP Statistical & Evaluation Metrics Suite.

Implements all metrics mandated by ROADMAP.md Section 3.7:
1. Standard Classification Metrics: AUROC, AUPRC, Balanced Accuracy, MCC, Precision, Recall, F1.
2. Threshold-tuning on calibration data.
3. PrimeVul Paired Metrics: P-C (both correct), P-V (both vuln), P-B (both benign), P-R (reversed).
4. Quantization Degradation Metrics: Flip rate vs FP16, mean |Δp_vuln| vs FP16.
5. 1,000x Paired Bootstrap 95% Confidence Intervals (resampling PAIRS, not functions).
6. Paired Bootstrap AUROC difference vs uniform NF4.
7. McNemar's Test vs uniform NF4.
8. Systems Latency & Hardware Footprint Reporting.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
)
from scipy.stats import chi2, binom


def reconstruct_pairs(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups sample records by pair_id into structured pairs for paired metrics.
    
    Returns:
        List of dicts with keys: 'pair_id', 'p_vuln_for_vuln', 'p_vuln_for_safe', 'cwe', 'project'.
    """
    by_pair = defaultdict(dict)
    for s in samples:
        pair_id = s.get("pair_id")
        if not pair_id:
            continue
        if s.get("target") == 1:
            by_pair[pair_id]["vuln"] = s
        elif s.get("target") == 0:
            by_pair[pair_id]["safe"] = s

    pairs = []
    for pair_id, group in by_pair.items():
        if "vuln" in group and "safe" in group:
            pairs.append({
                "pair_id": pair_id,
                "p_vuln_for_vuln": group["vuln"]["p_vuln"],
                "p_vuln_for_safe": group["safe"]["p_vuln"],
                "cwe": group["vuln"].get("cwe", "Unknown"),
                "project": group["vuln"].get("project", "Unknown"),
            })
    return pairs


def compute_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes threshold-independent (AUROC, AUPRC) and thresholded metrics.
    Accepts lists or numpy arrays for y_true and y_prob.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Threshold-independent metrics
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = float("nan")

    # Thresholded metrics
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    fpr_90 = compute_fpr_at_recall(y_true, y_prob, target_recall=0.90)
    fpr_95 = compute_fpr_at_recall(y_true, y_prob, target_recall=0.95)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "balanced_accuracy": bacc,
        "mcc": mcc,
        "threshold": threshold,
        "fpr_at_90_recall": fpr_90,
        "fpr_at_95_recall": fpr_95,
    }


def compute_fpr_at_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.90) -> float:
    """
    Computes the minimum False Positive Rate (FPR) required to achieve at least target_recall.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid_idx = np.where(tpr >= target_recall)[0]
    if len(valid_idx) == 0:
        return 1.0
    return float(np.min(fpr[valid_idx]))


def compute_pr_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Computes precision-recall curve points and AUPRC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"auprc": float("nan"), "precision": [], "recall": []}
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    auprc = float(average_precision_score(y_true, y_prob))
    return {
        "auprc": auprc,
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "thresholds": thresholds.tolist(),
    }


def find_optimal_calib_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Finds the probability threshold tau in [0.05, 0.95] that maximizes
    balanced accuracy on the calibration set.
    """
    best_thresh = 0.5
    best_bacc = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= t).astype(int)
        bacc = balanced_accuracy_score(y_true, y_pred)
        if bacc > best_bacc:
            best_bacc = bacc
            best_thresh = float(t)
    return best_thresh


def compute_paired_metrics(
    pairs_data: List[Dict[str, Any]],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes PrimeVul paired metrics:
    - P-C: % of pairs where both vulnerable and benign functions are correctly classified.
    - P-V: % of pairs where both are predicted vulnerable (over-prediction / false alarm).
    - P-B: % of pairs where both are predicted benign (under-prediction / blind).
    - P-R: % of pairs where predictions are inverted (reversed: safe predicted vuln, vuln predicted safe).

    Args:
        pairs_data: List of dicts, each with keys:
                    'pair_id', 'p_vuln_for_vuln', 'p_vuln_for_safe'
    """
    total = len(pairs_data)
    if total == 0:
        return {"P-C": 0.0, "P-V": 0.0, "P-B": 0.0, "P-R": 0.0, "n_pairs": 0}

    n_pc = 0
    n_pv = 0
    n_pb = 0
    n_pr = 0

    for p in pairs_data:
        pred_v = int(p["p_vuln_for_vuln"] >= threshold)
        pred_s = int(p["p_vuln_for_safe"] >= threshold)

        # Truth for vuln is 1; truth for safe is 0
        if pred_v == 1 and pred_s == 0:
            n_pc += 1  # Both correct
        elif pred_v == 1 and pred_s == 1:
            n_pv += 1  # Both predicted vulnerable
        elif pred_v == 0 and pred_s == 0:
            n_pb += 1  # Both predicted benign
        elif pred_v == 0 and pred_s == 1:
            n_pr += 1  # Inverted / Reversed

    return {
        "P-C": (n_pc / total) * 100.0,
        "P-V": (n_pv / total) * 100.0,
        "P-B": (n_pb / total) * 100.0,
        "P-R": (n_pr / total) * 100.0,
        "n_pairs": total,
    }


def compute_quantization_degradation(
    config_samples: List[Dict[str, Any]],
    fp16_samples: List[Dict[str, Any]],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Measures prediction flip rate and probability drift relative to FP16 reference.
    Matches samples by 'idx'.
    """
    fp16_map = {s["idx"]: s for s in fp16_samples}
    flips = 0
    delta_p_list = []
    matched = 0

    for s in config_samples:
        idx = s["idx"]
        if idx not in fp16_map:
            continue
        matched += 1
        ref = fp16_map[idx]

        pred_curr = int(s["p_vuln"] >= threshold)
        pred_ref = int(ref["p_vuln"] >= threshold)

        if pred_curr != pred_ref:
            flips += 1

        delta_p_list.append(abs(s["p_vuln"] - ref["p_vuln"]))

    if matched == 0:
        return {"flip_rate": 0.0, "mean_abs_delta_p": 0.0, "matched_samples": 0}

    return {
        "flip_rate": (flips / matched) * 100.0,
        "mean_abs_delta_p": float(np.mean(delta_p_list)),
        "matched_samples": matched,
    }


def bootstrap_paired_confidence_intervals(
    pairs_data: List[Dict[str, Any]],
    n_iterations: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% Bootstrap Confidence Intervals by resampling PAIRS with replacement.
    Ensures that vulnerable and patched versions of each function are sampled together.
    """
    rng = np.random.RandomState(seed)
    n_pairs = len(pairs_data)
    boot_metrics = {
        "auroc": [], "auprc": [], "f1": [], "balanced_accuracy": [], "mcc": [],
        "P-C": [], "P-V": [], "P-B": [], "P-R": []
    }

    for _ in range(n_iterations):
        sample_indices = rng.choice(n_pairs, size=n_pairs, replace=True)
        resampled_pairs = [pairs_data[i] for i in sample_indices]

        # Flatten resampled pairs into classification arrays
        y_true = []
        y_prob = []
        for p in resampled_pairs:
            y_true.extend([1, 0])
            y_prob.extend([p["p_vuln_for_vuln"], p["p_vuln_for_safe"]])

        y_true_np = np.array(y_true)
        y_prob_np = np.array(y_prob)

        cls_m = compute_classification_metrics(y_true_np, y_prob_np, threshold=threshold)
        pair_m = compute_paired_metrics(resampled_pairs, threshold=threshold)

        for k in ["auroc", "auprc", "f1", "balanced_accuracy", "mcc"]:
            boot_metrics[k].append(cls_m[k])
        for k in ["P-C", "P-V", "P-B", "P-R"]:
            boot_metrics[k].append(pair_m[k])

    ci_results = {}
    for k, values in boot_metrics.items():
        arr = np.array([v for v in values if not np.isnan(v)])
        if len(arr) == 0:
            ci_results[k] = (float("nan"), float("nan"))
        else:
            low = float(np.percentile(arr, 2.5))
            high = float(np.percentile(arr, 97.5))
            ci_results[k] = (low, high)

    return ci_results


def bootstrap_unpaired_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% Bootstrap Confidence Intervals by resampling individual samples with replacement.
    Used for unpaired datasets.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    boot_metrics = {
        "auroc": [], "auprc": [], "f1": [], "balanced_accuracy": [], "mcc": [],
        "fpr_at_90_recall": [], "fpr_at_95_recall": []
    }

    for _ in range(n_iterations):
        sample_indices = rng.choice(n, size=n, replace=True)
        res_true = y_true[sample_indices]
        res_prob = y_prob[sample_indices]
        if len(np.unique(res_true)) < 2:
            continue
        cls_m = compute_classification_metrics(res_true, res_prob, threshold=threshold)
        for k in boot_metrics.keys():
            boot_metrics[k].append(cls_m[k])

    ci_results = {}
    for k, values in boot_metrics.items():
        arr = np.array([v for v in values if not np.isnan(v)])
        if len(arr) == 0:
            ci_results[k] = (float("nan"), float("nan"))
        else:
            low = float(np.percentile(arr, 2.5))
            high = float(np.percentile(arr, 97.5))
            ci_results[k] = (low, high)

    return ci_results


def paired_bootstrap_auroc_diff(
    pairs_cfg: List[Dict[str, Any]],
    pairs_nf4: List[Dict[str, Any]],
    n_iterations: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes the paired bootstrap difference: Delta AUROC = AUROC(config) - AUROC(nf4).
    Returns mean delta, 95% CI, and empirical p-value.
    """
    rng = np.random.RandomState(seed)
    n_pairs = len(pairs_cfg)
    diffs = []

    for _ in range(n_iterations):
        idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        # Config
        y_true_c = []
        y_prob_c = []
        for i in idx:
            y_true_c.extend([1, 0])
            y_prob_c.extend([pairs_cfg[i]["p_vuln_for_vuln"], pairs_cfg[i]["p_vuln_for_safe"]])
        # NF4
        y_true_n = []
        y_prob_n = []
        for i in idx:
            y_true_n.extend([1, 0])
            y_prob_n.extend([pairs_nf4[i]["p_vuln_for_vuln"], pairs_nf4[i]["p_vuln_for_safe"]])

        try:
            auc_c = roc_auc_score(y_true_c, y_prob_c)
            auc_n = roc_auc_score(y_true_n, y_prob_n)
            diffs.append(auc_c - auc_n)
        except ValueError:
            pass

    diff_arr = np.array(diffs)
    mean_diff = float(np.mean(diff_arr))
    ci_low = float(np.percentile(diff_arr, 2.5))
    ci_high = float(np.percentile(diff_arr, 97.5))
    # p-value: fraction where difference is <= 0
    p_val = float(np.mean(diff_arr <= 0))

    return {
        "mean_delta_auroc": mean_diff,
        "ci_95": (ci_low, ci_high),
        "p_value": p_val,
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_cfg: np.ndarray,
    y_pred_ref: np.ndarray
) -> Dict[str, Any]:
    """
    Performs McNemar's test with continuity correction comparing two classifiers (e.g. cfg vs fp16).
    Contingency table on correct/incorrect classifications:
      b: config incorrect, reference correct
      c: config correct, reference incorrect
    """
    correct_cfg = (y_pred_cfg == y_true)
    correct_ref = (y_pred_ref == y_true)

    b = int(np.sum((~correct_cfg) & correct_ref))
    c = int(np.sum(correct_cfg & (~correct_ref)))

    total_discordant = b + c
    if total_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "b": b,
            "c": c,
            "b_cfg_wrong_ref_correct": b,
            "c_cfg_correct_ref_wrong": c,
        }

    if total_discordant < 25:
        # Exact binomial test
        p_val = float(2 * min(binom.cdf(min(b, c), total_discordant, 0.5), 0.5))
        stat = float(abs(b - c))
    else:
        # Edwards continuity-corrected chi-square
        stat = float(((abs(b - c) - 1.0) ** 2) / total_discordant)
        p_val = float(chi2.sf(stat, df=1))

    return {
        "statistic": stat,
        "p_value": p_val,
        "b": b,
        "c": c,
        "b_cfg_wrong_ref_correct": b,
        "c_cfg_correct_ref_wrong": c,
    }


def mcnemar_test_vs_fp16(
    y_true: np.ndarray,
    y_pred_cfg: np.ndarray,
    y_pred_fp16: np.ndarray
) -> Dict[str, Any]:
    """McNemar's test comparing a quantized configuration against the FP16 reference."""
    return mcnemar_test(y_true, y_pred_cfg, y_pred_fp16)


def mcnemar_test_vs_nf4(
    y_true: np.ndarray,
    y_pred_cfg: np.ndarray,
    y_pred_nf4: np.ndarray
) -> Dict[str, Any]:
    """McNemar's test comparing a quantized configuration against uniform NF4."""
    res = mcnemar_test(y_true, y_pred_cfg, y_pred_nf4)
    res["b_cfg_wrong_nf4_correct"] = res["b"]
    res["c_cfg_correct_nf4_wrong"] = res["c"]
    return res


def compute_systems_latency(latencies: List[float]) -> Dict[str, float]:
    """Computes median, p90, and mean latency from sample-level timings."""
    if not latencies:
        return {"median_s": 0.0, "p90_s": 0.0, "mean_s": 0.0}
    arr = np.array(latencies)
    return {
        "median_s": float(np.median(arr)),
        "p90_s": float(np.percentile(arr, 90)),
        "mean_s": float(np.mean(arr)),
    }


def compute_threshold_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.90
) -> float:
    """
    Finds the maximum decision threshold tau such that recall on positive (vulnerable)
    samples is at least target_recall.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos_probs = y_prob[y_true == 1]
    if len(pos_probs) == 0:
        return 0.5
    tau = float(np.percentile(pos_probs, (1.0 - target_recall) * 100.0))
    return tau


def generate_bootstrap_resample_indices(
    n_items: int,
    n_iterations: int = 2000,
    seed: int = 42
) -> np.ndarray:
    """
    Pre-generates a fixed (n_iterations, n_items) array of bootstrap sample indices.
    Using the exact same resample indices across all models/configs guarantees
    a true paired bootstrap comparison against FP16.
    """
    rng = np.random.RandomState(seed)
    return rng.choice(n_items, size=(n_iterations, n_items), replace=True)


def paired_bootstrap_auroc_diff_with_tost(
    y_true: np.ndarray,
    y_prob_cfg: np.ndarray,
    y_prob_ref: np.ndarray,
    resample_indices: np.ndarray,
    tost_margin: float = 0.02,
) -> Dict[str, Any]:
    """
    Computes paired bootstrap difference: Delta AUROC = AUROC(cfg) - AUROC(ref)
    using pre-drawn resample indices across 2000 replicates.
    Also performs Two One-Sided Tests (TOST) for equivalence within margin [-tost_margin, +tost_margin].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob_cfg = np.asarray(y_prob_cfg, dtype=float)
    y_prob_ref = np.asarray(y_prob_ref, dtype=float)
    n_iterations = len(resample_indices)

    diffs = []
    for b in range(n_iterations):
        idx = resample_indices[b]
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        yp_cfg = y_prob_cfg[idx]
        yp_ref = y_prob_ref[idx]
        try:
            auc_cfg = roc_auc_score(yt, yp_cfg)
            auc_ref = roc_auc_score(yt, yp_ref)
            diffs.append(auc_cfg - auc_ref)
        except ValueError:
            pass

    if len(diffs) == 0:
        return {
            "mean_delta_auroc": float("nan"),
            "ci_95": (float("nan"), float("nan")),
            "ci_90": (float("nan"), float("nan")),
            "p_val_superiority": float("nan"),
            "tost_p1": float("nan"),
            "tost_p2": float("nan"),
            "tost_p": float("nan"),
            "is_equivalent": False,
        }

    diff_arr = np.array(diffs)
    mean_diff = float(np.mean(diff_arr))
    ci_low = float(np.percentile(diff_arr, 2.5))
    ci_high = float(np.percentile(diff_arr, 97.5))
    ci_90_low = float(np.percentile(diff_arr, 5.0))
    ci_90_high = float(np.percentile(diff_arr, 95.0))

    p_sup = float(np.mean(diff_arr <= 0))
    p1 = float(np.mean(diff_arr <= -tost_margin))
    p2 = float(np.mean(diff_arr >= +tost_margin))
    p_tost = float(max(p1, p2))
    is_equivalent = bool(ci_90_low > -tost_margin and ci_90_high < tost_margin)

    return {
        "mean_delta_auroc": mean_diff,
        "ci_95": (ci_low, ci_high),
        "ci_90": (ci_90_low, ci_90_high),
        "p_val_superiority": p_sup,
        "tost_p1": p1,
        "tost_p2": p2,
        "tost_p": p_tost,
        "tost_margin": tost_margin,
        "is_equivalent": is_equivalent,
    }


def paired_bootstrap_auroc_diff_pairs_with_tost(
    pairs_cfg: List[Dict[str, Any]],
    pairs_ref: List[Dict[str, Any]],
    pair_resample_indices: np.ndarray,
    tost_margin: float = 0.02,
) -> Dict[str, Any]:
    """
    Computes paired bootstrap AUROC difference when data consists of paired samples (e.g. Juliet pairs),
    resampling pairs atomically with identical pre-generated pair indices.
    """
    n_iterations = len(pair_resample_indices)
    diffs = []

    for b in range(n_iterations):
        idx = pair_resample_indices[b]
        y_true_boot = []
        yp_cfg_boot = []
        yp_ref_boot = []

        for i in idx:
            y_true_boot.extend([1, 0])
            yp_cfg_boot.extend([pairs_cfg[i]["p_vuln_for_vuln"], pairs_cfg[i]["p_vuln_for_safe"]])
            yp_ref_boot.extend([pairs_ref[i]["p_vuln_for_vuln"], pairs_ref[i]["p_vuln_for_safe"]])

        try:
            auc_cfg = roc_auc_score(y_true_boot, yp_cfg_boot)
            auc_ref = roc_auc_score(y_true_boot, yp_ref_boot)
            diffs.append(auc_cfg - auc_ref)
        except ValueError:
            pass

    if len(diffs) == 0:
        return {
            "mean_delta_auroc": float("nan"),
            "ci_95": (float("nan"), float("nan")),
            "ci_90": (float("nan"), float("nan")),
            "p_val_superiority": float("nan"),
            "tost_p1": float("nan"),
            "tost_p2": float("nan"),
            "tost_p": float("nan"),
            "is_equivalent": False,
        }

    diff_arr = np.array(diffs)
    mean_diff = float(np.mean(diff_arr))
    ci_low = float(np.percentile(diff_arr, 2.5))
    ci_high = float(np.percentile(diff_arr, 97.5))
    ci_90_low = float(np.percentile(diff_arr, 5.0))
    ci_90_high = float(np.percentile(diff_arr, 95.0))

    p_sup = float(np.mean(diff_arr <= 0))
    p1 = float(np.mean(diff_arr <= -tost_margin))
    p2 = float(np.mean(diff_arr >= +tost_margin))
    p_tost = float(max(p1, p2))
    is_equivalent = bool(ci_90_low > -tost_margin and ci_90_high < tost_margin)

    return {
        "mean_delta_auroc": mean_diff,
        "ci_95": (ci_low, ci_high),
        "ci_90": (ci_90_low, ci_90_high),
        "p_val_superiority": p_sup,
        "tost_p1": p1,
        "tost_p2": p2,
        "tost_p": p_tost,
        "tost_margin": tost_margin,
        "is_equivalent": is_equivalent,
    }


def compute_flip_table(
    y_true: np.ndarray,
    y_prob_cfg: np.ndarray,
    y_prob_ref: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Computes prediction flip table vs reference (FP16) at threshold tau:
    - vuln_to_safe: ref predicted vuln (>= tau), cfg predicted safe (< tau)
    - safe_to_vuln: ref predicted safe (< tau), cfg predicted vuln (>= tau)
    - total_flips: vuln_to_safe + safe_to_vuln
    - flip_rate: total_flips / n_total
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob_cfg = np.asarray(y_prob_cfg, dtype=float)
    y_prob_ref = np.asarray(y_prob_ref, dtype=float)

    pred_cfg = (y_prob_cfg >= threshold).astype(int)
    pred_ref = (y_prob_ref >= threshold).astype(int)

    vuln_to_safe = int(np.sum((pred_ref == 1) & (pred_cfg == 0)))
    safe_to_vuln = int(np.sum((pred_ref == 0) & (pred_cfg == 1)))
    total_flips = vuln_to_safe + safe_to_vuln
    n_total = len(y_true)
    flip_rate = float(total_flips / max(n_total, 1))

    harmful_flips = int(np.sum((pred_ref == y_true) & (pred_cfg != y_true)))
    beneficial_flips = int(np.sum((pred_ref != y_true) & (pred_cfg == y_true)))

    return {
        "threshold": threshold,
        "n_samples": n_total,
        "vuln_to_safe": vuln_to_safe,
        "vuln_to_safe_pct": (vuln_to_safe / max(n_total, 1)) * 100.0,
        "safe_to_vuln": safe_to_vuln,
        "safe_to_vuln_pct": (safe_to_vuln / max(n_total, 1)) * 100.0,
        "total_flips": total_flips,
        "flip_rate_pct": flip_rate * 100.0,
        "harmful_flips": harmful_flips,
        "beneficial_flips": beneficial_flips,
        "net_shift": safe_to_vuln - vuln_to_safe,
    }
