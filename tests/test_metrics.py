"""
Unit tests for aqcap.metrics module.
Tests classification metrics, PrimeVul paired metrics, paired bootstrap CIs, and McNemar tests.
"""

import unittest
import numpy as np
from aqcap.metrics import (
    compute_classification_metrics,
    find_optimal_calib_threshold,
    compute_paired_metrics,
    compute_quantization_degradation,
    bootstrap_paired_confidence_intervals,
    mcnemar_test_vs_nf4,
    compute_systems_latency,
    reconstruct_pairs,
)


class TestMetrics(unittest.TestCase):
    def test_classification_metrics_perfect_predictions(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        m = compute_classification_metrics(y_true, y_prob, threshold=0.5)

        self.assertEqual(m["auroc"], 1.0)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["balanced_accuracy"], 1.0)
        self.assertEqual(m["mcc"], 1.0)

    def test_optimal_calib_threshold(self):
        y_true = np.array([1, 1, 0, 0])
        # Suppose model is biased and outputs probabilities between 0.6 and 0.85
        y_prob = np.array([0.85, 0.80, 0.65, 0.62])
        best_t = find_optimal_calib_threshold(y_true, y_prob)
        # Threshold between 0.65 and 0.80 will yield 100% balanced accuracy
        self.assertTrue(0.65 <= best_t <= 0.80)

    def test_paired_metrics(self):
        pairs_data = [
            {"pair_id": "1", "p_vuln_for_vuln": 0.9, "p_vuln_for_safe": 0.1},  # P-C: both correct
            {"pair_id": "2", "p_vuln_for_vuln": 0.9, "p_vuln_for_safe": 0.8},  # P-V: both vuln
            {"pair_id": "3", "p_vuln_for_vuln": 0.2, "p_vuln_for_safe": 0.1},  # P-B: both safe
            {"pair_id": "4", "p_vuln_for_vuln": 0.1, "p_vuln_for_safe": 0.9},  # P-R: reversed
        ]
        pm = compute_paired_metrics(pairs_data, threshold=0.5)
        self.assertEqual(pm["P-C"], 25.0)
        self.assertEqual(pm["P-V"], 25.0)
        self.assertEqual(pm["P-B"], 25.0)
        self.assertEqual(pm["P-R"], 25.0)

    def test_quantization_degradation(self):
        cfg = [
            {"idx": 1, "p_vuln": 0.9},
            {"idx": 2, "p_vuln": 0.3},
        ]
        fp16 = [
            {"idx": 1, "p_vuln": 0.9},
            {"idx": 2, "p_vuln": 0.7},  # Flipped across 0.5 threshold
        ]
        deg = compute_quantization_degradation(cfg, fp16, threshold=0.5)
        self.assertEqual(deg["flip_rate"], 50.0)
        self.assertAlmostEqual(deg["mean_abs_delta_p"], 0.2, places=4)

    def test_paired_bootstrap(self):
        pairs_data = [
            {"pair_id": str(i), "p_vuln_for_vuln": 0.8, "p_vuln_for_safe": 0.2}
            for i in range(20)
        ]
        ci = bootstrap_paired_confidence_intervals(pairs_data, n_iterations=50, seed=42)
        self.assertIn("auroc", ci)
        low, high = ci["auroc"]
        self.assertTrue(low <= high)

    def test_mcnemar_test(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_cfg = np.array([1, 1, 0, 0, 1, 0])  # all correct
        y_nf4 = np.array([1, 0, 1, 0, 0, 0])  # 3 errors

        res = mcnemar_test_vs_nf4(y_true, y_cfg, y_nf4)
        self.assertIn("p_value", res)
        self.assertEqual(res["c_cfg_correct_nf4_wrong"], 3)
        self.assertEqual(res["b_cfg_wrong_nf4_correct"], 0)

    def test_systems_latency(self):
        latencies = [0.1, 0.2, 0.3, 0.4, 0.5]
        sl = compute_systems_latency(latencies)
        self.assertAlmostEqual(sl["median_s"], 0.3)
        self.assertGreater(sl["p90_s"], 0.4)

    def test_reconstruct_pairs(self):
        samples = [
            {"idx": 1, "pair_id": "pair_100", "target": 1, "p_vuln": 0.85, "cwe": "CWE-119", "project": "linux"},
            {"idx": 2, "pair_id": "pair_100", "target": 0, "p_vuln": 0.15, "cwe": "CWE-119", "project": "linux"},
            {"idx": 3, "pair_id": "pair_101", "target": 1, "p_vuln": 0.70, "cwe": "CWE-787", "project": "qemu"},
            {"idx": 4, "pair_id": "pair_101", "target": 0, "p_vuln": 0.30, "cwe": "CWE-787", "project": "qemu"},
        ]
        pairs = reconstruct_pairs(samples)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["pair_id"], "pair_100")
        self.assertEqual(pairs[0]["p_vuln_for_vuln"], 0.85)
        self.assertEqual(pairs[0]["p_vuln_for_safe"], 0.15)
        self.assertEqual(pairs[0]["cwe"], "CWE-119")
        self.assertEqual(pairs[0]["project"], "linux")

    def test_threshold_at_recall_and_flip_table(self):
        from aqcap.metrics import compute_threshold_at_recall, compute_flip_table
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        # Vulnerable probabilities
        y_prob_ref = np.array([0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.15, 0.10])
        # tau_90 achieves >= 90% recall on the 5 positive samples
        tau_90 = compute_threshold_at_recall(y_true, y_prob_ref, target_recall=0.90)
        # Positive samples with p >= tau_90 should be at least 4 out of 5 (80-90%+)
        self.assertGreaterEqual(np.mean(y_prob_ref[:5] >= tau_90), 0.80)

        # Config where sample 4 flipped from vuln to safe (0.60 -> 0.40)
        y_prob_cfg = np.array([0.95, 0.90, 0.85, 0.80, 0.40, 0.70, 0.30, 0.20, 0.15, 0.10])
        flips = compute_flip_table(y_true, y_prob_cfg, y_prob_ref, threshold=tau_90)
        self.assertIn("vuln_to_safe", flips)
        self.assertIn("safe_to_vuln", flips)
        self.assertGreaterEqual(flips["total_flips"], 1)

    def test_paired_bootstrap_with_tost(self):
        from aqcap.metrics import generate_bootstrap_resample_indices, paired_bootstrap_auroc_diff_with_tost
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob_ref = np.array([0.9, 0.8, 0.85, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05])
        # Nearly identical predictions
        y_prob_cfg = y_prob_ref + 0.01 * np.random.randn(len(y_true))
        resample_idx = generate_bootstrap_resample_indices(len(y_true), n_iterations=100, seed=42)
        tost = paired_bootstrap_auroc_diff_with_tost(y_true, y_prob_cfg, y_prob_ref, resample_idx, tost_margin=0.05)
        self.assertIn("mean_delta_auroc", tost)
        self.assertIn("is_equivalent", tost)
        self.assertIn("tost_p", tost)


if __name__ == "__main__":
    unittest.main()
