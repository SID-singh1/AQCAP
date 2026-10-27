"""
Unit tests for data loaders, splits, and signal gate components.
"""

import os
import unittest
import numpy as np

from aqcap.data import (
    load_dataset_split,
    get_shortest_calib_demonstration,
)
from scripts.signal_gate import render_gate_table


class TestDataAndSignalGate(unittest.TestCase):
    def test_load_dataset_splits_calib(self):
        # 1. primevul_paired
        pv_p = load_dataset_split("primevul_paired", split="calib")
        self.assertTrue(pv_p["is_paired"])
        self.assertEqual(len(pv_p["samples"]), 200)
        self.assertEqual(len(pv_p["pairs"]), 100)
        targets_pv = [s["target"] for s in pv_p["samples"]]
        self.assertEqual(targets_pv.count(1), 100)
        self.assertEqual(targets_pv.count(0), 100)

        # 2. primevul_unpaired
        pv_unp = load_dataset_split("primevul_unpaired", split="calib")
        self.assertFalse(pv_unp["is_paired"])
        self.assertEqual(len(pv_unp["samples"]), 200)
        targets_unp = [s["target"] for s in pv_unp["samples"]]
        self.assertEqual(targets_unp.count(1), 100)
        self.assertEqual(targets_unp.count(0), 100)

        # 3. juliet
        juliet = load_dataset_split("juliet", split="calib")
        self.assertTrue(juliet["is_paired"])
        self.assertEqual(len(juliet["samples"]), 200)
        self.assertEqual(len(juliet["pairs"]), 100)
        targets_j = [s["target"] for s in juliet["samples"]]
        self.assertEqual(targets_j.count(1), 100)
        self.assertEqual(targets_j.count(0), 100)

    def test_shortest_calib_demonstrations(self):
        for ds in ["primevul_paired", "primevul_unpaired", "juliet"]:
            demo = get_shortest_calib_demonstration(ds)
            self.assertIn("vuln_idx", demo)
            self.assertIn("safe_idx", demo)
            self.assertIn("vuln_code", demo)
            self.assertIn("safe_code", demo)
            self.assertGreater(len(demo["vuln_code"]), 0)
            self.assertGreater(len(demo["safe_code"]), 0)
            self.assertNotEqual(demo["vuln_idx"], demo["safe_idx"])

    def test_render_gate_table(self):
        dummy_results = [
            {
                "model": "microsoft/Phi-3-mini-4k-instruct",
                "dataset": "primevul_paired",
                "mode": "zero_shot",
                "metrics": {
                    "auroc": 0.4814,
                    "auroc_ci_low": 0.400,
                    "auroc_ci_high": 0.560,
                    "auprc": 0.4900,
                    "mean_p_vuln": 0.5123,
                    "pc": 12.0,
                    "pv": 45.0,
                    "fpr_at_90_recall": 0.9500,
                    "fpr_at_95_recall": 0.9800,
                    "top1_compliance_pct": 98.5,
                }
            },
            {
                "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "dataset": "primevul_unpaired",
                "mode": "2_shot",
                "metrics": {
                    "auroc": 0.6850,
                    "auroc_ci_low": 0.610,
                    "auroc_ci_high": 0.755,
                    "auprc": 0.7100,
                    "mean_p_vuln": 0.4350,
                    "pc": None,
                    "pv": None,
                    "fpr_at_90_recall": 0.6500,
                    "fpr_at_95_recall": 0.8000,
                    "top1_compliance_pct": 100.0,
                }
            }
        ]
        md_table, csv_table = render_gate_table(dummy_results)
        self.assertIn("Phi-3-mini-4k-instruct", md_table)
        self.assertIn("Qwen2.5-Coder-7B-Instruct", md_table)
        self.assertIn("0.4814 (0.400, 0.560)", md_table)
        self.assertIn("0.6850 (0.610, 0.755)", md_table)
        self.assertIn("AUROC (95% CI)", md_table)
        self.assertIn("FPR@90% Rec", md_table)
        self.assertIn("FPR@95% Rec", md_table)
        self.assertIn("P-C (%)", md_table)
        self.assertIn("P-V (%)", md_table)
        self.assertIn("AUPRC", csv_table)


if __name__ == "__main__":
    unittest.main()
