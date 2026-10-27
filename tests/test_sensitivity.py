"""
Unit tests for aqcap.sensitivity module.
Tests binary KL divergence math, config generation, and saliency tracking.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import torch.nn.functional as F
import unittest
import numpy as np
import pandas as pd
from aqcap.sensitivity import (
    compute_binary_kl,
    compute_full_vocab_kl,
    generate_sensitivity_configs,
)


class TestSensitivity(unittest.TestCase):
    def test_full_vocab_kl_identical_models(self):
        # Requirement: identical models -> KL == 0
        seq_len, vocab_size = 15, 200
        torch.manual_seed(42)
        dummy_logits = torch.randn(seq_len, vocab_size)
        p_ref = F.softmax(dummy_logits, dim=-1)
        log_p_ref = F.log_softmax(dummy_logits, dim=-1)
        log_q_identical = log_p_ref.clone()

        kl_zero = compute_full_vocab_kl(p_ref, log_p_ref, log_q_identical)
        self.assertEqual(kl_zero, 0.0)

        # Perturbed model -> KL > 0
        perturbed_logits = dummy_logits + torch.randn(seq_len, vocab_size) * 0.5
        log_q_perturbed = F.log_softmax(perturbed_logits, dim=-1)
        kl_positive = compute_full_vocab_kl(p_ref, log_p_ref, log_q_perturbed)
        self.assertGreater(kl_positive, 0.0)

    def test_binary_kl_math(self):
        # KL between identical distributions is 0
        self.assertAlmostEqual(compute_binary_kl(0.7, 0.7), 0.0, places=5)
        # KL is non-negative and positive when distributions differ
        kl = compute_binary_kl(0.9, 0.1)
        self.assertGreater(kl, 1.0)
        # Boundary stability check (p=1.0, q=0.0)
        kl_boundary = compute_binary_kl(1.0, 0.0)
        self.assertTrue(np.isfinite(kl_boundary))

    def test_generate_sensitivity_configs(self):
        # Create a mock sensitivity dataframe with 32 layers
        layers = list(range(32))
        s_task = [float(i) for i in layers]       # layers 16-31 are highest
        s_gen = [float(31 - i) for i in layers]   # layers 0-15 are highest
        s_l2 = [10.0 + i for i in layers]

        df = pd.DataFrame({
            "layer": layers,
            "S_task": s_task,
            "delta_auroc": [0.01 * i for i in layers],
            "S_gen": s_gen,
            "S_l2": s_l2,
        })

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = generate_sensitivity_configs(df, configs_dir=tmp_dir, num_layers=32, int8_budget=16)

            # 1. top_task: layers 16-31 should be INT8, 0-15 should be NF4
            with open(paths["top_task"], "r", encoding="utf-8") as f:
                cfg_tt = json.load(f)
            int8_tt = [int(k) for k, v in cfg_tt["precision_map"].items() if v == "int8"]
            self.assertEqual(len(int8_tt), 16)
            self.assertEqual(sorted(int8_tt), list(range(16, 32)))

            # 2. top_gen: layers 0-15 should be INT8, 16-31 should be NF4
            with open(paths["top_gen"], "r", encoding="utf-8") as f:
                cfg_tg = json.load(f)
            int8_tg = [int(k) for k, v in cfg_tg["precision_map"].items() if v == "int8"]
            self.assertEqual(len(int8_tg), 16)
            self.assertEqual(sorted(int8_tg), list(range(0, 16)))

            # 3. bottom_task: layers 0-15 should be INT8 (least sensitive), 16-31 should be NF4
            with open(paths["bottom_task"], "r", encoding="utf-8") as f:
                cfg_bt = json.load(f)
            int8_bt = [int(k) for k, v in cfg_bt["precision_map"].items() if v == "int8"]
            self.assertEqual(len(int8_bt), 16)
            self.assertEqual(sorted(int8_bt), list(range(0, 16)))


if __name__ == "__main__":
    unittest.main()
