"""
Unit tests for aqcap.quant module.
Tests layer discovery, architecture auditing, and memory reporting.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from aqcap.quant import get_decoder_layers, memory_report, audit_model


class DummyDecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 64)
        )


class DummyModel(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.model = MagicMock()
        self.model.layers = nn.ModuleList([DummyDecoderBlock() for _ in range(num_layers)])
        self.model.embed_tokens = nn.Embedding(100, 64)
        self.lm_head = nn.Linear(64, 100)


class TestQuant(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel(num_layers=4)

    def test_get_decoder_layers(self):
        layers = get_decoder_layers(self.model)
        self.assertEqual(len(layers), 4)

    def test_memory_report(self):
        report = memory_report(self.model)
        self.assertIn("weights_bytes", report)
        self.assertIn("peak_bytes", report)
        self.assertGreater(report["weights_bytes"], 0)

    def test_audit_model(self):
        # All layers are pure nn.Linear (fp16)
        precision_map = {0: "fp16", 1: "fp16", 2: "fp16", 3: "fp16"}
        audit = audit_model(self.model, precision_map)
        self.assertEqual(audit["layers_audited"], 4)
        self.assertEqual(audit["mismatches"], 0)
        self.assertEqual(audit["counts"]["fp16"], 24)  # 6 linears per block * 4 blocks

        # Inject mismatch
        mismatched_map = {0: "nf4", 1: "fp16", 2: "fp16", 3: "fp16"}
        audit_mismatch = audit_model(self.model, mismatched_map)
        self.assertGreater(audit_mismatch["mismatches"], 0)

    def test_hqq_linear_patch_and_migration(self):
        try:
            from aqcap.quant import _create_hqq_linear, _patch_hqq_linear, HQQLinear
        except Exception:
            self.skipTest("HQQ not available")

        if HQQLinear is None:
            self.skipTest("HQQLinear not imported")

        lin = nn.Linear(32, 32)
        hlin = _create_hqq_linear(lin, nbits=4, group_size=32, device="cpu")
        self.assertIsNotNone(hlin)
        self.assertEqual(hlin.W_q.device.type, "cpu")

        # Test monkey-patched .to("cpu")
        hlin.to("cpu")
        self.assertEqual(hlin.W_q.device.type, "cpu")
        self.assertEqual(hlin.meta["scale"].device.type, "cpu")

        # Test forward pass with half precision
        x = torch.randn(2, 32, dtype=torch.float16)
        out = hlin(x)
        self.assertEqual(out.shape, (2, 32))


if __name__ == "__main__":
    unittest.main()
