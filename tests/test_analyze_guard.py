"""
Unit test for AQCAP analyze.py integrity guard.
Tests:
1. analyze.py refuses any JSONL containing dry-run metadata without --allow-dry-run.
2. analyze.py permits dry-run data when allow_dry_run=True and watermarks all captions and headers.
3. Clean real data is analyzed without refusal and without simulated watermarks.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import tempfile
import unittest
import numpy as np

from scripts.analyze import run_full_analysis


class TestAnalyzeGuard(unittest.TestCase):
    def _create_mock_jsonl(self, dir_path: str, is_dry_run: bool = False):
        filepath = os.path.join(dir_path, "phi3_fp16_test_cap_off.jsonl")
        meta = {
            "type": "metadata",
            "run_id": "phi3_fp16_test_cap_off",
            "model": "microsoft/Phi-3-mini-4k-instruct",
            "config_name": "fp16",
            "config": {"name": "fp16"},
            "split": "test",
            "cap": "off",
            "weights_mb": 7640.0,
            "is_dry_run": is_dry_run,
            "env": {"dry_run": is_dry_run, "device": "cpu"},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
            # 2 pairs (4 samples)
            samples = [
                {"type": "sample", "idx": 1, "pair_id": "p1", "target": 1, "p_vuln": 0.85, "cwe": "CWE-119", "project": "proj", "latency_s": 0.3},
                {"type": "sample", "idx": 2, "pair_id": "p1", "target": 0, "p_vuln": 0.15, "cwe": "CWE-119", "project": "proj", "latency_s": 0.3},
                {"type": "sample", "idx": 3, "pair_id": "p2", "target": 1, "p_vuln": 0.75, "cwe": "CWE-119", "project": "proj", "latency_s": 0.3},
                {"type": "sample", "idx": 4, "pair_id": "p2", "target": 0, "p_vuln": 0.25, "cwe": "CWE-119", "project": "proj", "latency_s": 0.3},
            ]
            for s in samples:
                f.write(json.dumps(s) + "\n")
        return filepath

    def test_guard_refuses_dry_run_data_without_flag(self):
        with tempfile.TemporaryDirectory() as res_dir, tempfile.TemporaryDirectory() as out_dir:
            self._create_mock_jsonl(res_dir, is_dry_run=True)
            # Must raise ValueError when allow_dry_run=False
            with self.assertRaises(ValueError) as ctx:
                run_full_analysis(res_dir, out_dir, allow_dry_run=False)
            self.assertIn("REFUSAL ERROR", str(ctx.exception))
            self.assertIn("simulated dry-run records", str(ctx.exception))

    def test_guard_permits_dry_run_with_flag_and_watermarks(self):
        with tempfile.TemporaryDirectory() as res_dir, tempfile.TemporaryDirectory() as out_dir:
            self._create_mock_jsonl(res_dir, is_dry_run=True)
            # When allow_dry_run=True, execution succeeds
            run_full_analysis(res_dir, out_dir, allow_dry_run=True)

            # Check markdown table header
            md_path = os.path.join(res_dir, "main_table.md")
            self.assertTrue(os.path.isfile(md_path))
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            self.assertIn("SIMULATED — NOT REAL DATA", md_content)

            # Check LaTeX table caption watermark
            tex_path = os.path.join(out_dir, "table_main.tex")
            self.assertTrue(os.path.isfile(tex_path))
            with open(tex_path, "r", encoding="utf-8") as f:
                tex_content = f.read()
            self.assertIn(r"\textbf{SIMULATED --- NOT REAL DATA.}", tex_content)

    def test_clean_real_data_succeeds_without_watermark(self):
        with tempfile.TemporaryDirectory() as res_dir, tempfile.TemporaryDirectory() as out_dir:
            self._create_mock_jsonl(res_dir, is_dry_run=False)
            # Real data succeeds without allow_dry_run
            run_full_analysis(res_dir, out_dir, allow_dry_run=False)

            tex_path = os.path.join(out_dir, "table_main.tex")
            with open(tex_path, "r", encoding="utf-8") as f:
                tex_content = f.read()
            self.assertNotIn("SIMULATED", tex_content)


if __name__ == "__main__":
    unittest.main()
