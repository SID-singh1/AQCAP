"""
Unit tests for aqcap.scoring module.
Verifies prompt construction, target token ID resolution, and 2-way logit scoring math.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
from unittest.mock import MagicMock
import torch
from aqcap.scoring import (
    build_prompt,
    get_target_token_ids,
    score_prompt,
    YES_CANDIDATES,
    NO_CANDIDATES,
    FIXED_INSTRUCTION_TEMPLATE,
)


class DummyTokenizer:
    """Mock tokenizer simulating HuggingFace tokenizer behavior."""
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.chat_template = "{{ messages[0]['content'] }}"
        self.vocab = {
            "YES": 101, "Yes": 102, "yes": 103, " YES": 104, " Yes": 105,
            "NO": 201, "No": 202, "no": 203, " NO": 204, " No": 205,
            "other": 999,
        }

    def encode(self, text, add_special_tokens=False):
        for k, v in self.vocab.items():
            if text == k or text.strip() == k:
                return [v]
        return [999]

    def decode(self, token_ids):
        rev = {v: k for k, v in self.vocab.items()}
        return "".join([rev.get(t, "<unk>") for t in token_ids])

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        formatted = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return f"<chat>{formatted}</chat>"

    def __call__(self, text, return_tensors="pt", truncation=False):
        return {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()

    def test_build_prompt_uses_chat_template(self):
        code_snippet = "void vuln() { char b[4]; strcpy(b, s); }"
        prompt = build_prompt(self.tokenizer, code_snippet)
        self.assertIn("<chat>", prompt)
        self.assertIn("```c", prompt)
        self.assertIn(code_snippet, prompt)

    def test_get_target_token_ids(self):
        yes_ids, no_ids = get_target_token_ids(self.tokenizer)
        self.assertTrue(len(yes_ids) > 0)
        self.assertTrue(len(no_ids) > 0)
        # Verify no intersection between YES and NO IDs
        self.assertEqual(len(set(yes_ids).intersection(set(no_ids))), 0)

    def test_get_target_token_ids_removes_whitespace_and_overlap(self):
        # Simulates Phi-3 / SentencePiece behavior where space prefix token 29871
        # is generated for both ' YES' and ' NO'
        class SentencePieceMockTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text == "YES":
                    return [101]
                elif text == " YES":
                    return [29871, 101]
                elif text == "NO":
                    return [201]
                elif text == " NO":
                    return [29871, 201]
                return [999]

            def decode(self, token_ids):
                mapping = {101: "YES", 201: "NO", 29871: " ", 999: "other"}
                return "".join([mapping.get(t, "") for t in token_ids])

        tok = SentencePieceMockTokenizer()
        yes_ids, no_ids = get_target_token_ids(tok)

        # 29871 is whitespace AND shared -> must NOT be in either
        self.assertNotIn(29871, yes_ids)
        self.assertNotIn(29871, no_ids)
        self.assertIn(101, yes_ids)
        self.assertIn(201, no_ids)
        self.assertEqual(len(set(yes_ids).intersection(set(no_ids))), 0)

    def test_build_prompt_2_shot(self):
        code_query = "int query_func() { return 0; }"
        few_shots = [
            {"code": "void bad() { gets(b); }", "label": "YES"},
            {"code": "void good() { fgets(b, s, f); }", "label": "NO"},
        ]
        prompt = build_prompt(self.tokenizer, code_query, mode="2_shot", few_shot_examples=few_shots)
        self.assertIn("gets(b)", prompt)
        self.assertIn("fgets(b, s, f)", prompt)
        self.assertIn(code_query, prompt)

    def test_score_prompt_logit_math(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = torch.nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, **kwargs):
                vocab_size = 1000
                logits = torch.zeros((1, input_ids.shape[1], vocab_size))
                # Set YES token 101 to high logit (+10.0), NO token 201 to low logit (-10.0)
                logits[0, -1, 101] = 10.0
                logits[0, -1, 201] = -10.0
                return MagicMock(logits=logits)

        model = DummyModel()
        yes_ids = [101]
        no_ids = [201]
        result = score_prompt(model, self.tokenizer, "test prompt", yes_ids=yes_ids, no_ids=no_ids)

        self.assertGreater(result["p_vuln"], 0.99)
        self.assertLess(result["p_safe"], 0.01)
        self.assertEqual(result["pred_50"], 1)
        self.assertEqual(result["top1_id"], 101)
        self.assertTrue(result["is_in_yes_no"])


if __name__ == "__main__":
    unittest.main()
