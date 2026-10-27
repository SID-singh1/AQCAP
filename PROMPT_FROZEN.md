# AQCAP — Frozen Evaluation Prompt Specification

**Status:** FROZEN  
**Target Split:** Calibrated on `calib` (100 PrimeVul pairs = 200 functions); strictly applied to `test` (326 pairs = 652 functions).  
**Protocol Compliance:** As mandated by [`ROADMAP.md`](./ROADMAP.md) Section 1 (Hard Rule #2 & #3), prompt structure, instruction phrasing, and logit candidate tokens are frozen prior to running test-set evaluation. No modifications to prompt wording are permitted after viewing test results.

---

## 1. Frozen Instruction Phrasing

The following instruction is applied universally across all models (Phi-3-mini and Qwen2.5-Coder) and all quantization configurations:

```text
Is the following function vulnerable to a security weakness? Answer with exactly one word: YES or NO.

```c
{code}
```
```

### Formatting Rules:
1. **Chat Template Standard:** The instruction is formatted using `tokenizer.apply_chat_template` with `add_generation_prompt=True` and `tokenize=False`.
2. **No Hardcoded Markers:** Hand-coded tags (such as `<|user|>`, `<|assistant|>`, or `<|im_start|>`) are forbidden to preserve architectural portability between Phi-3 and Qwen.
3. **Role Assignment:** Single-turn conversation with `role: "user"`.

---

## 2. Frozen Logit Extraction Specification

Inference bypasses generative text decoding (`generate()`). The vulnerability score $p_{\text{vuln}}$ is extracted from the model's logits at the final prompt token position:

### Affirmative (Vulnerable, Label 1) Tokens:
* Candidate set: `["YES", "Yes", "yes", " YES", " Yes"]`
* Resolved token IDs: Extracted via `tokenizer.encode(cand, add_special_tokens=False)[0]` and de-duplicated.

### Negative (Safe / Patched, Label 0) Tokens:
* Candidate set: `["NO", "No", "no", " NO", " No"]`
* Resolved token IDs: Extracted via `tokenizer.encode(cand, add_special_tokens=False)[0]` and de-duplicated.

### Probability Formulation:
$$\text{logits}_{\text{YES}} = \text{logsumexp}\left(\left[\text{logits}[t] \;\middle|\; t \in \text{YES\_IDs}\right]\right)$$
$$\text{logits}_{\text{NO}} = \text{logsumexp}\left(\left[\text{logits}[t] \;\middle|\; t \in \text{NO\_IDs}\right]\right)$$
$$p_{\text{vuln}} = \frac{e^{\text{logits}_{\text{YES}}}}{e^{\text{logits}_{\text{YES}}} + e^{\text{logits}_{\text{NO}}}}$$

---

## 3. Decision Thresholds

Two threshold evaluations are reported for every configuration on the test set:
1. **Standard Default Threshold:** $\tau = 0.5$ (prediction is vulnerable if $p_{\text{vuln}} \ge 0.5$).
2. **Calibrated Threshold:** $\tau_{\text{calib}} = \arg\max_{\tau} \text{BalancedAccuracy}(\text{calib}, \tau)$, selected exclusively on the 100-pair calibration set.
