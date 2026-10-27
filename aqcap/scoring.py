"""
AQCAP Scoring Engine.

Handles:
1. Chat template prompt construction (never hand-coding tags).
2. Deterministic 2-way logit probability scoring (YES vs NO) via a single forward pass (no generate()).
3. Top-1 token sanity monitoring.
4. Synchronized latency timing and token counting.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from typing import Dict, List, Tuple, Any, Optional

try:
    import torch
except ImportError:
    torch = None

# Canonical candidate tokens for affirmative (vulnerable) and negative (safe) responses
YES_CANDIDATES = ["YES", "Yes", "yes", " YES", " Yes"]
NO_CANDIDATES = ["NO", "No", "no", " NO", " No"]

FIXED_INSTRUCTION_TEMPLATE = (
    "Is the following function vulnerable to a security weakness? Answer with exactly one word: YES or NO.\n\n"
    "```c\n"
    "{code}\n"
    "```"
)


def build_prompt(
    tokenizer,
    code: str,
    mode: str = "zero",
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Constructs model-compliant prompts using the tokenizer's chat template.
    Strictly avoids hardcoded special tags (<|user|>, <|im_start|>, etc.).

    Args:
        tokenizer: Hugging Face tokenizer with apply_chat_template support.
        code: The function source code to audit.
        mode: Prompt mode ('zero' for standard zero-shot, or '2_shot'/'few' for few-shot).
        few_shot_examples: Optional list of dicts with 'code' and 'label' ('YES' or 'NO').

    Returns:
        Formatted prompt string ready for tokenization.
    """
    messages = []
    if mode in ("2_shot", "few") and few_shot_examples:
        for ex in few_shot_examples:
            ex_content = FIXED_INSTRUCTION_TEMPLATE.format(code=ex["code"])
            messages.append({"role": "user", "content": ex_content})
            messages.append({"role": "assistant", "content": ex["label"]})

    content = FIXED_INSTRUCTION_TEMPLATE.format(code=code)
    messages.append({"role": "user", "content": content})

    # In case the tokenizer doesn't have a chat template configured (rare for instruct models)
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        text_parts = []
        for m in messages:
            if m["role"] == "user":
                text_parts.append(f"User: {m['content']}")
            elif m["role"] == "assistant":
                text_parts.append(f"Assistant: {m['content']}")
        text_parts.append("Assistant: ")
        return "\n\n".join(text_parts)

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    return prompt


def get_target_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    """
    Extracts and de-duplicates the first non-whitespace token IDs for YES and NO variants.
    Handles space prefixes and BPE/SentencePiece tokenizer peculiarities.
    Strictly removes:
      1. Pure-whitespace tokens (e.g. ' ', ' ', '\n').
      2. Any token ID that appears in BOTH YES and NO sets (e.g. SPIECE space 29871).

    Returns:
        yes_ids: Unique list of token IDs corresponding to YES variants.
        no_ids: Unique list of token IDs corresponding to NO variants.
    """
    def extract_candidate_ids(candidates: List[str]) -> List[int]:
        ids = []
        for cand in candidates:
            enc = tokenizer.encode(cand, add_special_tokens=False)
            if not enc:
                continue
            # Pick first non-whitespace token in encoding (handles prefix space tokens)
            chosen_id = None
            for tid in enc:
                dec = tokenizer.decode([tid])
                if dec.strip():
                    chosen_id = tid
                    break
            # Fallback to enc[0] if all tokens had whitespace decode
            if chosen_id is None and enc:
                chosen_id = enc[0]
            if chosen_id is not None and chosen_id not in ids:
                ids.append(chosen_id)
        return ids

    yes_ids = extract_candidate_ids(YES_CANDIDATES)
    no_ids = extract_candidate_ids(NO_CANDIDATES)

    # 1. Purge any pure-whitespace token IDs
    yes_ids = [tid for tid in yes_ids if tokenizer.decode([tid]).strip()]
    no_ids = [tid for tid in no_ids if tokenizer.decode([tid]).strip()]

    # 2. Purge any token ID present in BOTH yes_ids and no_ids
    overlap = set(yes_ids).intersection(set(no_ids))
    yes_ids = [tid for tid in yes_ids if tid not in overlap]
    no_ids = [tid for tid in no_ids if tid not in overlap]

    if not yes_ids or not no_ids:
        raise ValueError(
            f"Failed to identify target token IDs. Got yes_ids={yes_ids}, no_ids={no_ids}, overlap={overlap}"
        )

    return yes_ids, no_ids


def score_prompt(
    model,
    tokenizer,
    prompt: str,
    yes_ids: Optional[List[int]] = None,
    no_ids: Optional[List[int]] = None,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """
    Scores a prompt by evaluating logits on a single forward pass without generation.

    Computes:
        p_vuln = softmax([logsumexp(logits[YES_ids]), logsumexp(logits[NO_ids])])[0]

    Returns a rich dictionary containing:
        - p_vuln: float (probability the code is vulnerable)
        - p_safe: float (probability the code is safe)
        - pred_50: int (binary prediction at standard 0.5 threshold)
        - top1_id: int (token ID with highest unconstrained logit)
        - top1_token: str (decoded top-1 token)
        - is_in_yes_no: bool (whether the model's top-1 prediction is in YES ∪ NO)
        - n_input_tokens: int (number of prompt tokens)
        - latency_s: float (synchronized inference latency in seconds)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for scoring.")

    if yes_ids is None or no_ids is None:
        yes_ids, no_ids = get_target_token_ids(tokenizer)

    device = next(model.parameters()).device
    raw_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
    )
    if hasattr(raw_inputs, "to"):
        inputs = raw_inputs.to(device)
    else:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in raw_inputs.items()}

    n_input_tokens = inputs["input_ids"].shape[1]
    if n_input_tokens > max_tokens:
        # Gracefully truncate from the left to keep the decision prompt and function suffix
        inputs["input_ids"] = inputs["input_ids"][:, -max_tokens:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -max_tokens:]
        n_input_tokens = max_tokens

    # Synchronize for precise GPU timing
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs)
        # Final token logits: shape [vocab_size]
        logits = outputs.logits[0, -1, :]

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    latency_s = time.perf_counter() - t0

    # Aggregate target token logits using logsumexp for numerical stability
    yes_logits = logits[yes_ids]
    no_logits = logits[no_ids]

    logsumexp_yes = torch.logsumexp(yes_logits, dim=0)
    logsumexp_no = torch.logsumexp(no_logits, dim=0)

    two_way_logits = torch.stack([logsumexp_yes, logsumexp_no])
    probs = torch.softmax(two_way_logits, dim=0)

    p_vuln = float(probs[0].item())
    p_safe = float(probs[1].item())

    # Sanity check: What would the unconstrained greedy next token be?
    top1_id = int(torch.argmax(logits).item())
    top1_token = tokenizer.decode([top1_id]).strip()
    is_in_yes_no = (top1_id in yes_ids) or (top1_id in no_ids)

    return {
        "p_vuln": p_vuln,
        "p_safe": p_safe,
        "pred_50": 1 if p_vuln >= 0.5 else 0,
        "top1_id": top1_id,
        "top1_token": top1_token,
        "is_in_yes_no": is_in_yes_no,
        "n_input_tokens": n_input_tokens,
        "latency_s": latency_s,
    }
