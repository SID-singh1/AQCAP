"""
Greedy (interaction-aware) layer allocation for low-bit mixed precision.

Single-layer sensitivity scores (S_task, S_gen, S_L2, S_up) rank layers independently and
fail at 2 bits because layer effects are non-additive. This script searches over layer
*sets* instead, with greedy forward selection:

    S_0 = {}                              (every layer at --base_prec, e.g. hqq2)
    S_k = S_{k-1} + argmin_i  KL( P_fp16 || P_{S_{k-1} + i} )   on the calibration split
    for k = 1 .. --budget

where layers in S are at --up_prec (e.g. hqq4) and KL is the mean binary KL of P(vuln).
Both quantized versions of every layer are built once and kept on the GPU, so trying a
candidate is a pointer swap rather than a model reload.

Outputs
    {output_csv}                 every candidate evaluation (step, layer, kl, auroc, compliance)
    configs/{prefix}greedy.json  the selected set at --budget (test it with run_eval.py)

Resumable: completed candidate rows in {output_csv} are reused after a restart.

Usage:
    python scripts/run_greedy_allocation.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset juliet --prefix qwen7b_hqq_2bit_ --budget 7 \
        --output_csv results/greedy_qwen7b_hqq2_juliet.csv
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import copy
import gc
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aqcap.data import load_dataset_split
from aqcap.quant import _create_hqq_linear, get_decoder_layers
from aqcap.scoring import build_prompt, get_target_token_ids, score_prompt
from aqcap.sensitivity import compute_binary_kl

NBITS = {"hqq2": 2, "hqq3": 3, "hqq4": 4}
DEVICE = "cuda"


def batched_scores(model, tokenizer, prompts, yes_ids, no_ids, batch_size):
    """P(vuln) and YES/NO top-1 compliance for each prompt, left-padded batches."""
    device = next(model.parameters()).device
    p_out, comp_out = [], []
    for b in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[b:b + batch_size], return_tensors="pt", padding=True).to(device)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = model(**enc, position_ids=pos).logits[:, -1, :].float()
        two = torch.stack([torch.logsumexp(logits[:, yes_ids], -1), torch.logsumexp(logits[:, no_ids], -1)], -1)
        p_out.extend(torch.softmax(two, -1)[:, 0].tolist())
        top1 = logits.argmax(-1).tolist()
        comp_out.extend([(t in yes_ids) or (t in no_ids) for t in top1])
    return np.array(p_out), np.array(comp_out)


def mean_kl(ref, q):
    return float(np.mean([compute_binary_kl(a, b) for a, b in zip(ref, q)]))


def build_dual_model(model_id, base_prec, up_prec):
    """FP16 model whose decoder linears can be switched between two HQQ versions on the GPU."""
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu")
    layers = get_decoder_layers(model)
    swaps = []  # per layer: list of (parent, attr, base_module, up_module)
    t0 = time.time()
    for li, block in enumerate(layers):
        entries = []
        for parent in list(block.modules()):
            for attr, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    q = {}
                    for prec in (base_prec, up_prec):
                        src = copy.deepcopy(child).to(DEVICE)
                        q[prec] = _create_hqq_linear(src, nbits=NBITS[prec], group_size=64, del_orig=True,
                                                     compute_dtype=torch.float16, device=DEVICE)
                        del src
                    setattr(parent, attr, q[base_prec])
                    entries.append((parent, attr, q[base_prec], q[up_prec]))
                    del child
        swaps.append(entries)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[GREEDY] layer {li} quantized to {base_prec}+{up_prec} ({time.time() - t0:.0f}s)", flush=True)
    for _, p in model.named_parameters():
        p.data = p.data.to(DEVICE)
    for _, b in model.named_buffers():
        b.data = b.data.to(DEVICE)
    model.eval()
    return model, swaps


def set_upgraded(swaps, upgraded):
    for li, entries in enumerate(swaps):
        for parent, attr, base_m, up_m in entries:
            setattr(parent, attr, up_m if li in upgraded else base_m)


def main():
    ap = argparse.ArgumentParser(description="Greedy interaction-aware layer allocation")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="juliet", choices=["juliet", "primevul_unpaired", "primevul_paired"])
    ap.add_argument("--base_prec", default="hqq2", choices=list(NBITS))
    ap.add_argument("--up_prec", default="hqq4", choices=list(NBITS))
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--configs_dir", default="configs")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_samples", type=int, default=None, help="Smoke test: use only the first N calib samples")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes_ids, no_ids = get_target_token_ids(tok)
    samples = load_dataset_split(dataset_name=args.dataset, split="calib", max_tokens=args.max_tokens)["samples"]
    if args.max_samples:
        samples = samples[: args.max_samples]
    prompts = [build_prompt(tok, s["func"], mode="zero") for s in samples]
    y = np.array([s["target"] for s in samples])
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    ref_path = args.output_csv.replace(".csv", "_fp16.npy")

    # 1. FP16 reference (batched, same scorer as the search)
    if os.path.isfile(ref_path):
        p_fp16 = np.load(ref_path)
    else:
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto").eval()
        p_fp16, _ = batched_scores(m, tok, prompts, yes_ids, no_ids, args.batch_size)
        np.save(ref_path, p_fp16)
        del m
        gc.collect()
        torch.cuda.empty_cache()
    print(f"[GREEDY] FP16 calib AUROC {roc_auc_score(y, p_fp16):.4f}")

    # 2. Dual-precision model
    model, swaps = build_dual_model(args.model, args.base_prec, args.up_prec)
    n_layers = len(swaps)

    # Batched scorer must agree with the unbatched scorer used for test evaluation
    set_upgraded(swaps, set())
    pb, _ = batched_scores(model, tok, prompts[:16], yes_ids, no_ids, args.batch_size)
    pu = np.array([score_prompt(model, tok, p, yes_ids=yes_ids, no_ids=no_ids)["p_vuln"] for p in prompts[:16]])
    gap = float(np.abs(pb - pu).max())
    print(f"[GREEDY] batched vs unbatched max |dP| = {gap:.4f}")
    if gap > 0.02:
        print("[GREEDY] gap too large; falling back to batch_size=1")
        args.batch_size = 1

    def evaluate(upgraded):
        set_upgraded(swaps, upgraded)
        p, comp = batched_scores(model, tok, prompts, yes_ids, no_ids, args.batch_size)
        return mean_kl(p_fp16, p), roc_auc_score(y, p), float(comp.mean())

    base_kl, base_auc, base_comp = evaluate(set())
    print(f"[GREEDY] uniform {args.base_prec}: KL {base_kl:.5f} AUROC {base_auc:.4f} compliance {base_comp:.3f}")

    # 3. Greedy forward selection (resumable)
    rows = pd.read_csv(args.output_csv).to_dict("records") if os.path.isfile(args.output_csv) else []
    done = {(int(r["step"]), int(r["layer"])): r for r in rows}
    selected = []
    for step in range(1, args.budget + 1):
        t0 = time.time()
        for li in range(n_layers):
            if li in selected or (step, li) in done:
                continue
            kl, auc, comp = evaluate(set(selected) | {li})
            r = {"step": step, "layer": li, "kl": kl, "auroc": auc, "compliance": comp,
                 "selected_before": " ".join(map(str, selected))}
            rows.append(r)
            done[(step, li)] = r
            pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        cand = [done[(step, li)] for li in range(n_layers) if li not in selected]
        best = min(cand, key=lambda r: r["kl"])
        selected.append(int(best["layer"]))
        print(f"[GREEDY] step {step}: + layer {int(best['layer'])} -> KL {best['kl']:.5f} "
              f"AUROC {best['auroc']:.4f} compliance {best['compliance']:.3f} ({time.time() - t0:.0f}s) "
              f"set={sorted(selected)}", flush=True)

    name = f"{args.prefix}greedy"
    cfg = {
        "name": name,
        "description": f"Greedy interaction-aware allocation: {args.up_prec} on {args.budget} layers chosen by "
                       f"forward selection on calibration KL to FP16, rest {args.base_prec}",
        "precision_map": {str(i): (args.up_prec if i in selected else args.base_prec) for i in range(n_layers)},
        "sensitive_layers": sorted(selected),
        "selection_order": selected,
    }
    with open(os.path.join(args.configs_dir, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[GREEDY] Wrote configs/{name}.json: {sorted(selected)}")


if __name__ == "__main__":
    main()
