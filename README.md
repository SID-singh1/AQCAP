# Is 4-Bit Enough? Signal-Gated Evaluation of Quantized LLM Vulnerability Detectors

Code, cleaned data splits, configurations and **all per-function predictions** for the paper:

> Siddhant Singh and Gunavathi C. *Is 4-Bit Enough? Signal-Gated Evaluation of Quantized LLM Vulnerability Detectors.* 2026. (arXiv link to be added)

This repository grew out of **AQCAP** (Adaptive Quantization and Context-Aware Pruning), a mixed-precision framework for LLM-based code security auditors. Evaluating AQCAP rigorously turned it into the measurement study below.

## Key findings

| # | Finding |
|---|---|
| 1 | **Gate on FP16 signal first.** On PrimeVul *paired*, Qwen2.5-Coder-7B/3B, Llama-3.1-8B and Phi-3-mini are all at chance (AUROC 0.49–0.52). Quantizing a model without signal gives spurious "gains". Llama-8B's AUROC *rises* from 0.533 to 0.598 at 2 bits, even though its output has collapsed. |
| 2 | **Juliet leaks its labels.** `bad`/`good` identifiers, `FLAW` comments and `OMITBAD` guards inflate Qwen-7B FP16 AUROC from 0.68 to 0.98. Our sanitiser removes them and has a unit test. |
| 3 | **4-bit (and 3-bit) is enough.** On Qwen-7B/cleaned Juliet (FP16 0.624), INT8, NF4, HQQ-4 and HQQ-3 are TOST-equivalent to FP16 within ±0.02 AUROC, at 34–57% of FP16 weight memory. Sensitivity-guided INT8/NF4 mixing adds nothing. |
| 4 | **Collapse is size-dependent.** Qwen-7B stays usable down to 3 bits, while Qwen-3B collapses at 3 bits: its top token is never YES/NO. A collapsed model's AUROC is arbitrary (0.30–0.67), so a **compliance gate** is needed. |
| 5 | **At 2 bits, layer allocation is a lottery.** Keeping 7 of 28 layers at 4 bits spans 0.557–0.620 AUROC at equal memory. No sensitivity metric (S_gen, S_L2, S_task, or our deployment-context upgrade metric S_up) beats a 10-seed random baseline, because layer effects are non-additive. |
| 6 | **Quantization shifts calibration more than ranking**, so thresholds must be re-selected after quantization. |

Full numbers: [`paper/summary.csv`](paper/summary.csv). Paper source: [`paper/`](paper/).

## Repository layout

```
aqcap/            core library
  quant.py          per-layer mixed precision (bitsandbytes INT8/NF4, HQQ 4/3/2-bit)
  scoring.py        frozen zero-shot prompt, single-forward-pass P(YES)/(P(YES)+P(NO))
  data.py           PrimeVul / Juliet loading, Juliet leakage sanitiser
  sensitivity.py    single-layer sensitivity: S_task, S_gen, S_L2
  metrics.py        AUROC, bootstrap CIs, pair metrics
  cap.py            context-aware pruning (comment removal)
scripts/
  run_eval.py                 evaluate one model x dataset x config
  run_sensitivity.py          single-layer sensitivity profiling
  run_upgrade_sensitivity.py  deployment-context (upgrade) sensitivity S_up
  kaggle_eval_launcher.py     stage runner used for all experiments
  summarize_results.py        paired bootstrap, TOST, compliance -> paper/summary.csv
  make_figures.py             paper figures
  signal_gate.py              FP16 signal gate across models/datasets/prompts
configs/          every precision map used in the paper
data/splits/      calibration/test splits (sample IDs; PrimeVul splits include code)
data/juliet_paired.jsonl   exact Juliet 1.3 extraction used in the paper
results/          per-function predictions (*.jsonl) + run metadata + sensitivity profiles
kaggle/           notebooks used to run everything on Kaggle (2x T4)
paper/            LaTeX source, figures, summary table
tests/            unit tests (incl. Juliet leakage audit)
```

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

**Re-derive every table and figure from the released predictions (CPU, ~2 min):**
```bash
python scripts/summarize_results.py --results_dir results --out paper/summary.csv
python scripts/make_figures.py
```

**Re-run experiments (GPU).** We used Kaggle 2x T4. You need a Hugging Face token with access to Llama-3.1. PrimeVul is downloaded automatically from the Hugging Face Hub.
```bash
python scripts/kaggle_eval_launcher.py --stage qwen7b_juliet        # 4/8-bit study
python scripts/kaggle_eval_launcher.py --stage qwen7b_hqq_sweep     # 4/3/2-bit + 2-bit layer selection
python scripts/kaggle_eval_launcher.py --stage qwen7b_hqq_upgrade   # S_up + random seeds
python scripts/kaggle_eval_launcher.py --stage qwen3b_primevul
python scripts/kaggle_eval_launcher.py --stage qwen3b_pvu_hqq
python scripts/kaggle_eval_launcher.py --stage qwen7b_primevul
python scripts/kaggle_eval_launcher.py --stage llama8b_juliet
```
A single evaluation:
```bash
python scripts/run_eval.py --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset juliet --config qwen7b_nf4 --split test --device auto
```

## Data

- **Juliet C/C++ 1.3** (NIST SARD) is public domain. `data/juliet_paired.jsonl` is the exact extraction used. The sanitiser in `aqcap/data.py` is applied at load time.
- **PrimeVul** (Ding et al., ICSE 2025) is fetched from the Hugging Face Hub. The split files contain the sampled functions for exact reproducibility. Please cite the original dataset.

## Citation

```bibtex
@misc{singh2026fourbit,
  title  = {Is 4-Bit Enough? Signal-Gated Evaluation of Quantized LLM Vulnerability Detectors},
  author = {Singh, Siddhant and C, Gunavathi},
  year   = {2026},
  note   = {arXiv preprint (link to be added)}
}
```

## License

Code: MIT (see `LICENSE`). Datasets keep their original licenses.
