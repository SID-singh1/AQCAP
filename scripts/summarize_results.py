"""
Summarise every evaluation run in a results directory into one table.

For each (model, dataset) group, every config is compared against that group's
FP16 run on the identical samples:
  - AUROC with a cluster bootstrap 95% CI (clusters = Juliet pair_id, else sample)
  - paired bootstrap dAUROC vs FP16 (same resample indices) + TOST at +/-0.02
  - YES/NO top-1 compliance, mean p_vuln, weight memory, latency
  - recall-side flips at a threshold chosen on FP16 to give 90% recall

Usage:
    python scripts/summarize_results.py --results_dir aqcap_results/results --out paper/summary.csv
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOT = 2000
MARGIN = 0.02


def load_run(path):
    meta, rows = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "metadata":
                meta = r
            elif r.get("type") == "sample":
                rows.append(r)
    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    return meta, df


def cluster_resamples(clusters, rng):
    """Bootstrap index arrays that resample whole clusters (pairs) with replacement."""
    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = list(groups)
    out = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        out.append(np.concatenate([groups[keys[k]] for k in pick]))
    return out


def boot_auc(y, s, resamples):
    vals = []
    for ix in resamples:
        yy = y[ix]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, s[ix]))
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="paper/summary.csv")
    args = ap.parse_args()

    runs = {}
    for p in sorted(glob.glob(os.path.join(args.results_dir, "*.jsonl"))):
        meta, df = load_run(p)
        if meta is None or df.empty:
            continue
        runs[meta["run_id"]] = (meta, df)

    groups = defaultdict(dict)
    for rid, (meta, df) in runs.items():
        tag = rid.split("_")[0]
        groups[(tag, meta["dataset"])][rid] = (meta, df)

    records = []
    rng = np.random.default_rng(42)
    for (tag, dataset), members in sorted(groups.items()):
        ref_id = next((r for r in members if r.endswith("_fp16_test_cap_off")), None)
        if ref_id is None:
            continue
        _, ref = members[ref_id]
        y = ref["target"].to_numpy()
        clusters = ref["pair_id"].to_numpy() if "pair_id" in ref and ref["pair_id"].notna().all() else np.arange(len(ref))
        resamples = cluster_resamples(clusters, rng)
        ref_s = ref["p_vuln"].to_numpy()
        ref_boot = boot_auc(y, ref_s, resamples)

        # FP16 threshold giving >= 90% recall (stand-in for a calib-chosen operating point)
        pos = np.sort(ref_s[y == 1])
        thr = pos[int(np.floor(0.10 * len(pos)))]
        ref_pred = ref_s >= thr

        for rid, (meta, df) in sorted(members.items()):
            if not np.array_equal(df["idx"].to_numpy(), ref["idx"].to_numpy()):
                print(f"[WARN] {rid}: sample set differs from FP16; skipped")
                continue
            s = df["p_vuln"].to_numpy()
            auc = roc_auc_score(y, s)
            b = boot_auc(y, s, resamples)
            d = b - ref_boot if len(b) == len(ref_boot) else np.array([np.nan])
            d_lo, d_hi = np.nanpercentile(d, [2.5, 97.5]) if rid != ref_id else (0.0, 0.0)
            # TOST via 90% CI inside +/- margin
            t_lo, t_hi = np.nanpercentile(d, [5, 95]) if rid != ref_id else (0.0, 0.0)
            pred = s >= thr
            vul = y == 1
            records.append({
                "model": tag,
                "dataset": dataset,
                "config": meta["config_name"] + ("+CAP" if meta.get("cap") == "on" else ""),
                "n": len(df),
                "weights_mb": round(meta.get("weights_mb", np.nan)),
                "auroc": round(auc, 4),
                "auroc_lo": round(np.percentile(b, 2.5), 4),
                "auroc_hi": round(np.percentile(b, 97.5), 4),
                "d_auroc": round(auc - roc_auc_score(y, ref_s), 4),
                "d_lo": round(d_lo, 4),
                "d_hi": round(d_hi, 4),
                "tost_equiv_0.02": bool(t_lo > -MARGIN and t_hi < MARGIN),
                "recall@fp16thr": round(pred[vul].mean(), 3),
                "fpr@fp16thr": round(pred[~vul].mean(), 3),
                "missed_vs_fp16": int((ref_pred & ~pred & vul).sum()),
                "gained_vs_fp16": int((~ref_pred & pred & vul).sum()),
                "yesno_top1": round(df["is_in_yes_no"].mean(), 3),
                "mean_p_vuln": round(s.mean(), 3),
                "latency_s": round(df["latency_s"].median(), 3),
            })

    out = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(out.drop(columns=["n"]).to_string(index=False))


if __name__ == "__main__":
    main()
