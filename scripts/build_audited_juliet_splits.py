import os
import sys
import re
import csv
import json
import random
from collections import defaultdict, Counter

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.audit_full_juliet import (
    preprocess_juliet_function,
    has_valid_function_body,
    FORBIDDEN_JULIET_SUBSTRINGS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MAX_TOKENS = 2048

def rough_token_count(text: str) -> int:
    """Conservative approximation of BPE tokens (~3.5 chars per token for code)."""
    return int(len(text) / 3.2) + len(text.split()) // 2

def build_audited_juliet(
    csv_path: str = os.path.join(DATA_DIR, "jts_c_1_3_test.csv"),
    output_jsonl: str = os.path.join(DATA_DIR, "juliet_paired.jsonl"),
    output_dir: str = SPLITS_DIR,
    n_calib_pairs: int = 100,
    n_test_pairs: int = 500,
    seed: int = 0,
):
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[AUDIT] Reading and preprocessing raw Juliet CSV {csv_path}...")
    valid_pairs = []
    by_cwe = defaultdict(list)
    by_testcase = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            fn = row.get("filename", "")
            m = re.search(r"CWE(\d+)", fn)
            cwe = f"CWE-{m.group(1)}" if m else "Unknown"
            g = row.get("good", "")
            b = row.get("bad", "")
            if not g or not b or "{" not in g or "{" not in b:
                continue

            pg = preprocess_juliet_function(g)
            pb = preprocess_juliet_function(b)

            if not has_valid_function_body(pg) or not has_valid_function_body(pb) or pg == pb:
                continue

            # Token length guard
            if rough_token_count(pg) > MAX_TOKENS or rough_token_count(pb) > MAX_TOKENS:
                continue

            # Base testcase grouping (e.g. 53a, 53b -> 53)
            base_fn = re.sub(r'_[0-9]{2}[a-z]\.c(?:pp)?$', '', fn, flags=re.I)

            pair_id = f"juliet_{i*2}"
            vuln_obj = {
                "idx": i * 2 + 1,
                "pair_id": pair_id,
                "target": 1,
                "func": pb,
                "cwe": cwe,
                "project": "Juliet",
                "filename": fn,
                "base_testcase": base_fn,
            }
            safe_obj = {
                "idx": i * 2 + 2,
                "pair_id": pair_id,
                "target": 0,
                "func": pg,
                "cwe": cwe,
                "project": "Juliet",
                "filename": fn,
                "base_testcase": base_fn,
            }

            pair_record = {
                "pair_id": pair_id,
                "filename": fn,
                "base_testcase": base_fn,
                "cwe": cwe,
                "vuln": vuln_obj,
                "safe": safe_obj,
            }
            valid_pairs.append(pair_record)
            by_cwe[cwe].append(pair_record)
            by_testcase[base_fn].append(pair_record)

    print(f"[AUDIT] Valid token-compliant pairs: {len(valid_pairs)} across {len(by_cwe)} CWEs")

    # Save complete clean preprocessed jsonl
    print(f"[AUDIT] Writing full audited jsonl -> {output_jsonl}...")
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for p in valid_pairs:
            f_out.write(json.dumps(p["vuln"]) + "\n")
            f_out.write(json.dumps(p["safe"]) + "\n")

    # CWE-stratified and testcase-disjoint sampling
    # We must ensure calib and test are completely disjoint by base_testcase
    for cwe in by_cwe:
        rng.shuffle(by_cwe[cwe])

    cwes = sorted(list(by_cwe.keys()))
    rng.shuffle(cwes)

    calib_pairs = []
    used_testcases = set()
    used_pair_ids = set()

    # Sample calib pairs (100 pairs = 200 funcs)
    idx_map = {cwe: 0 for cwe in cwes}
    while len(calib_pairs) < n_calib_pairs:
        added = 0
        for cwe in cwes:
            plist = by_cwe[cwe]
            while idx_map[cwe] < len(plist):
                cand = plist[idx_map[cwe]]
                idx_map[cwe] += 1
                base = cand["base_testcase"]
                if base not in used_testcases and cand["pair_id"] not in used_pair_ids:
                    calib_pairs.append(cand)
                    used_testcases.add(base)
                    used_pair_ids.add(cand["pair_id"])
                    added += 1
                    break
            if len(calib_pairs) >= n_calib_pairs:
                break
        if added == 0:
            break

    # Sample test pairs (500 pairs = 1000 funcs), strictly disjoint from calib
    test_pairs = []
    while len(test_pairs) < n_test_pairs:
        added = 0
        for cwe in cwes:
            plist = by_cwe[cwe]
            while idx_map[cwe] < len(plist):
                cand = plist[idx_map[cwe]]
                idx_map[cwe] += 1
                base = cand["base_testcase"]
                if base not in used_testcases and cand["pair_id"] not in used_pair_ids:
                    test_pairs.append(cand)
                    used_testcases.add(base)
                    used_pair_ids.add(cand["pair_id"])
                    added += 1
                    break
            if len(test_pairs) >= n_test_pairs:
                break
        if added == 0:
            break

    print(f"\n[AUDIT] Sampled Calib Pairs: {len(calib_pairs)} ({len(calib_pairs)*2} funcs)")
    print(f"[AUDIT] Sampled Test Pairs: {len(test_pairs)} ({len(test_pairs)*2} funcs)")

    # Assert strict disjointness
    calib_testcases = {p["base_testcase"] for p in calib_pairs}
    test_testcases = {p["base_testcase"] for p in test_pairs}
    overlap = calib_testcases.intersection(test_testcases)
    assert len(overlap) == 0, f"Error: {len(overlap)} testcases overlap between calib and test!"

    # Verify zero leakage in all sampled functions
    for p in calib_pairs + test_pairs:
        for code in (p["vuln"]["func"], p["safe"]["func"]):
            code_lower = code.lower()
            for sub in FORBIDDEN_JULIET_SUBSTRINGS:
                assert sub not in code_lower, f"Leakage '{sub}' in {p['pair_id']}!"

    # Save splits
    calib_meta = {
        "split": "calib",
        "dataset": "juliet",
        "seed": seed,
        "n_pairs": len(calib_pairs),
        "n_samples": len(calib_pairs) * 2,
        "source_file": os.path.basename(output_jsonl),
        "pairs": [
            {
                "pair_id": p["pair_id"],
                "vuln_idx": p["vuln"]["idx"],
                "safe_idx": p["safe"]["idx"],
                "cwe": p["cwe"],
                "project": "Juliet",
                "filename": p["filename"],
                "base_testcase": p["base_testcase"],
            }
            for p in calib_pairs
        ],
    }

    test_meta = {
        "split": "test",
        "dataset": "juliet",
        "seed": seed,
        "n_pairs": len(test_pairs),
        "n_samples": len(test_pairs) * 2,
        "source_file": os.path.basename(output_jsonl),
        "pairs": [
            {
                "pair_id": p["pair_id"],
                "vuln_idx": p["vuln"]["idx"],
                "safe_idx": p["safe"]["idx"],
                "cwe": p["cwe"],
                "project": "Juliet",
                "filename": p["filename"],
                "base_testcase": p["base_testcase"],
            }
            for p in test_pairs
        ],
    }

    calib_path = os.path.join(output_dir, "juliet_calib_split.json")
    test_path = os.path.join(output_dir, "juliet_test_split.json")

    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_meta, f, indent=2)

    print(f"[AUDIT] Saved {calib_path}")
    print(f"[AUDIT] Saved {test_path}")

    # Report CWE distribution in test
    test_cwes = Counter(p["cwe"] for p in test_pairs)
    print(f"\n[AUDIT] Test Split CWE Diversity: {len(test_cwes)} unique CWEs")
    for cwe, count in test_cwes.most_common(10):
        print(f"  {cwe}: {count} pairs ({count*2} funcs)")

    return calib_pairs, test_pairs

if __name__ == "__main__":
    build_audited_juliet()
