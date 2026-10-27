"""
AQCAP Data Ingestion & PrimeVul Paired Dataset Splitter.

Handles:
1. Automated downloading / local discovery of PrimeVul paired splits.
2. Pair validation (exactly one target=1 and one target=0 per commit_id).
3. Max-token filtering (dropping entire pairs if either member exceeds MAX_TOKENS).
4. Deterministic sampling and saving of calib (100 pairs) and test (400 pairs) splits.
"""

import os
import sys
import re
import csv
import json
import random
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

# Primary URLs for PrimeVul paired dataset on Hugging Face
PRIMEVUL_HF_BASE = "https://huggingface.co/datasets/colin/PrimeVul/resolve/main"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_SPLITS_DIR = os.path.join(DEFAULT_DATA_DIR, "splits")
MAX_TOKENS_LIMIT = 2048


def download_file(url: str, dest_path: str) -> None:
    """Download a file with progress reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"
    print(f"[DATA] Downloading {url} -> {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AQCAP-Research)"})
    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_f:
        chunk_size = 1024 * 1024  # 1MB chunks
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = (downloaded / total) * 100
                print(f"  {downloaded / (1024**2):.1f} MB / {total / (1024**2):.1f} MB ({pct:.1f}%)\r", end="", flush=True)
        print()
    os.replace(temp_path, dest_path)
    print(f"[DATA] Saved {dest_path}")


def ensure_primevul_data(data_dir: str = DEFAULT_DATA_DIR) -> Tuple[str, str]:
    """
    Ensure primevul_valid_paired.jsonl and primevul_test_paired.jsonl exist.
    Checks environment variable PRIMEVUL_DIR, common Kaggle paths, and local dir.
    Downloads from Hugging Face if not found.
    """
    valid_name = "primevul_valid_paired.jsonl"
    test_name = "primevul_test_paired.jsonl"

    candidate_dirs = [
        os.environ.get("PRIMEVUL_DIR", ""),
        data_dir,
        "/kaggle/input/primevul",
        "/kaggle/input/primevul-dataset",
        "/kaggle/input/datasets/sidsingh123456/primevul",
    ]

    found_valid = None
    found_test = None

    for cdir in candidate_dirs:
        if not cdir or not os.path.isdir(cdir):
            continue
        v_path = os.path.join(cdir, valid_name)
        t_path = os.path.join(cdir, test_name)
        if os.path.isfile(v_path) and os.path.isfile(t_path):
            found_valid = v_path
            found_test = t_path
            break

    if found_valid and found_test:
        print(f"[DATA] Found PrimeVul dataset in {os.path.dirname(found_valid)}")
        return found_valid, found_test

    # Download into data_dir
    os.makedirs(data_dir, exist_ok=True)
    valid_path = os.path.join(data_dir, valid_name)
    test_path = os.path.join(data_dir, test_name)

    if not os.path.isfile(valid_path):
        url = f"{PRIMEVUL_HF_BASE}/{valid_name}"
        download_file(url, valid_path)

    if not os.path.isfile(test_path):
        url = f"{PRIMEVUL_HF_BASE}/{test_name}"
        download_file(url, test_path)

    return valid_path, test_path


def normalize_cwe(cwe_val: Any) -> str:
    """Normalize CWE value to a single standard string (e.g., 'CWE-119')."""
    if isinstance(cwe_val, list):
        if len(cwe_val) == 0:
            return "Unknown"
        return str(cwe_val[0]).strip()
    if cwe_val is None or (isinstance(cwe_val, float) and str(cwe_val) == "nan"):
        return "Unknown"
    val = str(cwe_val).strip()
    return val if val else "Unknown"


def load_primevul_raw(filepath: str) -> List[Dict[str, Any]]:
    """Loads raw PrimeVul JSONL file and normalizes fields."""
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Schema verification: func, target, cwe, project, commit_id, idx
            items.append({
                "idx": int(obj["idx"]),
                "target": int(obj["target"]),
                "func": str(obj["func"]),
                "cwe": normalize_cwe(obj.get("cwe")),
                "project": str(obj.get("project", "Unknown")),
                "commit_id": str(obj.get("commit_id", "")),
            })
    return items


def count_tokens(text: str, tokenizer=None) -> int:
    """Measure token count using provided tokenizer or fallback heuristic."""
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    # Heuristic fallback (approx 3.5 chars per token for C/C++ code)
    return int(len(text) / 3.5)


def pair_and_filter(
    items: List[Dict[str, Any]],
    tokenizer=None,
    max_tokens: int = MAX_TOKENS_LIMIT
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Pairs items by commit_id.
    Enforces strict 1-to-1 pairing (exactly one target=1 and one target=0).
    Drops the whole pair if either side exceeds max_tokens.

    Returns:
        paired_records: List of pair dicts
        stats: dictionary of filtering statistics
    """
    commits = defaultdict(list)
    for it in items:
        commits[it["commit_id"]].append(it)

    stats = {
        "total_commits": len(commits),
        "valid_pairs": 0,
        "invalid_multi_or_single": 0,
        "dropped_over_max_tokens": 0,
    }

    pairs = []
    for commit_id, group in commits.items():
        targets = [it["target"] for it in group]
        # Check strict 1-to-1 pairing
        if len(group) != 2 or sorted(targets) != [0, 1]:
            stats["invalid_multi_or_single"] += 1
            continue

        vuln_item = group[0] if group[0]["target"] == 1 else group[1]
        safe_item = group[0] if group[0]["target"] == 0 else group[1]

        # Token length check
        len_vuln = count_tokens(vuln_item["func"], tokenizer)
        len_safe = count_tokens(safe_item["func"], tokenizer)

        if len_vuln > max_tokens or len_safe > max_tokens:
            stats["dropped_over_max_tokens"] += 1
            continue

        pairs.append({
            "pair_id": commit_id,
            "project": vuln_item["project"],
            "cwe": vuln_item["cwe"],
            "vuln": vuln_item,
            "safe": safe_item,
        })
        stats["valid_pairs"] += 1

    return pairs, stats


def generate_splits(
    valid_file: str,
    test_file: str,
    tokenizer=None,
    output_dir: str = DEFAULT_SPLITS_DIR,
    seed: int = 0,
    n_calib: int = 100,
    n_test: int = 400
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates deterministic calib (100 pairs) and test (400 pairs) splits.
    Saves chosen idx lists to data/splits/*.json as required by ROADMAP.md.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    print(f"[DATA] Processing valid set from {valid_file}...")
    valid_items = load_primevul_raw(valid_file)
    valid_pairs, valid_stats = pair_and_filter(valid_items, tokenizer, MAX_TOKENS_LIMIT)
    print(f"  Valid pairs available: {len(valid_pairs)} (dropped >{MAX_TOKENS_LIMIT} tok: {valid_stats['dropped_over_max_tokens']})")

    print(f"[DATA] Processing test set from {test_file}...")
    test_items = load_primevul_raw(test_file)
    test_pairs, test_stats = pair_and_filter(test_items, tokenizer, MAX_TOKENS_LIMIT)
    print(f"  Test pairs available: {len(test_pairs)} (dropped >{MAX_TOKENS_LIMIT} tok: {test_stats['dropped_over_max_tokens']})")

    # Sample calib pairs
    rng.shuffle(valid_pairs)
    calib_pairs = valid_pairs[:min(n_calib, len(valid_pairs))]

    # Sample test pairs
    rng.shuffle(test_pairs)
    selected_test_pairs = test_pairs[:min(n_test, len(test_pairs))]

    # Build split index structures for commitment
    calib_meta = {
        "split": "calib",
        "seed": seed,
        "n_pairs": len(calib_pairs),
        "source_file": os.path.basename(valid_file),
        "pairs": [
            {
                "pair_id": p["pair_id"],
                "vuln_idx": p["vuln"]["idx"],
                "safe_idx": p["safe"]["idx"],
                "project": p["project"],
                "cwe": p["cwe"],
            }
            for p in calib_pairs
        ]
    }

    test_meta = {
        "split": "test",
        "seed": seed,
        "n_pairs": len(selected_test_pairs),
        "source_file": os.path.basename(test_file),
        "pairs": [
            {
                "pair_id": p["pair_id"],
                "vuln_idx": p["vuln"]["idx"],
                "safe_idx": p["safe"]["idx"],
                "project": p["project"],
                "cwe": p["cwe"],
            }
            for p in selected_test_pairs
        ]
    }

    calib_json = os.path.join(output_dir, "calib_split.json")
    test_json = os.path.join(output_dir, "test_split.json")

    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2)
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(test_meta, f, indent=2)

    print(f"[DATA] Saved calib split ({len(calib_pairs)} pairs) -> {calib_json}")
    print(f"[DATA] Saved test split ({len(selected_test_pairs)} pairs) -> {test_json}")

    return calib_pairs, selected_test_pairs


def get_flattened_split_samples(pairs: List[Dict[str, Any]], shuffle_samples: bool = False, seed: int = 0) -> List[Dict[str, Any]]:
    """
    Flattens a list of pairs into an evaluation sample list.
    Each sample contains:
      - idx: int
      - pair_id: str (commit_id)
      - target: int (1 = vuln, 0 = safe)
      - func: str
      - cwe: str
      - project: str
      - is_vuln: bool
    """
    samples = []
    for p in pairs:
        # Vuln sample
        samples.append({
            "idx": p["vuln"]["idx"],
            "pair_id": p["pair_id"],
            "target": 1,
            "func": p["vuln"]["func"],
            "cwe": p["cwe"],
            "project": p["project"],
            "is_vuln": True,
        })
        # Safe sample
        samples.append({
            "idx": p["safe"]["idx"],
            "pair_id": p["pair_id"],
            "target": 0,
            "func": p["safe"]["func"],
            "cwe": p["cwe"],
            "project": p["project"],
            "is_vuln": False,
        })

    if shuffle_samples:
        rng = random.Random(seed)
        rng.shuffle(samples)

    return samples


# ==============================================================================
# PRIMEVUL UNPAIRED LOADERS & SPLIT GENERATION
# ==============================================================================

def ensure_primevul_unpaired_data(data_dir: str = DEFAULT_DATA_DIR) -> Tuple[str, str]:
    """
    Ensure primevul_valid.jsonl and primevul_test.jsonl exist.
    Checks environment variable PRIMEVUL_DIR, common Kaggle paths, and local dir.
    Downloads from Hugging Face if not found.
    """
    valid_name = "primevul_valid.jsonl"
    test_name = "primevul_test.jsonl"

    candidate_dirs = [
        os.environ.get("PRIMEVUL_DIR", ""),
        data_dir,
        "/kaggle/input/primevul",
        "/kaggle/input/primevul-dataset",
        "/kaggle/input/datasets/sidsingh123456/primevul",
    ]

    found_valid = None
    found_test = None

    for cdir in candidate_dirs:
        if not cdir or not os.path.isdir(cdir):
            continue
        v_path = os.path.join(cdir, valid_name)
        t_path = os.path.join(cdir, test_name)
        if os.path.isfile(v_path) and os.path.isfile(t_path):
            found_valid = v_path
            found_test = t_path
            break

    if found_valid and found_test:
        return found_valid, found_test

    os.makedirs(data_dir, exist_ok=True)
    valid_path = os.path.join(data_dir, valid_name)
    test_path = os.path.join(data_dir, test_name)

    if not os.path.isfile(valid_path):
        url = f"{PRIMEVUL_HF_BASE}/{valid_name}"
        download_file(url, valid_path)

    if not os.path.isfile(test_path):
        url = f"{PRIMEVUL_HF_BASE}/{test_name}"
        download_file(url, test_path)

    return valid_path, test_path


def generate_primevul_unpaired_splits(
    valid_file: Optional[str] = None,
    test_file: Optional[str] = None,
    output_dir: str = DEFAULT_SPLITS_DIR,
    seed: int = 0,
    n_calib_per_class: int = 100,
    n_test_per_class: int = 500,
    tokenizer=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates balanced unpaired calib (100 vuln, 100 safe) and test (500 vuln, 500 safe = 1000 funcs) splits.
    Saves to data/splits/primevul_unpaired_{calib,test}_split.json.
    """
    if valid_file is None or test_file is None:
        valid_file, test_file = ensure_primevul_unpaired_data()

    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    def extract_filtered_classes(filepath: str):
        vuln, safe = [], []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                func_text = str(obj["func"])
                if count_tokens(func_text, tokenizer) > MAX_TOKENS_LIMIT:
                    continue
                item = {
                    "idx": int(obj["idx"]),
                    "target": int(obj["target"]),
                    "func": func_text,
                    "cwe": normalize_cwe(obj.get("cwe")),
                    "project": str(obj.get("project", "Unknown")),
                }
                if item["target"] == 1:
                    vuln.append(item)
                elif item["target"] == 0:
                    safe.append(item)
        return vuln, safe

    print(f"[DATA] Processing PrimeVul unpaired valid from {valid_file}...")
    val_vuln, val_safe = extract_filtered_classes(valid_file)
    rng.shuffle(val_vuln)
    rng.shuffle(val_safe)
    calib_samples = val_vuln[:n_calib_per_class] + val_safe[:n_calib_per_class]
    rng.shuffle(calib_samples)

    print(f"[DATA] Processing PrimeVul unpaired test from {test_file}...")
    tst_vuln, tst_safe = extract_filtered_classes(test_file)
    rng.shuffle(tst_vuln)
    rng.shuffle(tst_safe)
    test_samples = tst_vuln[:n_test_per_class] + tst_safe[:n_test_per_class]
    rng.shuffle(test_samples)

    calib_meta = {
        "split": "calib",
        "dataset": "primevul_unpaired",
        "seed": seed,
        "n_samples": len(calib_samples),
        "source_file": os.path.basename(valid_file),
        "samples": calib_samples,
    }
    test_meta = {
        "split": "test",
        "dataset": "primevul_unpaired",
        "seed": seed,
        "n_samples": len(test_samples),
        "source_file": os.path.basename(test_file),
        "samples": test_samples,
    }

    calib_json = os.path.join(output_dir, "primevul_unpaired_calib_split.json")
    test_json = os.path.join(output_dir, "primevul_unpaired_test_split.json")

    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2)
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(test_meta, f, indent=2)

    print(f"[DATA] Saved PrimeVul unpaired calib ({len(calib_samples)} samples) -> {calib_json}")
    print(f"[DATA] Saved PrimeVul unpaired test ({len(test_samples)} samples) -> {test_json}")

    return calib_samples, test_samples


# ==============================================================================
# JULIET C/C++ (SARD) LOADERS & SPLIT GENERATION (LEAKAGE AUDITED)
# ==============================================================================

JULIET_HF_CSV = "https://huggingface.co/datasets/LorenzH/juliet_test_suite_c_1_3/resolve/main/jts_c_1_3_test.csv"
FORBIDDEN_JULIET_SUBSTRINGS = ["bad", "good", "g2b", "b2g", "cwe", "omitbad", "omitgood"]


def strip_c_comments(code: str) -> str:
    """Removes block /* ... */ and line // ... comments while strictly preserving string literals."""
    import re
    pattern = re.compile(
        r'(?P<raw_string>R"(?P<delim>[a-zA-Z0-9_]*)\([\s\S]*?\)(?P=delim)")|'
        r'(?P<string>"(?:\\.|[^"\\])*")|'
        r"(?P<char>'(?:\\.|[^'\\])*')|"
        r'(?P<bcomment>/\*[\s\S]*?\*/)|'
        r'(?P<lcomment>//[^\r\n]*)'
    )
    def replacer(m):
        if m.group('raw_string'):
            return m.group('raw_string')
        if m.group('string'):
            return m.group('string')
        if m.group('char'):
            return m.group('char')
        if m.group('bcomment'):
            return ' '
        if m.group('lcomment'):
            return ''
        return m.group(0)
    return pattern.sub(replacer, code)


def sanitize_string_literals(code: str) -> str:
    """Sanitizes forbidden substrings inside string literals and char literals."""
    import re
    str_pattern = re.compile(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')')
    def sanitize_str(m):
        s = m.group(1)
        s_lower = s.lower()
        if not any(sub in s_lower for sub in FORBIDDEN_JULIET_SUBSTRINGS):
            return s
        res = s
        res = re.sub(r'omitbad', 'macro_a', res, flags=re.I)
        res = re.sub(r'omitgood', 'macro_b', res, flags=re.I)
        res = re.sub(r'good', 'data_g', res, flags=re.I)
        res = re.sub(r'bad', 'data_b', res, flags=re.I)
        res = re.sub(r'g2b', 'flow_a', res, flags=re.I)
        res = re.sub(r'b2g', 'flow_b', res, flags=re.I)
        res = re.sub(r'cwe', 'spec', res, flags=re.I)
        return res
    return str_pattern.sub(sanitize_str, code)


def preprocess_juliet_function(raw_code: str) -> str:
    """
    Preprocesses Juliet source code to eliminate all shortcut cues and label leakage:
    1. Removes all OMITBAD / OMITGOOD macro lines and surrounding comments.
    2. Strips all comments (block and line).
    3. Sanitizes string/char literals.
    4. Renames identifiers containing bad/good/G2B/B2G/CWE/OMITBAD/OMITGOOD to neutral names (func_N, var_N).
    5. Normalizes whitespace and asserts zero leakage.
    """
    import re
    # Step 1: Remove lines with OMITBAD / OMITGOOD preprocessor directives
    lines = []
    for line in raw_code.splitlines():
        if re.search(r'OMIT(BAD|GOOD)', line, flags=re.I):
            continue
        lines.append(line)
    code = "\n".join(lines)

    # Step 2: Strip all comments
    code = strip_c_comments(code)

    # Step 3: Sanitize string literals
    code = sanitize_string_literals(code)

    # Step 4: Identifier renaming
    id_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    tokens = set(id_pattern.findall(code))

    leak_pattern = re.compile(r'(bad|good|g2b|b2g|cwe|omitbad|omitgood)', re.IGNORECASE)
    leaking_ids = sorted([t for t in tokens if leak_pattern.search(t)], key=lambda x: -len(x))

    func_counter = 1
    var_counter = 1
    rename_map = {}

    for ident in leaking_ids:
        if re.search(rf'\b{re.escape(ident)}\s*\(', code):
            neutral_name = f"func_{func_counter}"
            func_counter += 1
        else:
            neutral_name = f"var_{var_counter}"
            var_counter += 1
        rename_map[ident] = neutral_name

    for ident, neutral_name in rename_map.items():
        code = re.sub(rf'\b{re.escape(ident)}\b', neutral_name, code)

    # Step 5: Whitespace cleanup
    code = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n{3,}', '\n\n', code)
    code = code.strip()

    # Step 6: Hard assert that zero forbidden substrings survive
    code_lower = code.lower()
    for sub in FORBIDDEN_JULIET_SUBSTRINGS:
        if sub in code_lower:
            idx = code_lower.find(sub)
            start = max(0, idx - 40)
            end = min(len(code), idx + 40)
            raise AssertionError(f"Leakage check failed! Substring '{sub}' found: ...{code[start:end]}...")

    return code


_FUNC_DEF_RE = re.compile(
    r'(?:[a-zA-Z_][a-zA-Z0-9_*]*\s+)+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^;)]*\)\s*\{',
    re.MULTILINE
)


def has_valid_function_body(code: str) -> bool:
    """Verifies that the code contains at least one genuine function definition with a body."""
    if not _FUNC_DEF_RE.search(code):
        return False
    if '{' not in code or '}' not in code:
        return False
    body = code[code.find('{'):code.rfind('}')+1]
    return len(body.strip()) >= 15


def ensure_juliet_data(data_dir: str = DEFAULT_DATA_DIR, force_rebuild: bool = False) -> str:
    """
    Ensures juliet_paired.jsonl exists and is fully audited and preprocessed.
    Checks candidate dirs, downloads jts_c_1_3_test.csv if needed, and builds audited jsonl.
    """
    jsonl_name = "juliet_paired.jsonl"
    candidate_dirs = [
        data_dir,
        "/kaggle/input/juliet",
        "/kaggle/input/juliet-c",
        "/kaggle/input/juliet_test_suite_c_1_3",
    ]
    if not force_rebuild:
        for cdir in candidate_dirs:
            p = os.path.join(cdir, jsonl_name)
            if os.path.isfile(p):
                return p

    jsonl_path = os.path.join(data_dir, jsonl_name)
    csv_path = os.path.join(data_dir, "jts_c_1_3_test.csv")
    if not os.path.isfile(csv_path):
        download_file(JULIET_HF_CSV, csv_path)

    print(f"[DATA] Converting & Auditing Juliet CSV {csv_path} to {jsonl_path}...")
    import re
    import csv

    valid_count = 0
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(jsonl_path, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for i, row in enumerate(reader):
            fn = row.get("filename", "")
            m = re.search(r"CWE(\d+)", fn)
            cwe = f"CWE-{m.group(1)}" if m else "Unknown"
            good_code = row.get("good", "")
            bad_code = row.get("bad", "")
            if not good_code or not bad_code or "{" not in good_code or "{" not in bad_code:
                continue

            pg = preprocess_juliet_function(good_code)
            pb = preprocess_juliet_function(bad_code)

            if not has_valid_function_body(pg) or not has_valid_function_body(pb) or pg == pb:
                continue

            base_fn = re.sub(r'_[0-9]{2}[a-z]\.c(?:pp)?$', '', fn, flags=re.I)

            idx_base = i * 2
            pair_id = f"juliet_{idx_base}"
            vuln_obj = {
                "idx": idx_base + 1,
                "pair_id": pair_id,
                "target": 1,
                "func": pb,
                "cwe": cwe,
                "project": "Juliet",
                "filename": fn,
                "base_testcase": base_fn,
            }
            safe_obj = {
                "idx": idx_base + 2,
                "pair_id": pair_id,
                "target": 0,
                "func": pg,
                "cwe": cwe,
                "project": "Juliet",
                "filename": fn,
                "base_testcase": base_fn,
            }
            f_out.write(json.dumps(vuln_obj) + "\n")
            f_out.write(json.dumps(safe_obj) + "\n")
            valid_count += 1

    print(f"[DATA] Saved {valid_count} audited, non-leaking pairs to {jsonl_path}")
    return jsonl_path


def generate_juliet_splits(
    jsonl_file: Optional[str] = None,
    output_dir: str = DEFAULT_SPLITS_DIR,
    seed: int = 0,
    n_calib_pairs: int = 100,
    n_test_pairs: int = 500,
    tokenizer=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates balanced Juliet calib (~200 funcs = 100 pairs) and test (>=1000 funcs = 500 pairs) splits,
    stratified by CWE and strictly disjoint by testcase file and base testcase.
    Saves to data/splits/juliet_{calib,test}_split.json.
    """
    if jsonl_file is None:
        jsonl_file = ensure_juliet_data()

    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    items = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # Group into pairs
    pairs_by_id = defaultdict(dict)
    for it in items:
        pid = it["pair_id"]
        if it["target"] == 1:
            pairs_by_id[pid]["vuln"] = it
        else:
            pairs_by_id[pid]["safe"] = it

    valid_pairs = []
    by_cwe = defaultdict(list)
    for pid, p in pairs_by_id.items():
        if "vuln" in p and "safe" in p:
            if count_tokens(p["vuln"]["func"], tokenizer) <= MAX_TOKENS_LIMIT and \
               count_tokens(p["safe"]["func"], tokenizer) <= MAX_TOKENS_LIMIT:
                base_tc = p["vuln"].get("base_testcase", p["vuln"].get("filename", pid))
                pair_rec = {
                    "pair_id": pid,
                    "cwe": p["vuln"]["cwe"],
                    "project": "Juliet",
                    "filename": p["vuln"].get("filename", ""),
                    "base_testcase": base_tc,
                    "vuln": p["vuln"],
                    "safe": p["safe"],
                }
                valid_pairs.append(pair_rec)
                by_cwe[p["vuln"]["cwe"]].append(pair_rec)

    for cwe in by_cwe:
        rng.shuffle(by_cwe[cwe])

    cwes = sorted(list(by_cwe.keys()))
    rng.shuffle(cwes)

    calib_pairs = []
    used_testcases = set()
    used_pair_ids = set()

    # 1. Stratified Calib Sampling (100 pairs = 200 funcs)
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

    # 2. Stratified Test Sampling (500 pairs = 1000 funcs), strictly disjoint from calib
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

    # Verify disjointness
    calib_tc = {p["base_testcase"] for p in calib_pairs}
    test_tc = {p["base_testcase"] for p in test_pairs}
    overlap = calib_tc.intersection(test_tc)
    if overlap:
        raise RuntimeError(f"Data leak: {len(overlap)} testcases overlap between calib and test!")

    calib_meta = {
        "split": "calib",
        "dataset": "juliet",
        "seed": seed,
        "n_pairs": len(calib_pairs),
        "n_samples": len(calib_pairs) * 2,
        "source_file": os.path.basename(jsonl_file),
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
        "source_file": os.path.basename(jsonl_file),
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

    calib_json = os.path.join(output_dir, "juliet_calib_split.json")
    test_json = os.path.join(output_dir, "juliet_test_split.json")

    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2)
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(test_meta, f, indent=2)

    print(f"[DATA] Saved Juliet calib split ({len(calib_pairs)} pairs across {len({p['cwe'] for p in calib_pairs})} CWEs) -> {calib_json}")
    print(f"[DATA] Saved Juliet test split ({len(test_pairs)} pairs across {len({p['cwe'] for p in test_pairs})} CWEs) -> {test_json}")

    return calib_pairs, test_pairs


def get_shortest_calib_demonstration(
    dataset_name: str,
    data_dir: str = DEFAULT_DATA_DIR,
    splits_dir: str = DEFAULT_SPLITS_DIR,
) -> Dict[str, Any]:
    """
    Finds the shortest function pair in the calibration set to use as the fixed
    2-shot demonstration example. Excluded from scoring.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "juliet":
        split_path = os.path.join(splits_dir, "juliet_calib_split.json")
        raw_path = os.path.join(data_dir, "juliet_paired.jsonl")
        if not os.path.isfile(raw_path):
            raw_path = ensure_juliet_data(data_dir)
        if not os.path.isfile(split_path):
            generate_juliet_splits(raw_path, output_dir=splits_dir)

        with open(split_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        raw_items = load_primevul_raw(raw_path)
        raw_map = {item["idx"]: item for item in raw_items}

        best_pair = None
        min_len = float("inf")
        for p in meta["pairs"]:
            v_idx = p["vuln_idx"]
            s_idx = p["safe_idx"]
            if v_idx in raw_map and s_idx in raw_map:
                v_code = raw_map[v_idx]["func"]
                s_code = raw_map[s_idx]["func"]
                tot_len = len(v_code) + len(s_code)
                if tot_len < min_len:
                    min_len = tot_len
                    best_pair = {
                        "pair_id": p["pair_id"],
                        "vuln_idx": v_idx,
                        "safe_idx": s_idx,
                        "vuln_code": v_code,
                        "safe_code": s_code,
                        "cwe": p.get("cwe", "Unknown"),
                        "project": "Juliet",
                    }
        if best_pair is None:
            raise RuntimeError(f"Failed to find shortest calib demonstration for {dataset_name}")
        return best_pair

    elif dataset_name == "primevul_unpaired":
        split_path = os.path.join(splits_dir, "primevul_unpaired_calib_split.json")
        if not os.path.isfile(split_path):
            generate_primevul_unpaired_splits(output_dir=splits_dir)

        with open(split_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        samples = meta["samples"]
        vulns = [s for s in samples if s["target"] == 1]
        safes = [s for s in samples if s["target"] == 0]
        if not vulns or not safes:
            raise RuntimeError("PrimeVul unpaired calib split lacks vuln or safe samples")

        shortest_v = min(vulns, key=lambda s: len(s["func"]))
        shortest_s = min(safes, key=lambda s: len(s["func"]))

        return {
            "pair_id": f"unpaired_demo_{shortest_v['idx']}_{shortest_s['idx']}",
            "vuln_idx": shortest_v["idx"],
            "safe_idx": shortest_s["idx"],
            "vuln_code": shortest_v["func"],
            "safe_code": shortest_s["func"],
            "cwe": f"{shortest_v.get('cwe', 'Unknown')}/{shortest_s.get('cwe', 'Unknown')}",
            "project": "PrimeVul_Unpaired",
        }

    else:
        split_path = os.path.join(splits_dir, "calib_split.json")
        raw_path = os.path.join(data_dir, "primevul_valid_paired.jsonl")
        if not os.path.isfile(raw_path):
            v, _ = ensure_primevul_data(data_dir)
            raw_path = v
        if not os.path.isfile(split_path):
            v, t = ensure_primevul_data(data_dir)
            generate_splits(v, t, output_dir=splits_dir)

        with open(split_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        raw_items = load_primevul_raw(raw_path)
        raw_map = {item["idx"]: item for item in raw_items}

        best_pair = None
        min_len = float("inf")

        for p in meta["pairs"]:
            v_idx = p["vuln_idx"]
            s_idx = p["safe_idx"]
            if v_idx in raw_map and s_idx in raw_map:
                v_code = raw_map[v_idx]["func"]
                s_code = raw_map[s_idx]["func"]
                tot_len = len(v_code) + len(s_code)
                if tot_len < min_len:
                    min_len = tot_len
                    best_pair = {
                        "pair_id": p["pair_id"],
                        "vuln_idx": v_idx,
                        "safe_idx": s_idx,
                        "vuln_code": v_code,
                        "safe_code": s_code,
                        "cwe": p.get("cwe", "Unknown"),
                        "project": p.get("project", "Unknown"),
                    }

        if best_pair is None:
            raise RuntimeError(f"Failed to find shortest calib demonstration for {dataset_name}")

        return best_pair


def load_dataset_split(
    dataset_name: str,
    split: str = "calib",
    data_dir: str = DEFAULT_DATA_DIR,
    splits_dir: str = DEFAULT_SPLITS_DIR,
    tokenizer=None,
    max_tokens: int = MAX_TOKENS_LIMIT,
) -> Dict[str, Any]:
    """
    Unified dataset loader supporting 'primevul_paired', 'primevul_unpaired', and 'juliet'.
    Enforces strict token length filtering and pair preservation.
    """
    dataset_name = dataset_name.lower()
    split = split.lower()

    if dataset_name == "primevul_paired":
        split_path = os.path.join(splits_dir, f"{split}_split.json")
        raw_name = f"primevul_{split}_paired.jsonl" if split == "test" else "primevul_valid_paired.jsonl"
        raw_path = os.path.join(data_dir, raw_name)
        if not os.path.isfile(split_path) or not os.path.isfile(raw_path):
            v, t = ensure_primevul_data(data_dir)
            raw_path = t if split == "test" else v
            if not os.path.isfile(split_path):
                generate_splits(v, t, output_dir=splits_dir, tokenizer=tokenizer)

        with open(split_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        raw_items = load_primevul_raw(raw_path)
        raw_map = {item["idx"]: item for item in raw_items}

        valid_pairs = []
        ordered_samples = []
        for p in meta["pairs"]:
            v_idx = p["vuln_idx"]
            s_idx = p["safe_idx"]
            if v_idx in raw_map and s_idx in raw_map:
                v_item = dict(raw_map[v_idx])
                s_item = dict(raw_map[s_idx])
                v_item["pair_id"] = p["pair_id"]
                s_item["pair_id"] = p["pair_id"]

                len_v = count_tokens(v_item["func"], tokenizer)
                len_s = count_tokens(s_item["func"], tokenizer)
                if len_v <= max_tokens and len_s <= max_tokens:
                    valid_pairs.append({
                        "pair_id": p["pair_id"],
                        "vuln": v_item,
                        "safe": s_item,
                        "cwe": p.get("cwe", "Unknown"),
                        "project": p.get("project", "Unknown"),
                    })
                    ordered_samples.extend([v_item, s_item])

        return {
            "dataset": dataset_name,
            "split": split,
            "is_paired": True,
            "samples": ordered_samples,
            "pairs": valid_pairs,
        }

    elif dataset_name == "primevul_unpaired":
        split_path = os.path.join(splits_dir, f"primevul_unpaired_{split}_split.json")
        if not os.path.isfile(split_path):
            generate_primevul_unpaired_splits(output_dir=splits_dir, tokenizer=tokenizer)

        with open(split_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        samples = []
        for s in meta["samples"]:
            if count_tokens(s["func"], tokenizer) <= max_tokens:
                samples.append(s)

        return {
            "dataset": dataset_name,
            "split": split,
            "is_paired": False,
            "samples": samples,
            "pairs": None,
        }

    elif dataset_name == "juliet":
        split_path = os.path.join(splits_dir, f"juliet_{split}_split.json")
        raw_path = os.path.join(data_dir, "juliet_paired.jsonl")
        if not os.path.isfile(split_path) or not os.path.isfile(raw_path):
            raw_path = ensure_juliet_data(data_dir)
            if not os.path.isfile(split_path):
                generate_juliet_splits(raw_path, output_dir=splits_dir, tokenizer=tokenizer)

        with open(split_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        raw_items = load_primevul_raw(raw_path)
        raw_map = {item["idx"]: item for item in raw_items}

        valid_pairs = []
        ordered_samples = []
        for p in meta["pairs"]:
            v_idx = p["vuln_idx"]
            s_idx = p["safe_idx"]
            if v_idx in raw_map and s_idx in raw_map:
                v_item = dict(raw_map[v_idx])
                s_item = dict(raw_map[s_idx])
                v_item["pair_id"] = p["pair_id"]
                s_item["pair_id"] = p["pair_id"]

                len_v = count_tokens(v_item["func"], tokenizer)
                len_s = count_tokens(s_item["func"], tokenizer)
                if len_v <= max_tokens and len_s <= max_tokens:
                    valid_pairs.append({
                        "pair_id": p["pair_id"],
                        "vuln": v_item,
                        "safe": s_item,
                        "cwe": p.get("cwe", "Unknown"),
                        "project": "Juliet",
                    })
                    ordered_samples.extend([v_item, s_item])

        return {
            "dataset": dataset_name,
            "split": split,
            "is_paired": True,
            "samples": ordered_samples,
            "pairs": valid_pairs,
        }

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Must be 'primevul_paired', 'primevul_unpaired', or 'juliet'.")


if __name__ == "__main__":
    print("=== AQCAP DATA INGESTION & DATASET SETUP ===")
    print("\n1. Ensuring PrimeVul paired data & splits...")
    valid_p, test_p = ensure_primevul_data()
    calib_p, test_p = generate_splits(valid_p, test_p)
    print(f"   [SUCCESS] PrimeVul paired: {len(calib_p)} calib pairs, {len(test_p)} test pairs")

    print("\n2. Ensuring PrimeVul unpaired data & splits...")
    ensure_primevul_unpaired_data()
    calib_unp, test_unp = generate_primevul_unpaired_splits()
    print(f"   [SUCCESS] PrimeVul unpaired: {len(calib_unp)} calib samples, {len(test_unp)} test samples")

    print("\n3. Ensuring Juliet C/C++ data & splits...")
    ensure_juliet_data()
    j_calib, j_test = generate_juliet_splits()
    print(f"   [SUCCESS] Juliet C/C++: {len(j_calib)} calib pairs, {len(j_test)} test pairs")
