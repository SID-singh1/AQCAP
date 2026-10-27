import os
import sys
import re
import csv
import json
import random
from collections import defaultdict, Counter

FORBIDDEN_JULIET_SUBSTRINGS = ["bad", "good", "g2b", "b2g", "cwe", "omitbad", "omitgood"]

def strip_c_comments(code: str) -> str:
    """Removes block /* ... */ and line // ... comments preserving strings."""
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
    Preprocesses Juliet C/C++ code:
    1. Removes all OMITBAD / OMITGOOD macro lines and surrounding comments.
    2. Strips all comments (block and line).
    3. Sanitizes string/char literals.
    4. Renames identifiers containing bad/good/G2B/B2G/CWE/OMITBAD/OMITGOOD to neutral names (func_N, var_N).
    5. Normalizes whitespace and asserts zero leakage.
    """
    # Step 1: Remove lines with OMITBAD / OMITGOOD
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

    # Step 6: Verify zero leakage
    code_lower = code.lower()
    for sub in FORBIDDEN_JULIET_SUBSTRINGS:
        if sub in code_lower:
            idx = code_lower.find(sub)
            start = max(0, idx - 40)
            end = min(len(code), idx + 40)
            raise AssertionError(f"Leakage check failed! Substring '{sub}' found: ...{code[start:end]}...")

    return code

func_def_re = re.compile(
    r'(?:[a-zA-Z_][a-zA-Z0-9_*]*\s+)+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^;)]*\)\s*\{',
    re.MULTILINE
)

def has_valid_function_body(code: str) -> bool:
    """Verifies that the code contains at least one genuine function definition."""
    if not func_def_re.search(code):
        return False
    if '{' not in code or '}' not in code:
        return False
    body = code[code.find('{'):code.rfind('}')+1]
    return len(body.strip()) >= 15


if __name__ == "__main__":
    print("Verifying full Juliet dataset processing...")
    csv_path = "data/jts_c_1_3_test.csv"
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

            # base testcase (grouping multi-file variants e.g. 53a, 53b -> 53)
            base_fn = re.sub(r'_[0-9]{2}[a-z]\.c(?:pp)?$', '', fn, flags=re.I)

            pair_obj = {
                "row_idx": i,
                "pair_id": f"juliet_{i*2}",
                "filename": fn,
                "base_testcase": base_fn,
                "cwe": cwe,
                "vuln_code": pb,
                "safe_code": pg,
            }
            valid_pairs.append(pair_obj)
            by_cwe[cwe].append(pair_obj)
            by_testcase[base_fn].append(pair_obj)

            if (len(valid_pairs)) % 500 == 0:
                print(f"  Found {len(valid_pairs)} valid pairs so far ({len(by_cwe)} CWEs)...")

    print(f"\n[DONE] Total Valid Non-Leaking Pairs: {len(valid_pairs)}")
    print(f"Total Unique Base Testcases: {len(by_testcase)}")
    print(f"Total Unique CWEs: {len(by_cwe)}")
    print("Top 10 CWEs by count:")
    for cwe, count in Counter({k: len(v) for k, v in by_cwe.items()}).most_common(10):
        print(f"  {cwe}: {count} pairs")
