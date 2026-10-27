"""
AQCAP Context-Aware Token Pruning (CAP) Engine.

Handles:
1. Lexical comment stripping and whitespace normalization for C/C++ and Python.
2. Strict preservation of string literals, char literals, and C++ raw string literals.
3. Bug Fix (Section 3.6): In Python, strips triple-quoted strings ONLY if they are
   standalone statements (docstrings), preserving multi-line strings assigned to variables (e.g. SQL queries).
4. Information loss detection: checks for security keywords inside removed comments.
"""

import re
from typing import Tuple, List, Set

SECURITY_KEYWORDS: Set[str] = {
    "vuln", "overflow", "cve", "fix", "secur", "sanitiz", "check", "bound",
    "leak", "patch", "exploit", "underflow", "inject"
}


def _detect_comment_security_keywords(removed_comments: List[str]) -> List[str]:
    """Finds any security-relevant keywords in comments stripped by the pruner."""
    found = set()
    combined = " ".join(removed_comments).lower()
    for kw in SECURITY_KEYWORDS:
        if kw in combined:
            found.add(kw)
    return sorted(list(found))


def prune_c_cpp(raw_code: str) -> Tuple[str, List[str]]:
    """
    Prunes C/C++ source code:
    - Leaves raw string literals R"(...)" untouched.
    - Leaves standard double-quote and single-quote strings untouched.
    - Erases /* ... */ block comments and // ... line comments.
    - Normalizes horizontal and vertical whitespace.

    Returns:
        (pruned_code, list_of_removed_comments)
    """
    removed_comments = []

    # Regex matches tokens from left to right.
    # Group 1: Raw strings R"delim(...)delim"
    # Group 2: Regular double-quoted strings "..."
    # Group 3: Single-quoted character literals '...'
    # Group 4: Block comments /* ... */
    # Group 5: Line comments // ...
    pattern = re.compile(
        r'(?P<raw_string>R"(?P<delim>[a-zA-Z0-9_]*)\([\s\S]*?\)(?P=delim)")|'
        r'(?P<string>"(?:\\.|[^"\\])*")|'
        r"(?P<char>'(?:\\.|[^'\\])*')|"
        r'(?P<bcomment>/\*[\s\S]*?\*/)|'
        r'(?P<lcomment>//[^\r\n]*)'
    )

    def replacer(match):
        if match.group('raw_string') is not None:
            return match.group('raw_string')
        if match.group('string') is not None:
            return match.group('string')
        if match.group('char') is not None:
            return match.group('char')

        # Stripping comments
        if match.group('bcomment') is not None:
            removed_comments.append(match.group('bcomment'))
            return ' '
        if match.group('lcomment') is not None:
            removed_comments.append(match.group('lcomment'))
            return ''

        return match.group(0)

    pruned = pattern.sub(replacer, raw_code)

    # Whitespace normalization (safe for C/C++)
    pruned = re.sub(r'[ \t]+$', '', pruned, flags=re.MULTILINE)   # trailing whitespace
    pruned = re.sub(r'\n[ \t]*\n', '\n', pruned)                  # consecutive empty lines
    pruned = re.sub(r'^[ \t]+', ' ', pruned, flags=re.MULTILINE)  # squash leading indentation to single space

    return pruned.strip(), removed_comments


def prune_python(raw_code: str) -> Tuple[str, List[str]]:
    """
    Prunes Python source code:
    - Preserves all indentation (horizontal indentation is semantically critical in Python).
    - Preserves normal single/double quote strings.
    - Strips # comments.
    - BUG FIX (Section 3.6): Strips triple-quoted strings ONLY if they are standalone statements
      (docstrings), i.e., preceded on their line by only whitespace and either starting the file
      or preceded by a line ending with ':' or a blank line.
      Assigned triple-quoted strings (e.g. query = \"\"\"SELECT...\"\"\") are strictly preserved!

    Returns:
        (pruned_code, list_of_removed_comments)
    """
    removed_comments = []

    # 1. Parse line by line to distinguish standalone docstrings vs assigned strings
    lines = raw_code.splitlines()
    cleaned_lines = []

    in_docstring = False
    docstring_delim = None
    docstring_buf = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # If currently inside a multiline docstring
        if in_docstring:
            if docstring_delim in line:
                # End of docstring
                end_idx = line.find(docstring_delim) + len(docstring_delim)
                docstring_buf.append(line[:end_idx])
                removed_comments.append("\n".join(docstring_buf))
                in_docstring = False
                docstring_delim = None
                docstring_buf = []
                remaining = line[end_idx:].strip()
                if remaining:
                    cleaned_lines.append(remaining)
            else:
                docstring_buf.append(line)
            continue

        # Check if line begins a standalone docstring
        is_standalone = False
        delim = None
        if stripped.startswith('"""'):
            delim = '"""'
        elif stripped.startswith("'''"):
            delim = "'''"

        if delim is not None:
            # Check prefix on the same line before stripped
            prefix = line[:line.find(delim)]
            # Must be only whitespace preceding it
            if prefix.strip() == "":
                # Check previous line context
                prev_line = lines[i - 1].strip() if i > 0 else ""
                if i == 0 or prev_line == "" or prev_line.endswith(":") or prev_line.endswith('"""') or prev_line.endswith("'''"):
                    is_standalone = True

        if is_standalone:
            # Check if docstring closes on the same line
            rest_of_line = stripped[3:]
            if delim in rest_of_line:
                # One-line docstring: strip it
                removed_comments.append(stripped)
                continue
            else:
                in_docstring = True
                docstring_delim = delim
                docstring_buf = [line]
                continue

        # Line is regular Python code: strip trailing # comments (while respecting strings)
        line_pat = re.compile(
            r'(?P<string_double>"(?:\\.|[^"\\])*")|'
            r"(?P<string_single>'(?:\\.|[^'\\])*')|"
            r'(?P<comment>#[^\r\n]*)'
        )

        def line_replacer(match):
            if match.group('string_double') is not None:
                return match.group('string_double')
            if match.group('string_single') is not None:
                return match.group('string_single')
            if match.group('comment') is not None:
                removed_comments.append(match.group('comment'))
                return ''
            return match.group(0)

        pruned_line = line_pat.sub(line_replacer, line)
        cleaned_lines.append(pruned_line)

    pruned = "\n".join(cleaned_lines)

    # Vertical whitespace normalization only (never modify leading spaces)
    pruned = re.sub(r'[ \t]+$', '', pruned, flags=re.MULTILINE)
    pruned = re.sub(r'\n[ \t]*\n', '\n', pruned)

    return pruned.strip(), removed_comments


def prune_code(raw_code: str, language: str = "c") -> Tuple[str, List[str], List[str]]:
    """
    Entrypoint for Context-Aware Token Pruning.

    Args:
        raw_code: Source code text.
        language: Language dialect ('c', 'cpp', 'c++', 'python', 'py').

    Returns:
        (pruned_code, removed_comments, security_keywords_found)
    """
    if not raw_code:
        return "", [], []

    lang = language.lower().strip()
    if lang in ("python", "py"):
        pruned, comments = prune_python(raw_code)
    else:
        # Default to C/C++ pruner for C, C++, Java, JS
        pruned, comments = prune_c_cpp(raw_code)

    sec_keywords = _detect_comment_security_keywords(comments)
    return pruned, comments, sec_keywords
