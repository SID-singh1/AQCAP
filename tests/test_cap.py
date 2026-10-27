"""
Unit tests for aqcap.cap module as specified in ROADMAP.md Section 3.6.
"""

import unittest
from aqcap.cap import prune_code, prune_c_cpp, prune_python


class TestCAPEngine(unittest.TestCase):
    def test_empty_input(self):
        pruned, comments, kw = prune_code("", language="c")
        self.assertEqual(pruned, "")
        self.assertEqual(comments, [])
        self.assertEqual(kw, [])

    def test_cpp_string_embedded_comment_tokens_survive(self):
        # "/* not a comment */" inside a string survives
        code = 'char *msg = "/* not a comment */"; // real comment'
        pruned, comments, _ = prune_code(code, language="c")
        self.assertIn('"/* not a comment */"', pruned)
        self.assertNotIn("// real comment", pruned)

    def test_cpp_slash_slash_in_string_survives(self):
        # "//" inside a string survives
        code = 'const char *url = "https://example.com/api"; // line comment'
        pruned, comments, _ = prune_code(code, language="cpp")
        self.assertIn('"https://example.com/api"', pruned)
        self.assertNotIn("// line comment", pruned)

    def test_cpp_raw_strings_survive(self):
        # C++ raw strings are left untouched
        code = r'const char *re = R"(regex with /* and // here)"; /* comment */'
        pruned, comments, _ = prune_code(code, language="cpp")
        self.assertIn(r'R"(regex with /* and // here)"', pruned)
        self.assertNotIn("/* comment */", pruned)

    def test_python_triple_quoted_sql_assignment_survives(self):
        # The triple-quoted SQL case must survive (Bug fix from Section 3.6)
        code = '''def get_users(db):
    """Fetch all active users from db."""
    query = """
    SELECT id, name
    FROM users
    WHERE status = 'active'
    """
    return db.execute(query) # execute query
'''
        pruned, comments, _ = prune_code(code, language="python")
        # Docstring must be removed
        self.assertNotIn("Fetch all active users from db.", pruned)
        # Assigned SQL query must survive!
        self.assertIn("SELECT id, name", pruned)
        self.assertIn("FROM users", pruned)
        self.assertIn("query = \"\"\"", pruned)
        # Line comment stripped
        self.assertNotIn("# execute query", pruned)

    def test_security_keyword_information_loss_detection(self):
        code = """// WARNING: vulnerable to stack overflow CVE-2023-1234
void process(char *s) {
    char buf[10];
    strcpy(buf, s);
}"""
        pruned, comments, kw = prune_code(code, language="c")
        self.assertIn("vuln", kw)
        self.assertIn("overflow", kw)
        self.assertIn("cve", kw)


if __name__ == "__main__":
    unittest.main()
