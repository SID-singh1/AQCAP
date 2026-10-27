"""
Unit test suite asserting that the Juliet preprocessing leakage audit
guarantees that zero label cues, comments, OMIT macros, and leaking identifiers survive.
"""

import os
import json
import unittest

from aqcap.data import (
    preprocess_juliet_function,
    has_valid_function_body,
    FORBIDDEN_JULIET_SUBSTRINGS,
    DEFAULT_SPLITS_DIR,
    DEFAULT_DATA_DIR,
    load_dataset_split,
)


class TestJulietLeakageAudit(unittest.TestCase):
    def test_synthetic_leakage_cases(self):
        """Tests that synthetic code with comments, macros, leaking identifiers, and strings is 100% sanitized."""
        dirty_bad_code = """/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE121_Stack_Based_Buffer_Overflow__CWE129_fgets_01.c
Label Definition File: CWE121_Stack_Based_Buffer_Overflow__CWE129.label.xml
*/
/*
 * @description
 * CWE: 121 Stack Based Buffer Overflow
 * BadSource: fgets Read data from the console using fgets()
 * GoodSource: Set data to a small, non-negative value
 * Sink:
 *    BadSink : Use data as an index in an array
 */

#include "std_testcase.h"
#include "CWE121_Stack_Based_Buffer_Overflow__CWE129_fgets_01.h"

#ifndef OMITBAD

int CWE121_Stack_Based_Buffer_Overflow__CWE129_badGlobal = 0;

void CWE121_badSink(int data)
{
    int buffer[10] = {0};
    /* POTENTIAL FLAW: Attempt to access an index out of bounds */
    if (data >= 0)
    {
        buffer[data] = 1;
    }
}

void CWE121_Stack_Based_Buffer_Overflow__CWE129_fgets_01_bad()
{
    int bad_data = -1;
    printLine("Calling bad()...");
    CWE121_badSink(bad_data);
    printLine("Finished bad()");
}

#endif /* OMITBAD */
"""

        clean_bad = preprocess_juliet_function(dirty_bad_code)
        clean_lower = clean_bad.lower()

        # Assert no forbidden substrings survive
        for sub in FORBIDDEN_JULIET_SUBSTRINGS:
            self.assertNotIn(sub, clean_lower, f"Forbidden substring '{sub}' survived in preprocessed bad code!")

        # Assert comments are stripped
        self.assertNotIn("/*", clean_bad)
        self.assertNotIn("*/", clean_bad)
        self.assertNotIn("TEMPLATE GENERATED", clean_bad)
        self.assertNotIn("POTENTIAL FLAW", clean_bad)

        # Assert macros are stripped
        self.assertNotIn("OMITBAD", clean_bad)
        self.assertNotIn("#ifndef", clean_bad)

        # Assert identifiers were renamed to neutral names
        self.assertIn("func_", clean_bad)
        self.assertIn("var_", clean_bad)

        # Assert function body remains intact and valid
        self.assertTrue(has_valid_function_body(clean_bad))

    def test_synthetic_good_leakage_cases(self):
        """Tests that good code with goodG2B, goodB2G, and OMITGOOD is 100% sanitized."""
        dirty_good_code = """/* TEMPLATE GENERATED TESTCASE FILE */
#include "std_testcase.h"

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    int good_data = 5;
    /* FIX: Use VirtualLock() */
    printLine("Calling good()...");
}

void CWE121_Stack_Based_Buffer_Overflow__CWE129_fgets_01_good()
{
    goodG2B();
}

#endif /* OMITGOOD */
"""
        clean_good = preprocess_juliet_function(dirty_good_code)
        clean_lower = clean_good.lower()

        for sub in FORBIDDEN_JULIET_SUBSTRINGS:
            self.assertNotIn(sub, clean_lower, f"Forbidden substring '{sub}' survived in preprocessed good code!")

        self.assertNotIn("OMITGOOD", clean_good)
        self.assertNotIn("goodG2B", clean_good)
        self.assertNotIn("FIX", clean_good)
        self.assertTrue(has_valid_function_body(clean_good))

    def test_calib_and_test_splits_leak_free(self):
        """Asserts that all functions in the generated calib and test splits have ZERO forbidden substrings."""
        calib_data = load_dataset_split("juliet", split="calib")
        test_data = load_dataset_split("juliet", split="test")

        self.assertEqual(len(calib_data["samples"]), 200)
        self.assertEqual(len(test_data["samples"]), 1000)

        for s in calib_data["samples"]:
            code_lower = s["func"].lower()
            for sub in FORBIDDEN_JULIET_SUBSTRINGS:
                self.assertNotIn(sub, code_lower, f"Leak '{sub}' found in calib sample {s['idx']}!")
            self.assertTrue(has_valid_function_body(s["func"]), f"Sample {s['idx']} lacks a valid function body!")

        for s in test_data["samples"]:
            code_lower = s["func"].lower()
            for sub in FORBIDDEN_JULIET_SUBSTRINGS:
                self.assertNotIn(sub, code_lower, f"Leak '{sub}' found in test sample {s['idx']}!")
            self.assertTrue(has_valid_function_body(s["func"]), f"Sample {s['idx']} lacks a valid function body!")

    def test_splits_are_disjoint_by_testcase(self):
        """Asserts that calib and test are strictly disjoint by testcase file and base testcase."""
        calib_data = load_dataset_split("juliet", split="calib")
        test_data = load_dataset_split("juliet", split="test")

        calib_json = os.path.join(DEFAULT_SPLITS_DIR, "juliet_calib_split.json")
        test_json = os.path.join(DEFAULT_SPLITS_DIR, "juliet_test_split.json")

        with open(calib_json, "r", encoding="utf-8") as f:
            c_meta = json.load(f)
        with open(test_json, "r", encoding="utf-8") as f:
            t_meta = json.load(f)

        c_testcases = {p.get("base_testcase", p["pair_id"]) for p in c_meta["pairs"]}
        t_testcases = {p.get("base_testcase", p["pair_id"]) for p in t_meta["pairs"]}

        overlap = c_testcases.intersection(t_testcases)
        self.assertEqual(len(overlap), 0, f"Overlapping testcases between calib and test: {overlap}")


if __name__ == "__main__":
    unittest.main()
