import unittest

from prism.execution import ExampleVerifier, _tokens_match
from prism.preprocess import parse_examples, query_views, split_sections

STATEMENT = """Sasha wants the cheapest trip between $n$ cities.


-----Input-----

The first line contains two integers $n$ and $v$.


-----Output-----

Print one integer.


-----Examples-----
Input
4 2

Output
4

Input
7 6

Output
6



-----Note-----

In the first example the answer is 4.
"""

CORRECT = """n, v = map(int, input().split())
if v >= n - 1:
    print(n - 1)
else:
    print(v - 1 + (n - v) * (n - v + 1) // 2)
"""

WRONG = "n, v = map(int, input().split())\nprint(n * v)\n"
SLOW = "while True:\n    pass\n"


class PreprocessTest(unittest.TestCase):
    def test_sections_and_views(self):
        sections = split_sections(STATEMENT)
        self.assertEqual(set(sections), {"statement", "input", "output", "examples", "note"})
        views = query_views(STATEMENT)
        self.assertTrue(views["core"].startswith("Sasha"))
        self.assertNotIn("Input", views["core"])
        self.assertIn("two integers", views["io"])
        self.assertEqual(query_views("plain query")["io"], "plain query")

    def test_examples(self):
        self.assertEqual(parse_examples(STATEMENT), [("4 2\n", "4"), ("7 6\n", "6")])
        self.assertEqual(parse_examples("no examples here"), [])


class ExecutionTest(unittest.TestCase):
    def test_token_match(self):
        self.assertTrue(_tokens_match("1 2\n3\n", "1 2 3"))
        self.assertTrue(_tokens_match("0.3333333", "0.33333333"))
        self.assertFalse(_tokens_match("5", "6"))

    def test_rerank_promotes_passing_solution(self):
        verifier = ExampleVerifier(timeout=1.0, workers=2)
        candidates = [("wrong", WRONG, 0.9), ("slow", SLOW, 0.8), ("right", CORRECT, 0.7), ("tail", WRONG, 0.1)]
        ranked, info = verifier.rerank(STATEMENT, candidates, top_k=3)
        self.assertEqual(ranked[0][0], "right")
        self.assertEqual([d for d, _ in ranked], ["right", "wrong", "slow", "tail"])
        self.assertEqual(info, {"examples": 2, "verified": 1})


if __name__ == "__main__":
    unittest.main()
