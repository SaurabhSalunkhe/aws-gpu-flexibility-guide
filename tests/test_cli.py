import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = ["inference", "--parameters", "8e9", "--gpu-memory-gib", "80", "--layers", "32",
        "--query-heads", "32", "--kv-heads", "8", "--head-dim", "128", "--context-tokens", "8192",
        "--concurrency", "8", "--runtime-overhead-gib", "2"]


def run(*args):
    return subprocess.run([sys.executable, "-m", "gpu_fit", *args], cwd=ROOT, text=True, capture_output=True, timeout=10)


class CliTests(unittest.TestCase):
    def test_json_output_contains_assumptions(self):
        result = run(*BASE, "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["inputs"]["parameters"], 8_000_000_000)
        self.assertAlmostEqual(data["estimated_per_gpu_gib"], 24.901161193847656)
        self.assertTrue(data["warnings"])

    def test_human_output_is_screen_not_fit_guarantee(self):
        result = run(*BASE)
        self.assertEqual(result.returncode, 0)
        self.assertIn("24.90 GiB", result.stdout)
        self.assertIn("within_budget_screen", result.stdout)
        self.assertIn("not evidence", result.stdout)

    def test_budget_exceeded_is_valid_computation(self):
        result = run(*BASE, "--gpu-memory-gib", "1", "--json")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["decision"], "exceeds_budget_screen")

    def test_invalid_numbers_exit_two_without_traceback(self):
        for flag, value in (("--parameters", "8B"), ("--parameters", "nan"), ("--parameters", "0"),
                            ("--parameters", "1.1"), ("--parameters", "1e1000000"),
                            ("--gpu-memory-gib", "inf"), ("--headroom-fraction", "1"),
                            ("--tp", "3"), ("--context-tokens", "-1")):
            with self.subTest(flag=flag, value=value):
                result = run(*BASE, flag, value)
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_missing_inputs(self):
        result = run("inference")
        self.assertEqual(result.returncode, 2)

    def test_training_json(self):
        result = run("training", "--parameters", "8e9", "--gpu-memory-gib", "80", "--activation-gib", "8", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["mode"], "training")

    def test_help(self):
        for args in (("--help",), ("inference", "--help"), ("training", "--help")):
            with self.subTest(args=args):
                self.assertEqual(run(*args).returncode, 0)


if __name__ == "__main__":
    unittest.main()
