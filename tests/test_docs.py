"""Run the commands readers see and check repository links without network access."""

import csv
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import unittest
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "inference_8b": (24.901161193847656, 68, "within_budget_screen"),
    "inference_8b_long": (80.90116119384766, 68, "exceeds_budget_screen"),
    "inference_70b_tp2": (77.1925802230835, 68, "exceeds_budget_screen"),
    "inference_70b_tp4": (39.59629011154175, 68, "within_budget_screen"),
    "inference_70b_4bit": (58.50784492492676, 68, "within_budget_screen"),
    "inference_70b_single160": (134.885160446167, 136, "within_budget_screen"),
    "training_8b_zero3": (39.80232238769531, 68, "within_budget_screen"),
    "training_8b_zero3_fp32grad": (43.52761268615723, 68, "within_budget_screen"),
    "training_70b_tp8": (142.385160446167, 136, "exceeds_budget_screen"),
}


class DocumentationTests(unittest.TestCase):
    def test_documented_estimator_commands(self):
        pattern = re.compile(r"<!-- smoke-test: ([a-z0-9_]+) -->\s*```bash\n(.*?)```", re.S)
        seen = set()
        for path in sorted(ROOT.rglob("*.md")):
            for name, command in pattern.findall(path.read_text()):
                with self.subTest(path=str(path.relative_to(ROOT)), name=name):
                    seen.add(name)
                    args = shlex.split(command.replace("\\\n", " "))
                    self.assertEqual(args[:3], ["python3", "-m", "gpu_fit"])
                    # Never invoke a shell or execute arbitrary documentation code.
                    result = subprocess.run([sys.executable, *args[1:], "--json"], cwd=ROOT,
                                            text=True, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    data = json.loads(result.stdout)
                    estimate, budget, decision = EXPECTED[name]
                    self.assertAlmostEqual(data["estimated_per_gpu_gib"], estimate, places=8)
                    self.assertAlmostEqual(data["usable_budget_per_gpu_gib"], budget, places=8)
                    self.assertEqual(data["decision"], decision)
        self.assertEqual(seen, set(EXPECTED), "Every reference scenario must have a runnable documented example")

    def test_relative_markdown_links_exist(self):
        count = 0
        for path in sorted(ROOT.rglob("*.md")):
            for destination in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
                parsed = urlparse(destination)
                if parsed.scheme or not parsed.path:
                    continue
                count += 1
                target = (path.parent / unquote(parsed.path)).resolve()
                self.assertTrue(target.is_relative_to(ROOT), f"Link leaves repository: {path}: {destination}")
                self.assertTrue(target.exists(), f"Broken link: {path}: {destination}")
        self.assertGreater(count, 40)

    def test_local_heading_links_exist(self):
        def anchors(path):
            slugs = set()
            counts = {}
            for heading in re.findall(r"(?m)^#{1,6}\s+(.+)$", path.read_text()):
                slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
                number = counts.get(slug, 0)
                counts[slug] = number + 1
                slugs.add(f"{slug}-{number}" if number else slug)
            return slugs

        checked = 0
        for path in sorted(ROOT.rglob("*.md")):
            for destination in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
                parsed = urlparse(destination)
                if parsed.scheme or not parsed.fragment:
                    continue
                target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
                self.assertTrue(target.is_file(), f"Missing anchor target: {destination}")
                self.assertIn(unquote(parsed.fragment), anchors(target), f"Broken heading link: {path.name}: {destination}")
                checked += 1
        self.assertGreaterEqual(checked, 10)

    def test_catalog_is_sourced_and_uniquely_keyed(self):
        data = json.loads((ROOT / "data/instance-catalog.json").read_text())
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["checked_on"], "2026-10-11")
        rows = data["instances"]
        names = [row["instance_type"] for row in rows]
        self.assertEqual(len(rows), 31)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sum(row["architecture"] == "arm64" for row in rows), 1)
        for row in rows:
            with self.subTest(instance=row["instance_type"]):
                self.assertGreater(row["gpu_count"], 0)
                self.assertTrue(row["source_ids"])
                self.assertTrue(set(row["source_ids"]) <= set(data["source_urls"]))
        index = {row["instance_type"]: row for row in rows}
        self.assertEqual(index["g7e.12xlarge"]["gpu_count"], 2)
        self.assertEqual(index["g6e.12xlarge"]["gpu_count"], 4)
        self.assertEqual(index["g6e.16xlarge"]["gpu_count"], 1)
        self.assertFalse(index["g7e.8xlarge"]["gpudirect_rdma_supported"])
        self.assertTrue(index["g7e.12xlarge"]["gpudirect_rdma_supported"])
        self.assertFalse(index["p5.4xlarge"]["gpudirect_rdma_supported"])
        self.assertIsNone(index["p4d.24xlarge"]["gpudirect_rdma_supported"])
        self.assertIsNone(index["p4de.24xlarge"]["gpudirect_rdma_supported"])
        self.assertTrue(index["p6e-gb200.36xlarge"]["requires_ultraserver"])

    def test_benchmark_csv_has_no_fabricated_measurements(self):
        with (ROOT / "templates/benchmark-results.csv").open(newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), len(set(rows[0])))
        self.assertIn("successful_slo_passing_requests", rows[0])
        self.assertIn("peak_gpu_memory_gib_max_rank", rows[0])

    def test_source_ledger_covers_external_document_links(self):
        ledger = json.loads((ROOT / "data/source-ledger.json").read_text())
        known = {entry["url"] for entry in ledger["sources"]}
        for path in [*sorted(ROOT.glob("*.md")), *sorted((ROOT / "docs").glob("*.md"))]:
            for url in re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", path.read_text()):
                self.assertIn(url, known, f"Untracked source in {path.name}: {url}")


if __name__ == "__main__":
    unittest.main()
