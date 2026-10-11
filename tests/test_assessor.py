"""Offline scanner tests use synthetic files only; never execute a scanned file."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gpu_assess.cli import main, write_report
from gpu_assess.scanner import Limits, assess, load_catalog, render_markdown

ROOT = Path(__file__).resolve().parents[1]


class AssessorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.repo = self.base / "customer"
        self.repo.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def put(self, name, text):
        p = self.repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def scan(self, **kw):
        return assess(self.repo, **kw)

    def rules(self, report):
        return {x["rule_id"] for x in report["findings"]}

    def test_synthetic_fixture_detects_expected_categories(self):
        report = assess(ROOT / "examples/codebase-fixture", "g7e.48xlarge", "p5.48xlarge", "serving")
        self.assertEqual(report["coverage"]["files_scanned"], 6)
        self.assertTrue({"dependency", "container", "custom_kernel", "architecture", "device", "parallelism", "training_sharding", "transport", "host_copy", "precision", "host_pipeline", "entrypoint"} <= self.rules(report))
        self.assertEqual(report["portability_verdict"], "not_determined")
        self.assertFalse(report["known_compatibility_issues"])

    def test_never_executes_customer_python_or_setup(self):
        sentinel = self.base / "executed"
        dangerous = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n"
        self.put("setup.py", dangerous)
        self.put("main.py", dangerous)
        with mock.patch("subprocess.run", side_effect=AssertionError("subprocess used")), mock.patch("socket.socket", side_effect=AssertionError("network used")):
            report = self.scan()
        self.assertFalse(sentinel.exists())
        self.assertEqual(report["coverage"]["files_scanned"], 2)

    def test_does_not_change_customer_bytes_or_directory(self):
        self.put("app.py", 'device="cuda:0"\n')
        before = {p.relative_to(self.repo): p.read_bytes() for p in self.repo.rglob("*") if p.is_file()}
        self.scan()
        after = {p.relative_to(self.repo): p.read_bytes() for p in self.repo.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.repo / "__pycache__").exists())

    def test_no_raw_source_or_secret_values_in_report(self):
        sentinel = "SECRET_CANARY_8d1e95f97c"
        self.put("launch.sh", f'export NCCL_NET="{sentinel}"\nPASSWORD="{sentinel}"\n')
        report = self.scan()
        for output in (json.dumps(report), render_markdown(report)):
            self.assertNotIn(sentinel, output)
            self.assertNotIn(str(self.repo), output)
        self.assertEqual(report["findings"][0]["evidence"]["tokens"], ["NCCL_NET"])

    def test_secret_generated_git_and_dependency_paths_skipped(self):
        for name in (".env", ".env.production", "secrets.yaml", "config/credentials.json", "id_rsa", "key.pem", "tokenizer.py", ".git/config.py", "node_modules/a.py", ".venv/a.py", "build/a.py"):
            self.put(name, 'device="cuda:1"\n')
        report = self.scan()
        self.assertEqual(report["coverage"]["files_scanned"], 0)
        self.assertFalse(report["findings"])

    def test_symlink_files_and_directories_not_followed(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "outside.py").write_text('device="cuda:8"\n')
        (self.repo / "link.py").symlink_to(outside / "outside.py")
        (self.repo / "linked-directory").symlink_to(outside, target_is_directory=True)
        report = self.scan()
        self.assertEqual(report["coverage"]["symlinks_skipped"], 2)
        self.assertFalse(report["findings"])

    def test_symlink_root_and_parent_rejected(self):
        link = self.base / "root-link"
        link.symlink_to(self.repo, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Cannot open"):
            assess(link)
        self.put("nested/app.py", "x=1\n")
        with self.assertRaisesRegex(ValueError, "Cannot open"):
            assess(link / "nested")

    def test_hardlinks_and_fifo_skipped(self):
        outside = self.base / "outside.py"
        outside.write_text('device="cuda:8"\n')
        os.link(outside, self.repo / "hard.py")
        os.mkfifo(self.repo / "pipe.py")
        report = self.scan()
        self.assertFalse(report["findings"])
        self.assertEqual(report["coverage"]["hardlinks_skipped"], 1)
        self.assertEqual(report["coverage"]["non_regular_skipped"], 1)

    def test_gitignore_nested_anchoring_negation_and_double_star(self):
        self.put(".gitignore", "/root-only.py\n*.tmp.py\n!keep.tmp.py\nignored/\n**/generated/*.py\n")
        self.put("root-only.py", 'device="cuda:1"')
        self.put("nested/root-only.py", 'device="cuda:1"')
        self.put("a.tmp.py", 'device="cuda:1"')
        self.put("keep.tmp.py", 'device="cuda:1"')
        self.put("ignored/no.py", 'device="cuda:1"')
        self.put("nested/generated/no.py", 'device="cuda:1"')
        self.put("nested/.gitignore", "local.py\n")
        self.put("nested/local.py", 'device="cuda:1"')
        paths = {f["evidence"]["path"] for f in self.scan()["findings"]}
        self.assertEqual(paths, {"nested/root-only.py", "keep.tmp.py"})

    def test_gitignore_does_not_resurrect_pruned_parent(self):
        self.put(".gitignore", "hidden/\n!hidden/allowed.py\n")
        self.put("hidden/allowed.py", 'device="cuda:1"')
        self.assertFalse(self.scan()["findings"])

    def test_materialized_nested_repo_files_are_scanned_without_metadata(self):
        self.put("nested/.git", "gitdir: /not-followed/private-metadata\n")
        self.put("nested/app.py", 'device="cuda:1"')
        report = self.scan()
        self.assertEqual(report["findings"][0]["evidence"]["path"], "nested/app.py")
        self.assertNotIn("private-metadata", json.dumps(report))

    def test_unsupported_gitignore_fails_closed(self):
        self.put(".gitignore", r"\#literal.py" + "\n")
        self.put("app.py", 'device="cuda:1"')
        report = self.scan()
        self.assertFalse(report["findings"])
        self.assertIn("subtree_skipped_unsupported_gitignore", report["coverage_warnings"])

    def test_symlink_gitignore_fails_closed(self):
        external = self.base / "ignore"
        external.write_text("*.py\n")
        (self.repo / ".gitignore").symlink_to(external)
        self.put("app.py", 'device="cuda:1"')
        report = self.scan()
        self.assertFalse(report["findings"])
        self.assertIn("subtree_skipped_unreadable_gitignore", report["coverage_warnings"])

    def test_binary_and_invalid_utf8_skipped(self):
        (self.repo / "binary.py").write_bytes(b'\x00device="cuda:0"')
        (self.repo / "utf.py").write_bytes(b'\xffdevice="cuda:0"')
        report = self.scan()
        self.assertEqual(report["coverage"]["binary_skipped"], 1)
        self.assertEqual(report["coverage"]["non_utf8_skipped"], 1)

    def test_size_file_and_total_byte_bounds(self):
        self.put("a.py", 'device="cuda:0"\n')
        self.put("b.py", 'device="cuda:1"\n')
        self.assertEqual(self.scan(limits=Limits(max_files=1))["coverage"]["files_scanned"], 1)
        report = self.scan(limits=Limits(max_file_bytes=2))
        self.assertEqual(report["coverage"]["oversize_skipped"], 2)
        report = self.scan(limits=Limits(max_total_bytes=20))
        self.assertLessEqual(report["coverage"]["bytes_read"], 20)
        self.assertIn("total_byte_limit_reached", report["coverage_warnings"])

    def test_depth_entry_and_finding_bounds(self):
        self.put("a.py", 'device="cuda:0"\nprecision="fp8"\n')
        self.put("nested/deep/a.py", 'device="cuda:1"')
        self.assertIn("depth_limit_reached", self.scan(limits=Limits(max_depth=1))["coverage_warnings"])
        report = self.scan(limits=Limits(max_entries=1))
        self.assertIn("entry_limit_reached", report["coverage_warnings"])
        report = self.scan(limits=Limits(max_findings=1))
        self.assertEqual(len(report["findings"]), 1)
        self.assertIn("finding_limit_reached", report["coverage_warnings"])

    def test_long_lines_skipped_and_reported(self):
        self.put("a.py", 'device="cuda:0"' + " " * 40)
        report = self.scan(limits=Limits(max_line_chars=20))
        self.assertFalse(report["findings"])
        self.assertIn("long_lines_skipped", report["coverage_warnings"])

    def test_full_line_comments_skipped_but_strings_remain_heuristic(self):
        self.put("a.py", '# cuda:0\n# NCCL_P2P_DISABLE=1\ntext="example cuda:1"\n')
        report = self.scan()
        self.assertEqual(len(report["findings"]), 1)
        self.assertEqual(report["findings"][0]["category"], "heuristic")
        self.assertEqual(report["findings"][0]["evidence"]["line"], 3)

    def test_documented_false_negative_dynamic_or_alias_values(self):
        self.put("a.py", 'index = 1\ndevice = "cuda:" + str(index)\nsize = int(user_input)\ntensor_parallel_size=size\n')
        report = self.scan()
        self.assertNotIn("device", self.rules(report))
        self.assertNotIn("parallelism", self.rules(report))
        self.assertEqual(report["portability_verdict"], "not_determined")

    def test_examples_are_labelled(self):
        self.put("tests/example.py", 'device="cuda:0"')
        self.assertEqual(self.scan()["findings"][0]["evidence"]["context"], "test_or_example_text")

    def test_transport_zero_one_and_external_overrides_not_verdicts(self):
        self.put("launch.sh", "NCCL_P2P_DISABLE=0\nNCCL_P2P_DISABLE=1\nFI_PROVIDER=efa\n")
        report = self.scan()
        self.assertEqual(report["finding_counts"]["transport"], 3)
        self.assertFalse(report["known_compatibility_issues"])
        self.assertIn("Cannot deduce effective NCCL", json.dumps(report))
        self.assertNotIn("FI_PROVIDER=efa", json.dumps(report))

    def test_container_image_does_not_become_device_ordinal(self):
        self.put("Dockerfile", "FROM nvidia/cuda:12.8.1-devel-ubuntu22.04\n")
        report = self.scan()
        self.assertIn("container", self.rules(report))
        self.assertNotIn("device", self.rules(report))

    def test_python_from_import_not_container(self):
        self.put("app.py", "from torch import nn\n")
        self.assertNotIn("container", self.rules(self.scan()))

    def test_numeric_gpu_and_cli_parallelism_markers(self):
        self.put("launch.sh", "torchrun --nproc-per-node 8 main.py --tensor-parallel-size=4\n")
        self.put("app.py", 'x.cuda(2)\ntorch.cuda.set_device(3)\nx.to("cuda:7")\n')
        report = self.scan()
        self.assertEqual(report["finding_counts"]["device"], 3)
        self.assertIn("tensor_parallel_size: 4", json.dumps(report))
        self.assertIn("nproc_per_node: 8", json.dumps(report))

    def test_architecture_lists_and_arch_specific_tokens(self):
        self.put("CMakeLists.txt", 'set(CMAKE_CUDA_ARCHITECTURES "90;100;120-real")\n')
        self.put("build.sh", 'TORCH_CUDA_ARCH_LIST="9.0 10.0+PTX"\nnvcc -gencode=arch=compute_90a,code=sm_90a\n')
        text = json.dumps(self.scan())
        for token in ("declared target: 90", "declared target: 100", "declared target: 120-real", "declared target: 10.0+PTX", "sm_90a", "compute_90a"):
            self.assertIn(token, text)

    def test_dependency_version_token_is_not_pin_or_vulnerability(self):
        self.put("requirements.txt", "torch==2.5.1\nvllm>=0.10.2\ntransformers @ https://private.example/SECRET_TOKEN\n")
        report = self.scan()
        text = json.dumps(report)
        self.assertIn("version token (constraint not resolved): 0.10.2", text)
        self.assertNotIn("private.example", text)
        self.assertNotIn("SECRET_TOKEN", text)
        self.assertFalse(report["known_compatibility_issues"])

    def test_custom_peer_api_signal(self):
        self.put("transfer.cpp", "cudaMemcpyPeerAsync(a, 0, b, 1, n, stream);\n")
        self.assertIn("peer_api", self.rules(self.scan()))

    def test_catalog_exact_sizes_cc_differences_and_uncertainty(self):
        report = self.scan(source="g7e.12xlarge", target="p6-b300.48xlarge")
        self.assertEqual(report["source_hardware"]["gpu_count"], 2)
        self.assertEqual(report["source_hardware"]["compute_capability"], "12.0")
        self.assertEqual(report["target_hardware"]["compute_capability"], "10.3")
        self.assertIsNone(report["target_hardware"]["nominal_gpu_memory_per_device_gb"])
        md = render_markdown(report)
        self.assertIn("2148", md)
        self.assertIn("conflicts", md)
        self.assertIn("https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html", md)
        self.assertTrue(all(x["category"] == "catalog_fact" for x in report["hardware_differences"]))

    def test_all_catalog_models_have_explicit_cc(self):
        catalog = load_catalog()
        self.assertEqual(len(catalog["instances"]), 31)
        for row in catalog["instances"]:
            self.assertIsNotNone(self.scan(target=row["instance_type"])["target_hardware"]["compute_capability"])

    def test_invalid_instance_workload_and_limits_rejected(self):
        with self.assertRaises(ValueError):
            self.scan(target="p5")
        with self.assertRaises(ValueError):
            self.scan(workload="anything")
        with self.assertRaises(ValueError):
            Limits(max_files=0)

    def test_catalog_snapshot_is_dated_and_has_unique_sizes(self):
        catalog = load_catalog()
        self.assertEqual(catalog["checked_on"], "2026-10-11")
        names = [row["instance_type"] for row in catalog["instances"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(catalog["schema_version"], 1)

    def test_synthetic_report_is_reproducible(self):
        report = assess(ROOT / "examples/codebase-fixture", "g7e.48xlarge", "p5.48xlarge", "serving")
        self.assertEqual(report, json.loads((ROOT / "examples/codebase-assessment-sample.json").read_text()))
        self.assertEqual(render_markdown(report), (ROOT / "examples/codebase-assessment-sample.md").read_text())

    def test_target_context_stays_conditional(self):
        self.put("Dockerfile", "FROM --platform=linux/amd64 nvidia/cuda:12.8.1-base\n")
        self.put("app.py", 'x.to("cuda:7")\n')
        report = self.scan(target="p6e-gb200.36xlarge")
        text = json.dumps(report)
        self.assertIn("Observed container platform differs", text)
        self.assertIn("ordinal exceeds", text)
        self.assertTrue(all(f["category"] == "heuristic" for f in report["findings"]))

    def test_results_deterministic(self):
        self.put("b.py", 'x.cpu()\n')
        self.put("a.py", 'device="cuda:0"\n')
        self.assertEqual(self.scan(), self.scan())

    def test_training_runtime_matrix_and_empty_scan_unknown(self):
        report = self.scan(workload="training")
        self.assertIn("training_semantics", {x["id"] for x in report["validation_matrix"]})
        self.assertTrue(all(x["status"] == "not_run" for x in report["validation_matrix"]))
        self.assertIn("not evidence of compatibility", render_markdown(report))

    def test_path_control_and_markup_sanitized(self):
        self.put("<script>\napp.py", 'device="cuda:0"')
        report = self.scan()
        p = report["findings"][0]["evidence"]["path"]
        self.assertNotIn("<", p)
        self.assertNotIn("\n", p)
        self.assertFalse(p.startswith("/"))

    def test_cli_json_stdout(self):
        self.put("main.py", 'device="cuda:0"')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = main([str(self.repo), "--format", "json"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(output.getvalue())["schema_version"], 1)

    def test_cli_optional_report_private_and_never_overwritten(self):
        output = self.base / "report.json"
        with contextlib.redirect_stderr(io.StringIO()):
            rc = main([str(self.repo), "--format", "json", "--output", str(output)])
            before = output.read_bytes()
            second = main([str(self.repo), "--output", str(output)])
        self.assertEqual(rc, 0)
        self.assertEqual(second, 2)
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_output_inside_repository_or_symlink_parent_refused(self):
        with self.assertRaises(ValueError):
            write_report(self.repo / "new.md", self.repo, "report")
        link = self.base / "linked-output"
        link.symlink_to(self.repo, target_is_directory=True)
        with self.assertRaises(OSError):
            write_report(link / "new.md", self.repo, "report")
        self.assertFalse((self.repo / "new.md").exists())

    def test_cli_errors_do_not_echo_customer_path_or_file_values(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            rc = main([str(self.repo / "CONFIDENTIAL_missing")])
        self.assertEqual(rc, 2)
        self.assertNotIn("CONFIDENTIAL", output.getvalue())
        self.assertNotIn(str(self.repo), output.getvalue())


if __name__ == "__main__":
    unittest.main()
