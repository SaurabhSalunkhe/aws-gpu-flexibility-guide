"""Bounded standard-library-only static triage, not a compatibility certifier.

Read via directory file descriptors with no-follow opens. Never execute scanned
code, load project plugins, invoke subprocesses, resolve dependencies, or use a
network. Report fixed vocabulary plus narrowly validated technical tokens only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import fnmatch
import json
import os
from pathlib import Path
import re
import stat

BASE = Path(__file__).resolve().parents[1]
CATALOG = BASE / "data" / "instance-catalog.json"
SOURCES = {
    "cuda_arch": "https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html",
    "compute_capability": "https://developer.nvidia.com/cuda/gpus",
    "nccl": "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html",
    "efa": "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html",
    "ec2": "https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html",
}
# GPU identity only. No CUDA/framework minimum-version allowlist is inferred.
COMPUTE_CAPABILITY = {
    "NVIDIA L4": "8.9", "NVIDIA L40S": "8.9", "NVIDIA A100": "8.0",
    "NVIDIA H100": "9.0", "NVIDIA H200": "9.0", "NVIDIA B200": "10.0",
    "NVIDIA B300": "10.3",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": "12.0",
    "NVIDIA Blackwell (GB200 Grace Blackwell Superchip)": "10.0",
}
DENY_DIRS = frozenset({".git", ".hg", ".svn", "node_modules", "vendor", ".venv",
    "venv", "env", "__pycache__", ".tox", "build", "dist", ".cache", ".mypy_cache",
    ".pytest_cache", ".aws", ".ssh", ".gnupg", ".terraform", "checkpoints"})
SECRET_NAME = re.compile(r"(?i)(^\.env(?:$|[.])|secret|credential|password|token|^id_(?:rsa|ed25519|ecdsa)|^\.(?:netrc|npmrc|pypirc)$)")
SECRET_SUFFIX = {".pem", ".key", ".p12", ".pfx", ".jks", ".kdbx"}
EXTENSIONS = {".py", ".pyi", ".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".sh", ".bash", ".zsh", ".toml", ".json", ".yaml", ".yml", ".cfg", ".ini",
    ".cmake", ".txt", ".lock", ".mk", ".dockerfile"}
SPECIAL_NAMES = {"dockerfile", "containerfile", "makefile", "cmakelists.txt", "pipfile", "procfile"}
MANIFEST_NAMES = {"pyproject.toml", "setup.py", "setup.cfg", "pipfile", "pipfile.lock",
    "poetry.lock", "uv.lock", "environment.yml", "environment.yaml", "conda.yml", "conda.yaml"}
LIBRARIES = r"torch|torchvision|torchaudio|tensorflow|jaxlib|jax|vllm|tensorrt(?:-llm)?|triton|flash[-_]attn|xformers|deepspeed|transformers|accelerate|bitsandbytes|cuda[-_]python|cupy|nccl|cudnn"
LIB_RE = re.compile(r"(?i)(?<![\w.-])(" + LIBRARIES + r")(?![\w.-])")
VERSION_RE = re.compile(r"^\s*[\"']?\s*(?:[<>=!~^]{1,3}\s*)?(\d{1,2}(?:\.\d{1,3}){1,3}(?:(?:a|b|rc|dev|post)\d{1,3})?(?:\+cu\d{2,3})?)(?![\w.+-])")
ENV_RE = re.compile(r"\b(NCCL_(?:P2P_DISABLE|SHM_DISABLE|IB_DISABLE|NET_GDR_LEVEL|NET_PLUGIN|SOCKET_IFNAME|IB_HCA|P2P_LEVEL|COLLNET_ENABLE|ALGO|PROTO|NET)|FI_(?:PROVIDER|EFA_USE_DEVICE_RDMA|EFA_FORK_SAFE))\b")
ARCH_RE = re.compile(r"\b(?:sm|compute)_\d{2,3}[af]?\b", re.I)
COUNT_RE = re.compile(r"(?i)(?<![\w-])(?:--)?(tensor[-_]parallel[-_]size|tensor[-_]model[-_]parallel[-_]size|pipeline[-_]parallel[-_]size|pipeline[-_]model[-_]parallel[-_]size|data[-_]parallel[-_]size|nproc[-_]per[-_]node|world_size|num_gpus|gpus_per_node)[\"']?\s*(?:[:=]\s*|\s+)[\"']?(\d{1,5})(?![\w.])")
PRECISION_RE = re.compile(r"\b(?:bfloat16|float16|float32|bf16|fp16|fp32|fp8|fp4|int8|int4|awq|gptq|nf4|mxfp4)\b", re.I)

@dataclass(frozen=True)
class Limits:
    max_files: int = 2000
    max_file_bytes: int = 512 * 1024
    max_total_bytes: int = 32 * 1024 * 1024
    max_entries: int = 20000
    max_depth: int = 16
    max_findings: int = 1000
    max_line_chars: int = 8192
    max_ignore_rules: int = 2048

    def __post_init__(self):
        for value in asdict(self).values():
            if type(value) is not int or value <= 0:
                raise ValueError("Scan limits must be positive integers.")

@dataclass(frozen=True)
class IgnoreRule:
    base: str
    pattern: str
    negated: bool
    directories_only: bool
    anchored: bool

    def matches(self, relative: str, is_dir: bool) -> bool:
        if self.directories_only and not is_dir:
            return False
        prefix = self.base + "/" if self.base else ""
        if not relative.startswith(prefix):
            return False
        rel = relative[len(prefix):]
        if "/" not in self.pattern and not self.anchored:
            return fnmatch.fnmatchcase(rel.rsplit("/", 1)[-1], self.pattern)
        # Segment-based match: * never crosses /; ** matches zero or more segments.
        parts, pats = rel.split("/"), self.pattern.split("/")
        memo = {}
        def match(i, j):
            if (i, j) in memo:
                return memo[i, j]
            if j == len(pats):
                result = i == len(parts)
            elif pats[j] == "**":
                result = match(i, j + 1) or (i < len(parts) and match(i + 1, j))
            else:
                result = i < len(parts) and fnmatch.fnmatchcase(parts[i], pats[j]) and match(i + 1, j + 1)
            memo[i, j] = result
            return result
        return match(0, 0)


def parse_ignore(text: str, base: str, limit: int) -> tuple[list[IgnoreRule], bool]:
    """Supported subset documented in CODEBASE_ASSESSMENT.md; no git is invoked."""
    rules = []
    unsupported = False
    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        if len(raw) > 512 or "\\" in raw or raw != raw.rstrip() or raw.count("/") > 32:
            unsupported = True
            continue
        negated = raw.startswith("!")
        raw = raw[1:] if negated else raw
        anchored = raw.startswith("/")
        directories_only = raw.endswith("/")
        raw = raw.strip("/")
        if raw:
            if len(rules) >= limit:
                unsupported = True
                break
            rules.append(IgnoreRule(base, raw, negated, directories_only, anchored))
    return rules, unsupported


def ignored(relative: str, is_dir: bool, rules: list[IgnoreRule]) -> bool:
    answer = False
    for rule in rules:
        if rule.matches(relative, is_dir):
            answer = not rule.negated
    return answer


def load_catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def select_hardware(name: str | None, catalog: dict) -> dict | None:
    if name is None:
        return None
    for row in catalog["instances"]:
        if row["instance_type"] == name:
            result = dict(row)
            result["compute_capability"] = COMPUTE_CAPABILITY.get(row["gpu_model"])
            result["sources"] = [catalog["source_urls"][s] for s in row["source_ids"]]
            result["sources"].append(SOURCES["compute_capability"])
            return result
    raise ValueError("Instance not in the offline catalog; use --list-instances for exact sizes.")


def display_path(path: str) -> str:
    # Preserve useful relative paths, replace controls, bidi, non-ASCII and markup.
    return "".join(c if c.isascii() and (c.isalnum() or c in " /._-+") else "?" for c in path)


def context_for(path: str) -> str:
    parts = path.lower().split("/")
    if any(p in {"test", "tests", "example", "examples", "fixtures", "testdata", "docs"} for p in parts[:-1]) or parts[-1].startswith("test_"):
        return "test_or_example_text"
    return "unverified_source_or_configuration"


def is_manifest(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    return name in MANIFEST_NAMES or (name.startswith("requirements") and name.endswith(".txt"))


# Each finding includes an actionable review and a specific runtime check. Text
# never claims that a matching setting or branch is effective at runtime.
RULES = {
    "dependency": ("Framework or CUDA dependency declaration", "Record the resolved package, wheel, CUDA runtime and driver versions. Choose a target-supported combination from the engine/vendor support matrix; do not upgrade from this scan alone.", "Rebuild in a clean approved image, import the pinned runtime on target hardware, and execute representative kernels including custom extensions.", "1, 5"),
    "container": ("Container build/runtime declaration", "Review base-image digest, host CPU architecture, CUDA user-space libraries and driver requirements. Make separate target profiles where necessary.", "Build and start the exact image on each target; record image digest, host architecture, driver and resolved runtime versions.", "1, 5"),
    "custom_kernel": ("Custom or specialized GPU kernel signal", "Locate kernel owners and inspect wheel/fatbin targets, dispatch guards and build toolchain. Rebuild or select supported fallbacks only after checking semantics.", "Exercise every relevant kernel branch and shape on target hardware, compare outputs to a trusted reference, and capture compile/cold-start failures.", "1, 5, 6"),
    "architecture": ("Explicit GPU architecture target", "Compare declared targets with the selected GPU compute capability. Inspect built artifacts for compatible cubin/PTX; a text flag does not establish what ships. Architecture-specific a/f targets need separate compatibility review.", "Inspect actual artifacts and build logs; run representative custom kernels on each target. If appropriate, test the documented PTX JIT path separately and unset diagnostic overrides afterward.", "1, 5"),
    "device": ("Fixed GPU selection or device index", "Replace accidental ordinal assumptions with verified local-rank mapping or an explicit deployment profile. Keep deliberate single-device choices only where intended.", "Launch with the intended visible-device set and rank count; verify every rank's GPU identity and per-device peak memory.", "1, 5, 6"),
    "parallelism": ("Fixed parallelism or GPU-count configuration", "Reconcile local GPU count, global world size, TP/PP/DP and engine partition rules. A global group may span nodes; do not compare it with a node count as an automatic error.", "Test the proposed rank layout, shard divisibility, per-rank peak memory and actual collectives with the exact workload.", "1, 3, 4, 6"),
    "training_sharding": ("Distributed training or state-sharding signal", "Preserve optimizer/checkpoint semantics while reviewing FSDP/ZeRO/DDP groups, communication frequency and CPU offload for the new topology.", "Compare gradients/loss, effective batch, max-rank memory, step time, checkpoint restore and recovery under the target layout.", "3, 4, 5, 6"),
    "transport": ("NCCL or Libfabric/EFA override reference", "Determine whether this setting is effective and why it exists. Review target-specific transport defaults and remove obsolete workarounds only in a controlled comparison.", "Capture initialization logs on every rank, provider/plugin selection and representative collective traces. Confirm application paths, including custom communication that bypasses NCCL.", "2, 3, 4, 5"),
    "peer_api": ("Custom peer or GPU-initiated communication signal", "Review peer-access guards, rank/device mapping and topology assumptions. A direct API call may bypass NCCL; retain a tested fallback where the design requires it.", "Check supported peer pairs, correctness and measured transfers, then use an application trace to establish the path actually taken.", "1, 2, 5"),
    "host_copy": ("Potential host/device transfer or CPU offload", "Check whether the transfer is on the critical path, intended checkpoint/output work, or offload. Review batching, placement and overlap before changing it.", "Profile representative execution for H2D/D2H copies, synchronization and CPU gaps; compare quality and unprofiled end-to-end results after any change.", "2, 5, 6"),
    "precision": ("Precision or quantization selection", "Verify engine/kernel support for the exact precision and model format on each GPU. Keep quality tolerances fixed; this scanner makes no datatype support verdict.", "Compare task quality and numerical behavior with the agreed reference, then measure memory, latency and goodput for each proposed precision.", "5, 6"),
    "host_pipeline": ("Host preprocessing or data-loader signal", "Review CPU threads, affinity, NUMA placement, pinned buffers and preprocessing throughput against the target host; do not assume more GPU capacity removes a host bottleneck.", "Measure host-stage latency and GPU idle gaps under representative input, concurrency and storage conditions.", "1, 2, 5, 6"),
    "entrypoint": ("Potential workload entry point", "Identify the actual serving/training launch and its deployed overrides. Trace configuration precedence manually; scanning does not resolve imports or launch scripts.", "Freeze the exact command and configuration for baseline and target, then use that same path in the end-to-end matrix.", "5, 6"),
}


class Scanner:
    def __init__(self, limits: Limits):
        self.limits = limits
        self.stats = Counter()
        self.findings = []
        self.warnings = set()
        self.seen = set()
        self.read_files = set()

    def add(self, rule, path, line, tokens=()):
        key = (rule, path, line, tuple(tokens))
        if key in self.seen:
            return
        if len(self.findings) >= self.limits.max_findings:
            self.warnings.add("finding_limit_reached")
            return
        self.seen.add(key)
        title, change, validate, sections = RULES[rule]
        sources = [SOURCES["cuda_arch"]] if rule in {"architecture", "custom_kernel"} else []
        if rule == "transport":
            sources = [SOURCES["nccl"], SOURCES["efa"]]
        self.findings.append({"rule_id": rule, "category": "heuristic", "title": title,
            "evidence": {"path": display_path(path), "line": line,
                         "context": context_for(path), "tokens": list(tokens)},
            "interpretation": "Text signal only; reachability, precedence and effective runtime configuration are unverified.",
            "suggested_change": change, "validation": validate,
            "runbook_sections": sections, "sources": sources})

    def scan_text(self, text: str, path: str):
        suffix = Path(path).suffix.lower()
        if suffix in {".cu", ".cuh"}:
            self.add("custom_kernel", path, 1, ["CUDA source file"])
        name = path.rsplit("/", 1)[-1].lower()
        docker = name in {"dockerfile", "containerfile"} or name.startswith(("dockerfile.", "containerfile.")) or suffix == ".dockerfile"
        c_block = False
        for number, raw in enumerate(text.splitlines(), 1):
            if len(raw) > self.limits.max_line_chars:
                self.stats["long_lines_skipped"] += 1
                self.warnings.add("long_lines_skipped")
                continue
            s = raw.strip()
            # Ignore full-line comments; this is not a language parser. Quoted
            # documentation, inline comments and dead branches can still match.
            if c_block:
                if "*/" in s:
                    c_block = False
                continue
            if s.startswith("/*"):
                c_block = "*/" not in s
                continue
            if not s or s.startswith(("#", "//", ";")):
                continue
            if is_manifest(path):
                for m in LIB_RE.finditer(s):
                    lib = m.group(1).lower().replace("_", "-")
                    vm = VERSION_RE.match(s[m.end():].lstrip("\"' :"))
                    tokens = [lib]
                    if vm:
                        tokens.append("version token (constraint not resolved): " + vm.group(1))
                    self.add("dependency", path, number, tokens)
                # TOML lock name/version on separate lines: no attempted association.
                if re.search(r"(?i)\b(?:cuda|cudatoolkit|cuda-version)\b", s):
                    self.add("dependency", path, number, ["CUDA dependency reference"])
            if docker and re.match(r"(?i)^FROM\s+", s):
                tokens = ["FROM (image reference withheld)"]
                cuda = re.search(r"\bnvidia/cuda:(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\b", s)
                if cuda:
                    tokens.append("CUDA image version: " + cuda.group(1))
                platform = re.search(r"--platform\s*=\s*linux/(amd64|arm64)\b", s)
                if platform:
                    tokens.append("linux/" + platform.group(1))
                self.add("container", path, number, tokens)
            if re.search(r"\b(?:CUDAExtension|cpp_extension|load_inline|RawKernel|RawModule|__global__|__device__)\b|@(?:triton\.)?jit\b|\b(?:flash_attn|flash-attn|xformers)\b", s):
                self.add("custom_kernel", path, number, ["specialized kernel or extension marker"])
            arch = sorted(set(x.lower() for x in ARCH_RE.findall(s)))
            if re.search(r"\b(?:TORCH_CUDA_ARCH_LIST|CMAKE_CUDA_ARCHITECTURES)\b", s):
                marker = "TORCH_CUDA_ARCH_LIST" if "TORCH_CUDA_ARCH_LIST" in s else "CMAKE_CUDA_ARCHITECTURES"
                # Only tiny architecture-form tokens, never the entire assignment.
                arch += [marker]
                tail = s.split(marker, 1)[1]
                pattern = r"(?<![\w.])(\d{1,2}\.\d(?:\+PTX)?)(?![\w.])" if marker == "TORCH_CUDA_ARCH_LIST" else r"(?<![\w.])(\d{2,3}[af]?(?:-(?:real|virtual))?)(?![\w.])"
                for target in re.findall(pattern, tail):
                    arch.append("declared target: " + target)
            if arch:
                self.add("architecture", path, number, sorted(set(arch)))
            env = sorted(set(ENV_RE.findall(s)))
            if env:
                self.add("transport", path, number, env)
            if re.search(r"\b(?:cudaDeviceEnablePeerAccess|cudaMemcpyPeer|cudaMemcpyPeerAsync|nvshmem_[A-Za-z0-9_]+)\b", s):
                self.add("peer_api", path, number, ["peer or GPU-initiated communication API marker"])
            device = []
            if re.search(r"\bCUDA_VISIBLE_DEVICES\b", s):
                device.append("CUDA_VISIBLE_DEVICES (value withheld)")
            for m in re.finditer(r"(?<![\w/])cuda:(\d{1,4})(?![\w.])|\b(?:set_device|\.cuda)\s*\(\s*(\d{1,4})\s*\)", s):
                device.append("fixed CUDA ordinal: " + (m.group(1) or m.group(2)))
            if device:
                self.add("device", path, number, sorted(set(device)))
            counts = [m.group(1).lower().replace("-", "_") + ": " + m.group(2) for m in COUNT_RE.finditer(s)]
            if counts:
                self.add("parallelism", path, number, counts)
            if re.search(r"\b(?:FSDP|FullyShardedDataParallel|DistributedDataParallel|zero_optimization|zero_stage|deepspeed)\b", s):
                self.add("training_sharding", path, number, ["distributed training/sharding marker"])
            if re.search(r"\.cpu\s*\(|\.to\s*\(\s*[\"']cpu[\"']|\b(?:offload_param|offload_optimizer|cpu_offload|cudaMemcpy|cudaMemcpyAsync|cudaMemcpyHostToDevice|cudaMemcpyDeviceToHost)\b", s):
                self.add("host_copy", path, number, ["host/device copy or offload marker"])
            precision = sorted(set(x.lower() for x in PRECISION_RE.findall(s)))
            if precision or re.search(r"\b(?:load_in_4bit|load_in_8bit|quantization_config)\b", s):
                self.add("precision", path, number, precision or ["quantization marker"])
            if re.search(r"\b(?:DataLoader|pin_memory|num_workers|OMP_NUM_THREADS|MKL_NUM_THREADS|tokenizer|tokenize|numactl)\b", s):
                self.add("host_pipeline", path, number, ["host pipeline marker"])
            if (docker and re.match(r"(?i)^(?:ENTRYPOINT|CMD)\s+", s)) or re.search(r"__name__\s*==\s*[\"']__main__[\"']|\[project\.scripts\]|\b(?:console_scripts|torchrun)\b", s):
                self.add("entrypoint", path, number, ["launch declaration (command withheld)"])

    def read_regular(self, directory_fd: int, name: str, *, ignore_file=False):
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            fd = os.open(name, flags, dir_fd=directory_fd)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    self.stats["non_regular_skipped"] += 1
                    return None
                if info.st_nlink > 1:
                    self.stats["hardlinks_skipped"] += 1
                    return None
                if info.st_size > self.limits.max_file_bytes:
                    self.stats["oversize_skipped"] += 1
                    return None
                # Counts ignore-file reads too; reads stop before the total bound.
                remaining = self.limits.max_total_bytes - self.stats["bytes_read"]
                if info.st_size + 1 > remaining:
                    self.warnings.add("total_byte_limit_reached")
                    return None
                content = os.read(fd, min(self.limits.max_file_bytes + 1, remaining))
                self.stats["bytes_read"] += len(content)
                if len(content) > self.limits.max_file_bytes:
                    self.stats["oversize_skipped"] += 1
                    return None
                if b"\x00" in content:
                    self.stats["binary_skipped"] += 1
                    return None
                try:
                    result = content.decode("utf-8-sig")
                except UnicodeDecodeError:
                    self.stats["non_utf8_skipped"] += 1
                    return None
                if ignore_file:
                    self.stats["ignore_files_read"] += 1
                return result
            finally:
                os.close(fd)
        except OSError:
            self.stats["unreadable_skipped"] += 1
            return None

    def walk(self, directory_fd: int, prefix: str, inherited: list[IgnoreRule], depth: int):
        if depth > self.limits.max_depth:
            self.warnings.add("depth_limit_reached")
            return
        rules = inherited
        try:
            # Bound directory enumeration too; giant directories are not sorted
            # partially and silently claimed complete.
            names = []
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    self.stats["entries_seen"] += 1
                    if self.stats["entries_seen"] > self.limits.max_entries:
                        self.warnings.add("entry_limit_reached")
                        return
                    names.append(entry.name)
            if ".gitignore" in names:
                ignore_text = self.read_regular(directory_fd, ".gitignore", ignore_file=True)
                if ignore_text is None:
                    # Fail closed for a subtree when its ignore policy is unreadable.
                    self.warnings.add("subtree_skipped_unreadable_gitignore")
                    return
                local, unsupported = parse_ignore(ignore_text, prefix, self.limits.max_ignore_rules - len(inherited))
                if unsupported:
                    self.warnings.add("subtree_skipped_unsupported_gitignore")
                    return
                rules = inherited + local
            for name in sorted(names):
                if name == ".gitignore":
                    continue
                if self.stats["files_scanned"] >= self.limits.max_files:
                    self.warnings.add("file_limit_reached")
                    return
                if "total_byte_limit_reached" in self.warnings or "entry_limit_reached" in self.warnings:
                    return
                rel = prefix + "/" + name if prefix else name
                if name.lower() in DENY_DIRS or SECRET_NAME.search(name) or Path(name).suffix.lower() in SECRET_SUFFIX:
                    self.stats["sensitive_or_generated_skipped"] += 1
                    continue
                try:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        self.stats["symlinks_skipped"] += 1
                        continue
                    is_dir = stat.S_ISDIR(info.st_mode)
                    if ignored(rel, is_dir, rules):
                        self.stats["gitignored_skipped"] += 1
                        continue
                    if is_dir:
                        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
                        try:
                            self.walk(child, rel, rules, depth + 1)
                        finally:
                            os.close(child)
                    elif stat.S_ISREG(info.st_mode):
                        if Path(name).suffix.lower() not in EXTENSIONS and name.lower() not in SPECIAL_NAMES and not name.lower().startswith(("dockerfile.", "containerfile.")):
                            self.stats["unselected_extension_skipped"] += 1
                            continue
                        content = self.read_regular(directory_fd, name)
                        if content is not None:
                            self.stats["files_scanned"] += 1
                            self.read_files.add(rel)
                            self.scan_text(content, rel)
                    else:
                        self.stats["non_regular_skipped"] += 1
                except OSError:
                    self.stats["unreadable_skipped"] += 1
        except OSError:
            self.stats["unreadable_directories"] += 1
            self.warnings.add("unreadable_directory")


def runtime_matrix(workload: str) -> list[dict]:
    rows = [
        ("software_and_kernels", "1, 5", "Exact image, driver/runtime, host ISA, GPU compute capability and custom-kernel correctness on each source/target profile."),
        ("memory_and_layout", "1, 6", "Peak allocated/reserved memory on every rank at the largest supported shapes, context, concurrency or training batch; valid TP/PP/DP partitioning."),
        ("host_and_peer_paths", "2, 5", "H2D/D2H and peer copies for relevant size/pair/NUMA placements, then application traces showing effective transfer paths."),
        ("collectives_and_network", "3, 4, 5", "Actual rank layout and message sizes; NCCL/provider logs on every rank; intended plugin/EFA selection and custom non-NCCL paths."),
        ("end_to_end_quality_slo", "6, 7", "Predeclared quality/SLO acceptance, representative load/tails, warm/cold starts, independent repetitions, goodput and cost per qualifying result."),
        ("recovery_and_rollback", "6, 7", "Drain/failure/restart, checkpoint restore where relevant, replacement capacity, and a verified rollback profile."),
    ]
    if workload == "training":
        rows.append(("training_semantics", "5, 6", "Equal effective batch and optimizer semantics; gradient/loss checks; step-time spread; checkpoint portability and time to a fixed quality target."))
    return [{"id": key, "runbook_sections": sections, "required_evidence": evidence, "status": "not_run"} for key, sections, evidence in rows]


def open_directory_no_symlinks(path) -> int:
    """Open an absolute path component by component; reject every symlink."""
    path = Path(os.path.abspath(Path(path).expanduser()))
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def assess(root, source=None, target=None, workload="unknown", limits=None) -> dict:
    if workload not in {"unknown", "serving", "offline-inference", "training"}:
        raise ValueError("Unsupported workload; choose unknown, serving, offline-inference or training.")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY") or os.open not in os.supports_dir_fd or os.scandir not in os.supports_fd:
        raise ValueError("Secure no-follow directory traversal requires a supported POSIX Python environment.")
    catalog = load_catalog()
    src, dst = select_hardware(source, catalog), select_hardware(target, catalog)
    scanner = Scanner(limits or Limits())
    try:
        root_fd = open_directory_no_symlinks(root)
    except (OSError, RuntimeError):
        raise ValueError("Cannot open the requested repository directory.") from None
    try:
        scanner.walk(root_fd, "", [], 0)
    finally:
        os.close(root_fd)
    changes = []
    if src and dst:
        for field in ("gpu_model", "gpu_count", "architecture", "cpu", "vcpu", "host_memory_gib", "gpu_memory_type", "ec2_spec_reported_per_device_gib", "topology", "nvlink", "network_max_gbps", "network_is_burst_ceiling", "efa_supported", "efa_generation", "gpudirect_rdma_supported", "requires_ultraserver", "compute_capability"):
            if src.get(field) != dst.get(field):
                changes.append({"field": field, "source": src.get(field), "target": dst.get(field),
                    "category": "catalog_fact", "meaning": "Catalog platform difference; application impact is unverified. Network rates are advertised maxima, not achieved bandwidth; missing values are unknown."})
    if dst:
        for finding in scanner.findings:
            tokens = finding["evidence"]["tokens"]
            if finding["rule_id"] == "architecture":
                finding["target_context"] = (f"Selected target GPU compute capability is {dst['compute_capability']}. "
                    "Compare actual compiled artifacts and dispatch support; different text targets alone do not establish incompatibility.")
            elif finding["rule_id"] == "device":
                outside = [int(t.rsplit(": ", 1)[1]) for t in tokens if t.startswith("fixed CUDA ordinal: ") and int(t.rsplit(": ", 1)[1]) >= dst["gpu_count"]]
                if outside:
                    finding["target_context"] = (f"An observed ordinal exceeds the default zero-based range for {dst['gpu_count']} GPUs in this exact target instance. "
                        "Review whether this code is used and how devices are exposed; this is not an executed-device error.")
            elif finding["rule_id"] == "container":
                expected = "linux/arm64" if dst["architecture"] == "arm64" else "linux/amd64"
                if any(t.startswith("linux/") and t != expected for t in tokens):
                    finding["target_context"] = (f"Observed container platform differs from target host ISA ({dst['architecture']}). "
                        "Review the actual multi-architecture image and build/runtime selection; a declaration may be an unused build stage.")
    scanner.findings.sort(key=lambda f: (f["evidence"]["path"], f["evidence"]["line"], f["rule_id"], f["evidence"]["tokens"]))
    stats = dict(sorted(scanner.stats.items()))
    stats.setdefault("files_scanned", 0)
    return {"schema_version": 1, "tool": "gpu_assess", "assessment": "static_triage_only",
        "portability_verdict": "not_determined", "workload": workload,
        "repository": "local repository (absolute path withheld)",
        "catalog_checked_on": catalog["checked_on"], "source_hardware": src, "target_hardware": dst,
        "hardware_differences": changes, "limits": asdict(scanner.limits), "coverage": stats,
        "coverage_warnings": sorted(scanner.warnings),
        "findings": scanner.findings,
        "finding_counts": dict(sorted(Counter(f["rule_id"] for f in scanner.findings).items())),
        "known_compatibility_issues": [],
        "known_issue_policy": "None asserted from text alone. A flag or declared version is not a resolved artifact or an executed path; an empty list is not evidence of compatibility.",
        "runtime_unknowns": [
            {"category": "runtime_unknown", "id": "effective_software", "detail": "Installed driver/runtime, resolved dependencies, image digest, external or generated configuration, branch reachability and built cubin/PTX are unknown."},
            {"category": "runtime_unknown", "id": "effective_transport", "detail": "Cannot deduce effective NCCL/P2P/EFA transport, fallback, overlap or custom communication from static text."},
            {"category": "runtime_unknown", "id": "fit_quality_performance", "detail": "Cannot determine per-rank memory fit, numerical quality, throughput, latency, cost or certify portability without target-hardware tests."},
        ],
        "validation_matrix": runtime_matrix(workload),
        "safety": {"customer_code_executed": False, "network_used": False,
                   "customer_files_modified": False, "source_lines_in_report": False,
                   "notes": "Allowlisted technical tokens and sanitized relative paths only. Paths and dependency versions can still be confidential; review locally before sharing."},
        "method_limitations": [
            "Lexical patterns, not AST/data-flow analysis; inline comments, quoted strings, examples and dead code can match.",
            "Full-line comments are skipped. Test/example paths are labelled, not excluded. Aliases, computed values, split-line assignments and unsupported formats may be missed.",
            "No requirements includes, imports, submodule metadata, symlinks, hardlinks, container images or external paths are resolved. Materialized nested files may be scanned. No binary introspection or vulnerability diagnosis.",
            "Bounded UTF-8 allowlist scan with a documented .gitignore subset; unsupported/unreadable ignore policy skips its whole subtree.",
            "File extensions and filenames can hide relevant code; no findings never means no migration work.",
        ]}


def render_markdown(report: dict) -> str:
    lines = ["# GPU codebase migration triage", "", "Static text review only. Portability verdict: **not determined**.", "",
        f"Workload: {report['workload']}. Catalog reviewed: {report['catalog_checked_on']}.",
        f"Files scanned: {report['coverage']['files_scanned']}. Findings: {len(report['findings'])}.",
        "Absolute repository path and source lines are withheld. Keep this report private until reviewed.", "",
        "## Hardware scope"]
    for label, row in (("Source", report["source_hardware"]), ("Target", report["target_hardware"])):
        if row:
            lines.append(f"- {label}: {row['instance_type']}; {row['gpu_count']} × {row['gpu_model']}; CPU {row['architecture']}; GPU compute capability {row['compute_capability']}.")
            for note in row.get("notes", []):
                lines.append("  - Catalog caveat: " + note)
            if row.get("memory_source_values"):
                lines.append("  - Preserved memory source values: " + json.dumps(row["memory_source_values"], sort_keys=True))
            if row.get("gpudirect_rdma_source_values"):
                lines.append("  - Preserved GPUDirect source values: " + json.dumps(row["gpudirect_rdma_source_values"], sort_keys=True))
            lines += ["  - Catalog source: " + url for url in row["sources"]]
        else:
            lines.append(f"- {label}: not supplied; exact-size comparison unavailable.")
    if report["hardware_differences"]:
        lines += ["", "Catalog differences (facts, not application compatibility verdicts):"]
        for item in report["hardware_differences"]:
            lines.append(f"- {item['field']}: {item['source']} → {item['target']}")
    lines += ["", "Network rates are advertised maxima, not achieved bandwidth; missing values are unknown. Nominal GB and source-reported GiB are not normalized into an allocatable-memory guarantee. Preserve catalog uncertainties and verify device readings.", "", "## Coverage"]
    lines += [f"- {key}: {value}" for key, value in report["coverage"].items()]
    lines += [f"- Warning: {warning}" for warning in report["coverage_warnings"]]
    lines += ["", "## Findings"]
    if not report["findings"]:
        lines.append("No selected signals found. This is not evidence of compatibility or a complete scan.")
    for n, f in enumerate(report["findings"], 1):
        e = f["evidence"]
        lines += ["", f"### {n}. {f['title']} ({f['category']})", f"- Evidence: {e['path']}:{e['line']} ({e['context']})",
            "- Tokens: " + ", ".join(e["tokens"]), f"- Meaning: {f['interpretation']}",
            f"- Review/change: {f['suggested_change']}", f"- Validate: {f['validation']}",
            f"- Guide: docs/10-validation-runbook.md, sections {f['runbook_sections']}."]
        if finding_context := f.get("target_context"):
            lines.append("- Target context: " + finding_context)
        lines += [f"- Source: {url}" for url in f["sources"]]
    lines += ["", "## What remains unknown", "", report["known_issue_policy"]]
    lines += ["- " + u["detail"] for u in report["runtime_unknowns"]]
    lines += ["", "## Target-hardware validation matrix (not run)"]
    for row in report["validation_matrix"]:
        lines.append(f"- {row['id']} — runbook sections {row['runbook_sections']}: {row['required_evidence']}")
    lines += ["", "## Method and privacy limits"]
    lines += ["- " + item for item in report["method_limitations"]]
    lines += ["- " + report["safety"]["notes"], ""]
    return "\n".join(lines)
