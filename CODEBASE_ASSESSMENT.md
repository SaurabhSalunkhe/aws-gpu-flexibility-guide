# Assess an existing GPU codebase before migrating

A local, read-only static scanner turns code and configuration signals into a review checklist and a target-hardware validation plan. It helps answer **“What in this repository deserves attention before moving to this exact instance size?”**

It does not execute the application, inspect installed binaries, resolve dependencies, measure performance, or certify portability. **No customer repository or GPU workload was tested to create this tool.** All committed examples are synthetic.

## Quickstart

Use Python **3.10+ on Linux or macOS** with POSIX no-follow directory operations. Run from this **trusted guide checkout**, in a clean approved Python environment, with no customer repository on Python's import path. Do not run from the customer project's working directory. A hostile Python startup hook, site package, or `PYTHONPATH` can execute before this scanner starts; scanner traversal controls cannot sandbox the interpreter itself. Use an already-approved environment rather than installing project dependencies.

From the guide root:

```bash
python3 -B -m gpu_assess --help
python3 -B -m gpu_assess --list-instances
```

Scan an authorized local checkout without modifying it:

```bash
python3 -B -m gpu_assess /absolute/path/to/customer-repository \
  --source-instance g7e.48xlarge \
  --target-instance p5.48xlarge \
  --workload serving
```

The default is a Markdown report on stdout. JSON can be saved to a **new private file outside the scanned repository**:

```bash
python3 -B -m gpu_assess /absolute/path/to/customer-repository \
  --source-instance g7e.48xlarge \
  --target-instance p6-b300.48xlarge \
  --workload serving --format json \
  --output /existing/private/directory/new-migration-report.json
```

`--output` never overwrites an existing file, creates files with mode `0600`, refuses symlink path components, and does not create directories. Shell redirection is controlled by your shell and does **not** have these safeguards. Keep reports outside the repository and review them before sharing.

The source/target flags are optional but require an exact size when supplied. `p5` is not an accepted substitute for `p5.4xlarge` or `p5.48xlarge`. Workload choices are `serving`, `offline-inference`, `training`, and `unknown` (default). Unknown hardware/workload remains unknown; the tool does not guess from the customer's organization or filenames.

A symlinked checkout root, a symlink in its parent path, or a symlinked report directory is refused. Use the real directory path. The scanner fails closed when its secure traversal primitives are unavailable; it does not downgrade to unsafe following of links.

## Try the synthetic example first

```bash
python3 -B -m gpu_assess examples/codebase-fixture \
  --source-instance g7e.48xlarge --target-instance p5.48xlarge \
  --workload serving --format markdown
python3 -B -m unittest discover -s tests -p 'test_assessor.py' -v
```

The fixture is intentionally incomplete and must not be built, installed, or executed. It contains a CUDA container declaration, framework version tokens, Hopper/Blackwell build-target flags, a disabled-P2P reference, an interface override, fixed `cuda:7`, TP/PP/world-size values, a CPU copy, precision choices, a CUDA kernel marker, and training offload configuration.

The checked-in [sample Markdown](examples/codebase-assessment-sample.md) and [sample JSON](examples/codebase-assessment-sample.json) show the resulting report. They are **static triage examples, not measured migration results**. Their relative fixture paths are safe demonstration data, not a customer's code.

## What the report tells you

1. **Selected hardware and catalog differences.** Exact GPU model/count, compute capability, host CPU/ISA/memory, topology and transport capabilities retain their catalog source units and caveats. Source-reported GiB is not a measured allocatable-memory budget. Advertised network maxima are not achieved application bandwidth. Null/missing values remain unknown.
2. **Local evidence.** Relative path, one-based line number, a coarse context label, and allowlisted technical tokens. Raw source lines, command strings, private container names, URLs in dependency declarations, and environment-variable values are withheld.
3. **Why to review it.** Each signal has a suggested change/review and a specific validation step. These are review suggestions, not automatic edits.
4. **Unknowns and required experiments.** The matrix maps to numbered sections of the [existing target-hardware runbook](docs/10-validation-runbook.md). Every runtime check starts as `not_run`.
5. **Coverage.** Files/bytes scanned, skipped categories, limits and warnings make exclusions visible. A report with no findings or an empty known-issues list is never an “all clear.”

Categories have deliberately different meanings:

- `catalog_fact`: a sourced hardware difference, not an application failure.
- `heuristic`: text matched a rule; the scanner has not proved that it executes or controls the deployed workload.
- `runtime_unknown`: evidence that static scanning cannot obtain.
- `known_compatibility_issues`: reserved for strong, applicable, sourced incompatibility evidence. Version 1 leaves this list empty because text alone does not establish the resolved artifact or effective execution path. It never upgrades a version token into an incompatibility or vulnerability diagnosis.

**The scanner cannot deduce the effective NCCL, peer-to-peer, or EFA path.** An environment variable can be overridden, a branch can be unused, a deployment can add settings, and custom transfers can bypass NCCL. It also cannot establish memory fit, quality, throughput, latency, cost, capacity availability, or portability certification.

## Transparent rules and appropriate next steps

| Rule | Text signal | Review/change | Specific follow-up |
|---|---|---|---|
| `dependency` | Known framework/CUDA package names in selected manifests; bounded numeric version tokens | Resolve the actual wheel/build, runtime, driver and support matrix | Start the frozen image and exercise relevant kernels on target |
| `container` | Docker/Containerfile `FROM`; optional official CUDA version and Linux platform marker | Check base-image digest and host ISA; retain target-specific build profiles | Build/start exact target image; record resolved runtime inventory |
| `custom_kernel` | CUDA source files, extension APIs, Triton/JIT, FlashAttention/xFormers markers | Review compiler/artifact targets and fallback semantics | Test custom-kernel branches/shapes and reference correctness |
| `architecture` | `sm_`/`compute_` targets; `TORCH_CUDA_ARCH_LIST`; `CMAKE_CUDA_ARCHITECTURES` | Inspect actual cubin/PTX and architecture-specific assumptions | Inspect artifacts/build logs; run relevant kernels on each target |
| `device` | Fixed CUDA ordinal or `CUDA_VISIBLE_DEVICES` reference | Reconcile visible-device/local-rank mapping | Verify GPU identity and peak memory on every rank |
| `parallelism` | TP/PP/DP, local process count, world size, GPU count | Check group layout, partition rules and node boundaries | Validate rank layout, collective behavior and per-rank memory |
| `training_sharding` | DDP/FSDP/ZeRO/DeepSpeed markers | Preserve training/checkpoint semantics; revisit offload/group settings | Check gradients/loss, step time and checkpoint restore |
| `transport` | Selected NCCL and Libfabric/EFA control names | Review why overrides exist before changing them | Prove selection with every rank's logs plus workload traces |
| `peer_api` | CUDA peer APIs or NVSHMEM markers | Review peer guards, mapping, topology and fallback | Capability, copy measurements and application path evidence |
| `host_copy` | `.cpu()`, `.to('cpu')`, copy/offload markers | Identify intended copies versus a critical-path bottleneck | Profile H2D/D2H and synchronization, then compare end-to-end |
| `precision` | Dtype/quantization tokens and selected quantization keys | Check exact model format and kernel support | Separate quality/numerical tests from memory/performance tests |
| `host_pipeline` | DataLoader, worker/pinned-memory, threading, tokenizer markers | Check CPU pipeline, NUMA placement and overlap | Measure host-stage latency and GPU idle gaps |
| `entrypoint` | Docker entrypoint/CMD, Python main, scripts table, torchrun | Locate the real launch and configuration precedence | Freeze the actual invocation used in baseline and target tests |

These are lexical rules, not a parser, call graph, deployment resolver or binary analyzer. Full-line comments are skipped. Inline comments, quoted documentation, test fixtures, dead branches and unrelated symbols can still match. Test/example paths are labelled rather than silently treated as production. Version ranges are not resolved; a displayed version is a token, not an installed version or an exact pin. Split-line settings, aliases, dynamically computed values and indirect imports may be missed.

For example, `--tensor-parallel-size 8` is discoverable, while `tensor_parallel_size = choose_layout()` is unknown. A match for `NCCL_P2P_DISABLE` does not say whether its value was zero, one, inherited, or effective. An occurrence of `compute_90` is not proof that PTX is embedded in the shipped artifact; build flags and output packaging must be inspected separately. The generic Blackwell name is insufficient: the offline map distinguishes RTX PRO 6000 Blackwell (12.0), B200 (10.0), B300 (10.3), and H100/H200 (9.0).

## Read-only and privacy boundaries

The scanner uses only the Python standard library. It never:

- Executes/imports customer Python, setup scripts, entrypoints or build files
- Invokes subprocesses, installs packages, fetches URLs, uploads data or applies fixes
- Reads the process environment to collect credentials or runtime settings
- Resolves requirements includes, imports, submodule metadata, links or external configuration
- Inspects model weights, checkpoints, compiled binaries or live GPU devices

It selects UTF-8 text by a small filename/extension allowlist. It skips `.git`, version-control metadata, dependency directories, virtual environments, common build/cache outputs, checkpoint directories, symlinks, hardlinks, FIFOs/devices, binary/NUL content, non-UTF-8 files and oversized files. Symlink protection uses directory file descriptors and no-follow opens, including the chosen root's parents.

Secret-like names and key formats are conservatively skipped. Examples include `.env*`, names containing `secret`, `credential`, `password` or `token`, private keys, `.aws`, `.ssh`, `.netrc`, `.npmrc`, and `.pypirc`. **This also skips innocent names such as `tokenizer.py`.** Tokenization elsewhere can be detected, but inspect such excluded code manually in your approved environment. This is privacy-oriented triage, not a comprehensive secret detector. Credentials hidden in ordinary source can be read into memory but are not copied into report values; paths and version metadata can still be confidential. Do not upload reports without reviewing them.

Relative paths replace non-ASCII, control, bidirectional and markup characters with `?`; unusual names can collide visually. Use the local repository to disambiguate rather than trusting a displayed path as an executable command. Never execute report contents. The tool emits no absolute repository path; error messages use fixed wording rather than source data.

No content-level customer file writes occur during scanning. Normal filesystem reads may update access times. Optional output is a separately requested new report outside the customer tree. The code is not an operating-system sandbox against a hostile interpreter, mounted filesystem or concurrent privileged attacker; use an approved isolated environment for untrusted checkouts.

### Bounded coverage

Defaults:

- 2,000 scanned files
- 512 KiB per file
- 32 MiB total bytes read, including ignore files
- 20,000 directory entries before stopping enumeration
- 16 nested directory levels
- 1,000 findings
- 8,192 characters per scanned line
- 2,048 inherited ignore rules; 512 characters per rule

The CLI exposes the first six as positive-integer `--max-*` flags. Raising limits consumes more resources; it does not improve rule semantics. Long lines are skipped and reported. A large directory that hits the entry limit is skipped rather than pretending its unsorted partial enumeration is complete. Some exclusions are intentional even without a limit warning; inspect coverage counts and method limits in every report.

### `.gitignore` handling

No `git` subprocess or Git configuration is read. The scanner loads local and nested `.gitignore` files using a documented subset:

- Blank lines and full-line `#` comments
- Relative patterns and root-of-that-file `/` anchoring
- `*`, `?`, character classes, whole-segment `**`
- Trailing `/` directory patterns
- Ordered `!` negation, including nested policy

As with Git, a file cannot be resurrected beneath a pruned ignored parent. Backslash escapes, trailing whitespace semantics, patterns longer than 512 characters, patterns deeper than 32 components, or excessive rule counts are unsupported. **An unsupported or unreadable `.gitignore` causes that entire subtree to be skipped**, with a warning. Built-in secret/generated exclusions always win. Global Git excludes, `.git/info/exclude`, Git attributes, linked worktree metadata and tracked-file exceptions are not resolved. Already-materialized submodule or nested-worktree files inside the selected root may be scanned unless excluded; no submodule metadata is fetched or followed. This may exclude tracked files that Git itself would retain; review is required, and findings are never a whole-repository guarantee.

## Turn a finding into a migration decision

1. Supply an authorized checkout, exact source/target size, workload class, model/checkpoint identity, important shapes/load and quality/SLO contract. The scanner accepts the first four; keep the remaining information in the private [benchmark plan](templates/benchmark-plan.md).
2. Review scope and warnings before interpreting findings. Locate the real entrypoint, trace settings and overrides, and mark irrelevant tests/examples locally.
3. Compare catalog differences and actual artifact support. Keep GB/GiB and unresolved source conflicts intact. Inspect resolved dependencies and compiled kernels without assuming declared versions are installed.
4. Create a target deployment profile. Make only intentional changes; static suggestions are not an authorization to alter source, drivers, security settings, network configuration or paid capacity.
5. Run the [hardware-to-application validation runbook](docs/10-validation-runbook.md) in an approved test allocation. Record physical capability, runtime path selection and workload benefit separately.
6. Qualify only the tested workload boundaries, retain a known-good profile, and define a rollback trigger. A passing static scan is never the promotion gate.

### Required runtime matrix

| Gate | What must be established | Runbook sections |
|---|---|---|
| Software and kernels | Exact image/runtime/driver/CPU ISA; relevant kernel correctness, build and cold start | 1, 5 |
| Memory and layout | Worst supported shapes and load; every rank below its measured memory budget; valid partitioning | 1, 6 |
| Host and peer paths | Correct GPU/NUMA pairs and message sizes; observed H2D/D2H and peer behavior | 2, 5 |
| Collectives and network | Actual group shapes; selected NCCL/plugin/provider/EFA path on every rank; custom communication | 3, 4, 5 |
| Quality and end-to-end SLO | Preset quality, realistic tails/load, warm/cold cases, repetitions, goodput and total cost | 6, 7 |
| Recovery and rollback | Drain/failure/restart, checkpoints where relevant, capacity replacement and rollback | 6, 7 |
| Training only | Equal effective batch/optimizer semantics, gradient/loss checks, checkpoint portability and fixed-target time | 5, 6 |

## Maintenance and source constraints

The tool consumes the guide's single [instance catalog](data/instance-catalog.json), preserving all 31 exact sizes, checked on 2026-10-11. Update that catalog, recheck primary sources, regenerate synthetic reports and rerun the tests; do not silently substitute generic GPU specifications for EC2 allocations.

The small compute-capability mapping uses NVIDIA's published GPU list. The architecture review rule points to NVIDIA's compatibility guide; NCCL references point to the documented control semantics; EFA checks use AWS's procedure. These sources explain the review, not a claim that this scanner detects every incompatibility. No framework/driver minimum-version compatibility database or vulnerability feed is bundled.

The source URLs and scope are recorded in [assessment sources](data/assessment-sources.json). All source lookups were performed during tool development; running the scanner is offline.
