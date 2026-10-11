# GPU codebase migration triage

Static text review only. Portability verdict: **not determined**.

Workload: serving. Catalog reviewed: 2026-10-11.
Files scanned: 6. Findings: 26.
Absolute repository path and source lines are withheld. Keep this report private until reviewed.

## Hardware scope
- Source: g7e.48xlarge; 8 × NVIDIA RTX PRO 6000 Blackwell Server Edition; CPU x86_64; GPU compute capability 12.0.
  - Catalog caveat: Multi-GPU sizes support EFAv4 GPUDirect RDMA.
  - Catalog caveat: AWS nominal GB and EC2 GiB labels differ without numeric conversion.
  - Catalog source: https://aws.amazon.com/ec2/instance-types/g7e/
  - Catalog source: https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html
  - Catalog source: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html
  - Catalog source: https://developer.nvidia.com/cuda/gpus
- Target: p5.48xlarge; 8 × NVIDIA H100; CPU x86_64; GPU compute capability 9.0.
  - Catalog source: https://aws.amazon.com/ec2/instance-types/p5/
  - Catalog source: https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html
  - Catalog source: https://developer.nvidia.com/cuda/gpus

Catalog differences (facts, not application compatibility verdicts):
- gpu_model: NVIDIA RTX PRO 6000 Blackwell Server Edition → NVIDIA H100
- cpu: Intel Xeon Emerald Rapids → AMD EPYC 7R13
- gpu_memory_type: GDDR7 → HBM3
- ec2_spec_reported_per_device_gib: 96 → 80
- topology: GPUDirect P2P over PCIe; no NVLink → NVLink/NVSwitch
- nvlink: False → True
- network_max_gbps: 1600 → 3200
- efa_generation: 4 → 2
- compute_capability: 12.0 → 9.0

Network rates are advertised maxima, not achieved bandwidth; missing values are unknown. Nominal GB and source-reported GiB are not normalized into an allocatable-memory guarantee. Preserve catalog uncertainties and verify device readings.

## Coverage
- bytes_read: 1365
- entries_seen: 7
- files_scanned: 6
- ignore_files_read: 1

## Findings

### 1. Explicit GPU architecture target (heuristic)
- Evidence: CMakeLists.txt:2 (unverified_source_or_configuration)
- Tokens: CMAKE_CUDA_ARCHITECTURES, declared target: 100, declared target: 90
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Compare declared targets with the selected GPU compute capability. Inspect built artifacts for compatible cubin/PTX; a text flag does not establish what ships. Architecture-specific a/f targets need separate compatibility review.
- Validate: Inspect actual artifacts and build logs; run representative custom kernels on each target. If appropriate, test the documented PTX JIT path separately and unset diagnostic overrides afterward.
- Guide: docs/10-validation-runbook.md, sections 1, 5.
- Target context: Selected target GPU compute capability is 9.0. Compare actual compiled artifacts and dispatch support; different text targets alone do not establish incompatibility.
- Source: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html

### 2. Explicit GPU architecture target (heuristic)
- Evidence: CMakeLists.txt:3 (unverified_source_or_configuration)
- Tokens: compute_90a, sm_90a
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Compare declared targets with the selected GPU compute capability. Inspect built artifacts for compatible cubin/PTX; a text flag does not establish what ships. Architecture-specific a/f targets need separate compatibility review.
- Validate: Inspect actual artifacts and build logs; run representative custom kernels on each target. If appropriate, test the documented PTX JIT path separately and unset diagnostic overrides afterward.
- Guide: docs/10-validation-runbook.md, sections 1, 5.
- Target context: Selected target GPU compute capability is 9.0. Compare actual compiled artifacts and dispatch support; different text targets alone do not establish incompatibility.
- Source: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html

### 3. Container build/runtime declaration (heuristic)
- Evidence: Dockerfile:2 (unverified_source_or_configuration)
- Tokens: FROM (image reference withheld), CUDA image version: 12.4.1, linux/amd64
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Review base-image digest, host CPU architecture, CUDA user-space libraries and driver requirements. Make separate target profiles where necessary.
- Validate: Build and start the exact image on each target; record image digest, host architecture, driver and resolved runtime versions.
- Guide: docs/10-validation-runbook.md, sections 1, 5.

### 4. Explicit GPU architecture target (heuristic)
- Evidence: Dockerfile:3 (unverified_source_or_configuration)
- Tokens: TORCH_CUDA_ARCH_LIST, declared target: 9.0
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Compare declared targets with the selected GPU compute capability. Inspect built artifacts for compatible cubin/PTX; a text flag does not establish what ships. Architecture-specific a/f targets need separate compatibility review.
- Validate: Inspect actual artifacts and build logs; run representative custom kernels on each target. If appropriate, test the documented PTX JIT path separately and unset diagnostic overrides afterward.
- Guide: docs/10-validation-runbook.md, sections 1, 5.
- Target context: Selected target GPU compute capability is 9.0. Compare actual compiled artifacts and dispatch support; different text targets alone do not establish incompatibility.
- Source: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html

### 5. NCCL or Libfabric/EFA override reference (heuristic)
- Evidence: Dockerfile:4 (unverified_source_or_configuration)
- Tokens: NCCL_P2P_DISABLE
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Determine whether this setting is effective and why it exists. Review target-specific transport defaults and remove obsolete workarounds only in a controlled comparison.
- Validate: Capture initialization logs on every rank, provider/plugin selection and representative collective traces. Confirm application paths, including custom communication that bypasses NCCL.
- Guide: docs/10-validation-runbook.md, sections 2, 3, 4, 5.
- Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- Source: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html

### 6. NCCL or Libfabric/EFA override reference (heuristic)
- Evidence: Dockerfile:5 (unverified_source_or_configuration)
- Tokens: NCCL_SOCKET_IFNAME
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Determine whether this setting is effective and why it exists. Review target-specific transport defaults and remove obsolete workarounds only in a controlled comparison.
- Validate: Capture initialization logs on every rank, provider/plugin selection and representative collective traces. Confirm application paths, including custom communication that bypasses NCCL.
- Guide: docs/10-validation-runbook.md, sections 2, 3, 4, 5.
- Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- Source: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html

### 7. Potential workload entry point (heuristic)
- Evidence: Dockerfile:6 (unverified_source_or_configuration)
- Tokens: launch declaration (command withheld)
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Identify the actual serving/training launch and its deployed overrides. Trace configuration precedence manually; scanning does not resolve imports or launch scripts.
- Validate: Freeze the exact command and configuration for baseline and target, then use that same path in the end-to-end matrix.
- Guide: docs/10-validation-runbook.md, sections 5, 6.

### 8. Custom or specialized GPU kernel signal (heuristic)
- Evidence: kernel.cu:1 (unverified_source_or_configuration)
- Tokens: CUDA source file
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Locate kernel owners and inspect wheel/fatbin targets, dispatch guards and build toolchain. Rebuild or select supported fallbacks only after checking semantics.
- Validate: Exercise every relevant kernel branch and shape on target hardware, compare outputs to a trusted reference, and capture compile/cold-start failures.
- Guide: docs/10-validation-runbook.md, sections 1, 5, 6.
- Source: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html

### 9. Custom or specialized GPU kernel signal (heuristic)
- Evidence: kernel.cu:2 (unverified_source_or_configuration)
- Tokens: specialized kernel or extension marker
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Locate kernel owners and inspect wheel/fatbin targets, dispatch guards and build toolchain. Rebuild or select supported fallbacks only after checking semantics.
- Validate: Exercise every relevant kernel branch and shape on target hardware, compare outputs to a trusted reference, and capture compile/cold-start failures.
- Guide: docs/10-validation-runbook.md, sections 1, 5, 6.
- Source: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html

### 10. Framework or CUDA dependency declaration (heuristic)
- Evidence: requirements.txt:2 (unverified_source_or_configuration)
- Tokens: torch, version token (constraint not resolved): 2.5.1
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Record the resolved package, wheel, CUDA runtime and driver versions. Choose a target-supported combination from the engine/vendor support matrix; do not upgrade from this scan alone.
- Validate: Rebuild in a clean approved image, import the pinned runtime on target hardware, and execute representative kernels including custom extensions.
- Guide: docs/10-validation-runbook.md, sections 1, 5.

### 11. Framework or CUDA dependency declaration (heuristic)
- Evidence: requirements.txt:3 (unverified_source_or_configuration)
- Tokens: vllm, version token (constraint not resolved): 0.6.4
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Record the resolved package, wheel, CUDA runtime and driver versions. Choose a target-supported combination from the engine/vendor support matrix; do not upgrade from this scan alone.
- Validate: Rebuild in a clean approved image, import the pinned runtime on target hardware, and execute representative kernels including custom extensions.
- Guide: docs/10-validation-runbook.md, sections 1, 5.

### 12. Custom or specialized GPU kernel signal (heuristic)
- Evidence: requirements.txt:4 (unverified_source_or_configuration)
- Tokens: specialized kernel or extension marker
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Locate kernel owners and inspect wheel/fatbin targets, dispatch guards and build toolchain. Rebuild or select supported fallbacks only after checking semantics.
- Validate: Exercise every relevant kernel branch and shape on target hardware, compare outputs to a trusted reference, and capture compile/cold-start failures.
- Guide: docs/10-validation-runbook.md, sections 1, 5, 6.
- Source: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html

### 13. Framework or CUDA dependency declaration (heuristic)
- Evidence: requirements.txt:4 (unverified_source_or_configuration)
- Tokens: flash-attn, version token (constraint not resolved): 2.7.0
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Record the resolved package, wheel, CUDA runtime and driver versions. Choose a target-supported combination from the engine/vendor support matrix; do not upgrade from this scan alone.
- Validate: Rebuild in a clean approved image, import the pinned runtime on target hardware, and execute representative kernels including custom extensions.
- Guide: docs/10-validation-runbook.md, sections 1, 5.

### 14. Distributed training or state-sharding signal (heuristic)
- Evidence: training.yaml:2 (unverified_source_or_configuration)
- Tokens: distributed training/sharding marker
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Preserve optimizer/checkpoint semantics while reviewing FSDP/ZeRO/DDP groups, communication frequency and CPU offload for the new topology.
- Validate: Compare gradients/loss, effective batch, max-rank memory, step time, checkpoint restore and recovery under the target layout.
- Guide: docs/10-validation-runbook.md, sections 3, 4, 5, 6.

### 15. Potential host/device transfer or CPU offload (heuristic)
- Evidence: training.yaml:4 (unverified_source_or_configuration)
- Tokens: host/device copy or offload marker
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Check whether the transfer is on the critical path, intended checkpoint/output work, or offload. Review batching, placement and overlap before changing it.
- Validate: Profile representative execution for H2D/D2H copies, synchronization and CPU gaps; compare quality and unprofiled end-to-end results after any change.
- Guide: docs/10-validation-runbook.md, sections 2, 5, 6.

### 16. Host preprocessing or data-loader signal (heuristic)
- Evidence: training.yaml:6 (unverified_source_or_configuration)
- Tokens: host pipeline marker
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Review CPU threads, affinity, NUMA placement, pinned buffers and preprocessing throughput against the target host; do not assume more GPU capacity removes a host bottleneck.
- Validate: Measure host-stage latency and GPU idle gaps under representative input, concurrency and storage conditions.
- Guide: docs/10-validation-runbook.md, sections 1, 2, 5, 6.

### 17. Host preprocessing or data-loader signal (heuristic)
- Evidence: training.yaml:7 (unverified_source_or_configuration)
- Tokens: host pipeline marker
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Review CPU threads, affinity, NUMA placement, pinned buffers and preprocessing throughput against the target host; do not assume more GPU capacity removes a host bottleneck.
- Validate: Measure host-stage latency and GPU idle gaps under representative input, concurrency and storage conditions.
- Guide: docs/10-validation-runbook.md, sections 1, 2, 5, 6.

### 18. Custom or specialized GPU kernel signal (heuristic)
- Evidence: workload.py:3 (unverified_source_or_configuration)
- Tokens: specialized kernel or extension marker
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Locate kernel owners and inspect wheel/fatbin targets, dispatch guards and build toolchain. Rebuild or select supported fallbacks only after checking semantics.
- Validate: Exercise every relevant kernel branch and shape on target hardware, compare outputs to a trusted reference, and capture compile/cold-start failures.
- Guide: docs/10-validation-runbook.md, sections 1, 5, 6.
- Source: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html

### 19. Fixed parallelism or GPU-count configuration (heuristic)
- Evidence: workload.py:6 (unverified_source_or_configuration)
- Tokens: tensor_parallel_size: 8
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Reconcile local GPU count, global world size, TP/PP/DP and engine partition rules. A global group may span nodes; do not compare it with a node count as an automatic error.
- Validate: Test the proposed rank layout, shard divisibility, per-rank peak memory and actual collectives with the exact workload.
- Guide: docs/10-validation-runbook.md, sections 1, 3, 4, 6.

### 20. Fixed parallelism or GPU-count configuration (heuristic)
- Evidence: workload.py:7 (unverified_source_or_configuration)
- Tokens: pipeline_parallel_size: 2
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Reconcile local GPU count, global world size, TP/PP/DP and engine partition rules. A global group may span nodes; do not compare it with a node count as an automatic error.
- Validate: Test the proposed rank layout, shard divisibility, per-rank peak memory and actual collectives with the exact workload.
- Guide: docs/10-validation-runbook.md, sections 1, 3, 4, 6.

### 21. Fixed parallelism or GPU-count configuration (heuristic)
- Evidence: workload.py:8 (unverified_source_or_configuration)
- Tokens: world_size: 16
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Reconcile local GPU count, global world size, TP/PP/DP and engine partition rules. A global group may span nodes; do not compare it with a node count as an automatic error.
- Validate: Test the proposed rank layout, shard divisibility, per-rank peak memory and actual collectives with the exact workload.
- Guide: docs/10-validation-runbook.md, sections 1, 3, 4, 6.

### 22. Precision or quantization selection (heuristic)
- Evidence: workload.py:9 (unverified_source_or_configuration)
- Tokens: bf16
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Verify engine/kernel support for the exact precision and model format on each GPU. Keep quality tolerances fixed; this scanner makes no datatype support verdict.
- Validate: Compare task quality and numerical behavior with the agreed reference, then measure memory, latency and goodput for each proposed precision.
- Guide: docs/10-validation-runbook.md, sections 5, 6.

### 23. Precision or quantization selection (heuristic)
- Evidence: workload.py:10 (unverified_source_or_configuration)
- Tokens: fp8
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Verify engine/kernel support for the exact precision and model format on each GPU. Keep quality tolerances fixed; this scanner makes no datatype support verdict.
- Validate: Compare task quality and numerical behavior with the agreed reference, then measure memory, latency and goodput for each proposed precision.
- Guide: docs/10-validation-runbook.md, sections 5, 6.

### 24. Fixed GPU selection or device index (heuristic)
- Evidence: workload.py:14 (unverified_source_or_configuration)
- Tokens: fixed CUDA ordinal: 7
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Replace accidental ordinal assumptions with verified local-rank mapping or an explicit deployment profile. Keep deliberate single-device choices only where intended.
- Validate: Launch with the intended visible-device set and rank count; verify every rank's GPU identity and per-device peak memory.
- Guide: docs/10-validation-runbook.md, sections 1, 5, 6.

### 25. Potential host/device transfer or CPU offload (heuristic)
- Evidence: workload.py:15 (unverified_source_or_configuration)
- Tokens: host/device copy or offload marker
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Check whether the transfer is on the critical path, intended checkpoint/output work, or offload. Review batching, placement and overlap before changing it.
- Validate: Profile representative execution for H2D/D2H copies, synchronization and CPU gaps; compare quality and unprofiled end-to-end results after any change.
- Guide: docs/10-validation-runbook.md, sections 2, 5, 6.

### 26. Potential workload entry point (heuristic)
- Evidence: workload.py:19 (unverified_source_or_configuration)
- Tokens: launch declaration (command withheld)
- Meaning: Text signal only; reachability, precedence and effective runtime configuration are unverified.
- Review/change: Identify the actual serving/training launch and its deployed overrides. Trace configuration precedence manually; scanning does not resolve imports or launch scripts.
- Validate: Freeze the exact command and configuration for baseline and target, then use that same path in the end-to-end matrix.
- Guide: docs/10-validation-runbook.md, sections 5, 6.

## What remains unknown

None asserted from text alone. A flag or declared version is not a resolved artifact or an executed path; an empty list is not evidence of compatibility.
- Installed driver/runtime, resolved dependencies, image digest, external or generated configuration, branch reachability and built cubin/PTX are unknown.
- Cannot deduce effective NCCL/P2P/EFA transport, fallback, overlap or custom communication from static text.
- Cannot determine per-rank memory fit, numerical quality, throughput, latency, cost or certify portability without target-hardware tests.

## Target-hardware validation matrix (not run)
- software_and_kernels — runbook sections 1, 5: Exact image, driver/runtime, host ISA, GPU compute capability and custom-kernel correctness on each source/target profile.
- memory_and_layout — runbook sections 1, 6: Peak allocated/reserved memory on every rank at the largest supported shapes, context, concurrency or training batch; valid TP/PP/DP partitioning.
- host_and_peer_paths — runbook sections 2, 5: H2D/D2H and peer copies for relevant size/pair/NUMA placements, then application traces showing effective transfer paths.
- collectives_and_network — runbook sections 3, 4, 5: Actual rank layout and message sizes; NCCL/provider logs on every rank; intended plugin/EFA selection and custom non-NCCL paths.
- end_to_end_quality_slo — runbook sections 6, 7: Predeclared quality/SLO acceptance, representative load/tails, warm/cold starts, independent repetitions, goodput and cost per qualifying result.
- recovery_and_rollback — runbook sections 6, 7: Drain/failure/restart, checkpoint restore where relevant, replacement capacity, and a verified rollback profile.

## Method and privacy limits
- Lexical patterns, not AST/data-flow analysis; inline comments, quoted strings, examples and dead code can match.
- Full-line comments are skipped. Test/example paths are labelled, not excluded. Aliases, computed values, split-line assignments and unsupported formats may be missed.
- No requirements includes, imports, submodule metadata, symlinks, hardlinks, container images or external paths are resolved. Materialized nested files may be scanned. No binary introspection or vulnerability diagnosis.
- Bounded UTF-8 allowlist scan with a documented .gitignore subset; unsupported/unreadable ignore policy skips its whole subtree.
- File extensions and filenames can hide relevant code; no findings never means no migration work.
- Allowlisted technical tokens and sanitized relative paths only. Paths and dependency versions can still be confidential; review locally before sharing.
