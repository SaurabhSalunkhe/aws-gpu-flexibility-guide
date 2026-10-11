# 10 · Validation runbook: prove the path, then the outcome

[Guide home](../README.md) · Reviewed 2026-10-11

**This is a proposed target-hardware test plan, not a record of tests run for this repository.** The repo's offline tools need no GPU. Commands below require already-provisioned, authorized test machines, installed official tools, and a safe test window; GPU benchmarks consume resources and can disrupt shared workloads. Do not run them on production traffic without an approved plan.

Copy the [benchmark plan](../templates/benchmark-plan.md). Record immutable image/model/tool versions and a budget before a test. Keep logs in an approved private location; review them before sharing.

## 1. Capture inventory and physical topology

On each selected test machine, these commands inspect the host and devices:

```bash
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,pci.bus_id --format=csv
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
lscpu
numactl --hardware
```

`numactl` is optional and must already be installed. Save CPU affinity, NUMA relationships, GPU/NIC layout, and exact device counts. Read topology labels using the installed `nvidia-smi` version; an NVLink/PCIe path label is not a measured bandwidth. Peer read/write capability is also separate from application use. [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

Record memory in its reported unit, convert carefully, and check all ranks. The catalog's nominal values are not substitutes for the device readings.

## 2. Measure the relevant host and peer transfer paths

For an already-built, pinned version of NVIDIA's `nvbandwidth`, run from its build location:

```bash
./nvbandwidth --version
./nvbandwidth --list
./nvbandwidth -t host_to_device_memcpy_ce device_to_host_memcpy_ce -i 10
./nvbandwidth -t device_to_device_memcpy_write_ce -i 10
```

Choose supported tests from the installed version. Sweep transfer sizes and relevant GPU pairs; capture direction, affinity, test mode, correctness, and repeated samples. Copy-engine and kernel-copy tests exercise different paths. These measurements characterize copies on that system, not model throughput. A CUDA P2P sample can help establish capability, but a single sample result is not sufficient performance evidence. [NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth)

If a peer path is missing or unexpectedly slow, inspect topology and supported driver/platform configuration first. Do not bypass security controls or apply unverified driver patches to manufacture a favorable result.

## 3. Check collectives inside one node

Example for an authorized eight-GPU test node with a compatible, already-built `nccl-tests` binary:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,P2P,SHM \
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8 -w 10 -n 50
```

Change GPU count to the actual test group. The byte-range sweep includes small and larger messages; choose additional sizes from your application trace. Preserve correctness checks. Repeat across independent launches and test the collectives the application uses, such as all-gather, reduce-scatter, or all-to-all. MPI builds are required for multi-process/multi-node tests. [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)

Read latency for small messages as well as throughput for larger ones. `algbw` and `busbw` are different metrics; bus bandwidth uses a collective/rank-dependent normalization. Neither is an application token rate, and a single all-reduce number is not a full network score. [NCCL test metric definitions](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

## 4. Prove the multi-node transport

Before running a distributed job:

- Confirm supported instance combinations, EFA attachment, provider visibility, security-group configuration, and same-AZ placement under the documented AWS requirements.
- Record NVIDIA driver, CUDA, NCCL, EFA installer, Libfabric and AWS OFI NCCL versions.
- Inspect initialization/network logs for the intended plugin and transport on every rank. A successful job can still use an unintended fallback.
- Check NIC/rank placement and message-size behavior; repeat with the actual application layout.

Use the current [AWS EFA/NCCL procedure](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html) for the installed image. The [AWS OFI NCCL plugin](https://github.com/aws/aws-ofi-nccl) connects NCCL to Libfabric; having the library installed alone does not prove it was selected. The [EFA support matrix](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html) distinguishes EFA, RDMA, and GPUDirect capabilities.

For a preconfigured **two-node, eight-GPU-per-node** test allocation, an MPI-built `nccl-tests` launch shape is:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,P2P,SHM
mpirun -np 16 -N 8 --hostfile hosts --bind-to none \
  -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 -w 10 -n 50
```

This is a launch template, not an EFA setup script. Supply an authorized hostfile and the image-specific MPI/network settings. There is one GPU per process here; `-g 8` would change the intended rank count. Do not copy NIC names or provider flags from a different instance without verifying them. [NVIDIA multi-node test usage](https://github.com/NVIDIA/nccl-tests)

An application can also use custom collectives or transfer kernels that bypass NCCL. NCCL microbenchmarks and logs therefore characterize only the NCCL path; the application profile must establish what its own critical path uses.

Keep diagnostic environment changes scoped to the test. NCCL's debug controls help identify initialization, graph, and network choices; forcing algorithms or disabling transports changes what is measured and must be recorded. Remove temporary debug settings from the production profile after diagnosis. [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

## 5. Verify what the application actually does

Capture a short representative profile using the installed profiler and your real launch command. For example, with an existing Python workload script:

```bash
nsys profile --trace=cuda,nvtx,osrt --output application-profile \
  python3 your_workload.py
```

Replace the script with the actual launch entry point. Check profiling permissions and output location first. Inspect the timeline for host staging, repeated small copies, kernel gaps, synchronization, CPU preprocessing, and collective overlap. Do not infer the transport from GPU utilization alone. Profiling can perturb performance; measure unprofiled runs separately. [Nsight Systems CLI](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

Record these three statements independently:

1. The hardware supports the path: evidence = inventory/topology/capability
2. The runtime selected the path: evidence = initialization logs and traces
3. The path helped this workload: evidence = controlled end-to-end comparison

## 6. Run a representative end-to-end matrix

The following is a suggested experimental protocol, not a universal statistical guarantee. Set thresholds before running it and preserve the same workload/quality contract across candidates.

| Dimension | Required cases |
|---|---|
| Startup | Cold model fetch/load/compile/capture; warm restart; checkpoint restore where relevant |
| Prompt/output mix | Representative short, median, and long tails; maximum supported path; mixed lengths |
| Load | Isolated request, expected load, near-SLO limit, and bounded burst |
| Cache | Cache-cold and representative cache-warm; report hit rate and prefix reuse |
| Layout | Smallest valid replica; selected TP/PP; replica scaling; multi-node only where needed |
| Precision | Baseline and each proposed quantization; quality evaluated separately |
| Host path | Representative tokenization, preprocessing, postprocessing, and client load generation |
| Failure | Drain, unhealthy replica/rank, restart, replacement capacity, rollback |

Warm up until the measured phase is stable; record the criterion and excluded duration. Use at least three independent measured repetitions as an initial reproducibility check, more if variation is material. Preserve sample counts and run-to-run spread. A p99 from a tiny sample is weak evidence; use enough completed requests and duration to observe tails at the intended load, and disclose uncertainty. Do not average per-run percentiles and label the result a global percentile.

For serving, report P50/P95/P99 client-observed TTFT, inter-token latency, end-to-end latency, and throughput alongside errors and queue behavior. Per-request TPOT can be defined as `(completion time − first-token time) / (output token count − 1)` for outputs longer than one token; it is an average, not the distribution of inter-token gaps. Document the tokenizer and streaming instrumentation because one response chunk may contain multiple tokens.

For offline inference, use completed records or valid outputs per second where that describes the task better. For training, report effective batch, validated quality/loss progress, step-time distributions, checkpoints, and time/cost to a fixed target.

## 7. Decide on useful work, scaling, cost, and reliability

- **Goodput:** completed requests satisfying the selected quality and latency bar per second; disclose the definition and exclude failed/incomplete work.
- **Strong-scaling efficiency:** speedup divided by the GPU-count multiplier for the same workload, measured relative to a stated feasible baseline.
- **Cost per useful result:** total service cost divided by successful qualifying results over the same window. Include idle reserve, retries, loading/checkpoint time, and required surrounding infrastructure.
- **Recovery:** report detection, restart/restore, replacement-capacity delay, and time to regain the target service level.

Use the [benchmark protocol](07-benchmarking.md) and [results template](../templates/benchmark-results.csv). A candidate qualifies only for the tested boundaries. Preserve a known-good profile and a concrete rollback trigger before promotion.

## What this repository has actually validated

Offline tests execute the estimator examples and analytical simulations, check their formulas and input errors, and verify local documentation links. They do **not** execute the GPU commands above, validate EFA in an AWS account, test a checkpoint on hardware, or establish a performance/cost winner. Those remain target-hardware work.

---

Next: [Read the published evidence](11-published-evidence.md)
