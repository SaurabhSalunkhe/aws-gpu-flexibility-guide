# 09 · Workload portability: follow the whole critical path

[Guide home](../README.md) · Reviewed 2026-10-11

**Enough memory is the first gate. The slowest important part of the real execution path determines whether an alternative is useful.** A migration can pass a model-load test and still lose on host preprocessing, unsupported kernels, collective latency, storage, queueing, or recovery.

This chapter is a diagnostic framework, not a measured ranking of AWS GPUs. Use the [validation runbook](10-validation-runbook.md) to test each hypothesis.

## Four different kinds of evidence

| Evidence | What it can establish | What it cannot establish |
|---|---|---|
| Vendor specification | Hardware capability and advertised configuration | The path your application actually uses |
| Published measured study | Results under that study's stated setup | Your model's performance on an untested instance |
| Offline arithmetic or analytical simulation | Sensitivity to explicit assumptions; likely constraints worth testing | GPU emulation, a benchmark, or calibrated production latency |
| Your controlled target-hardware experiment | Behavior of the exact tested stack and workload | Every other model, precision, load level, or future release |

The [published-evidence chapter](11-published-evidence.md) preserves study conditions and corrections; the [CPU analytical lab](../SIMULATION.md) makes simulated assumptions inspectable and reproducible.

Keep these labels attached to tables and charts when sharing them. Do not blend published measurements with simulated points into an unlabeled “GPU comparison.”

## Hardware capability is not the application data path

Think of a five-step chain:

1. **Physical path:** which GPU/CPU/NIC pairs have PCIe, NVLink/NVSwitch, or an inter-node network path?
2. **Supported access:** can the current driver/runtime use peer memory or the required RDMA feature on those exact devices?
3. **Selected backend:** did the application load the intended attention kernels, NCCL transport, and network plugin?
4. **Observed traffic:** do traces show peer/collective traffic where expected, or host copies, synchronization, and idle gaps?
5. **Useful outcome:** does the complete request or training step meet its quality, latency, throughput, cost, and recovery requirements?

An NVLink-capable system does not prove that a model is sharded. A P2P-capable GPU pair does not prove that a program has enabled peer access. An EFA interface does not prove the job selected an EFA transport. Topology, test transfers, collective logs, and application profiling answer different questions. [NVIDIA NCCL troubleshooting](https://github.com/NVIDIA/nccl/blob/master/docs/userguide/source/troubleshooting/performance_and_tuning.rst), [AWS OFI NCCL](https://github.com/aws/aws-ofi-nccl)

## Bottleneck map

The observations below are hypotheses to investigate, not automatic diagnoses. Several constraints can interact.

| Suspected constraint | What you might observe | Evidence to collect | A controlled comparison |
|---|---|---|---|
| Compute or kernel efficiency | Long kernels; adding link bandwidth changes little | Kernel names, precision, dimensions, achieved compute, fallback messages | Same precision and workload on target-specific supported kernels |
| Device-memory bandwidth | Data movement dominates certain phases | Phase timing, bytes moved, achieved bandwidth, batch dependence | Decode and prefill separately; vary batch without changing quality |
| GPU-to-GPU communication | Collectives dominate; more TP ranks add little speed | Per-message latency/bandwidth, rank paths, overlap, stragglers | TP vs PP vs replicas; same total GPUs and same workload contract |
| GPU-to-host or CPU locality | Copy gaps, CPU pressure, or asymmetric paths | CPU/NUMA affinity, H2D/D2H timing, pinned-buffer behavior | Correct placement and batching, one variable at a time |
| Inter-node transport | Multi-node scaling falls sharply | EFA eligibility, provider/plugin logs, collective traces | One node vs homogeneous two-node placement |
| CPU/service pipeline | GPUs wait while tokenization, preprocessing, or serialization runs | CPU timeline, queues, thread usage, client saturation | More appropriate CPU workers or preprocessing strategy |
| Storage/checkpoint loading | Long cold starts or training stalls | Read throughput, download/deserialization time, cache state | Cold vs warm load; checkpoint/storage layouts |
| Scheduler or queueing | Good isolated latency but poor p99 under load | Arrival trace, backlog, admission/rejection, preemptions | Fixed offered-load sweep with declared limits |
| Reliability/recovery | Fast steady state but costly restarts | Failure/drain behavior, checkpoint restore, capacity replacement | Controlled failure and rollback exercises |

For transfers and CPU locality, NVIDIA recommends minimizing unnecessary host/device movement, batching small transfers, and considering pinned memory and NUMA placement. These are options to qualify, not a reason to pin unlimited host RAM or blindly alter affinity. [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

## Test phases separately, then together

- **Autoregressive serving:** prefill, decode, queueing, and streaming delivery can have different limits. A faster prefill kernel need not improve inter-token latency under a mixed load.
- **Batch inference or embedding:** compare completed records per second at a fixed quality/output contract; token-generation metrics may not describe the task.
- **Training:** forward, backward, gradient synchronization, optimizer updates, data loading, and checkpoints all matter. Hold effective batch and convergence criteria constant when comparing useful progress.
- **Multimodal/image/video:** include decoding, preprocessing, encoders, resolution, frame count, and postprocessing in the contract. A language-only memory formula is incomplete.

A profile should include CPU and GPU timelines, launch gaps, copies, and synchronization. Profile a representative interval, then repeat performance measurements without profiling to quantify instrument overhead. [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

## Why a faster component can leave the service unchanged

Suppose an illustrative request spends 40% of its time in a component that becomes twice as fast, while everything else is unchanged. The serial arithmetic speedup is `1 / (0.60 + 0.40 / 2) = 1.25×`, not 2×. This is a deliberately simple sensitivity example. Real overlapping execution and queueing need a richer model and measurements.

Similarly, a device with more advertised compute can expose a CPU, memory, or network bottleneck that was previously hidden. Bigger batches can improve device efficiency while making a latency target harder to meet. A larger TP group can lower each rank's compute while increasing communication and scheduling costs. The correct experiment holds the user-visible contract fixed and records where time moved.

## Migration-specific questions

### G7e to H100/H200

Check the destination memory envelope first; single-GPU H100 has less nominal VRAM than G7e. Then check kernels for Hopper, per-phase performance, and whether the new GPU count/layout changes the communication pattern. A single-GPU `p5.4xlarge` and eight-GPU `p5.48xlarge` are materially different alternatives. [AWS P5](https://aws.amazon.com/ec2/instance-types/p5/)

### G7e to B200/B300

Both product lines use Blackwell-generation technology, but RTX PRO 6000, B200, and B300 are distinct CUDA targets. Validate the actual binary/backend support and precision path. Higher advertised low-precision throughput is useful only when supported kernels and acceptable quality expose it. [NVIDIA CUDA GPU targets](https://developer.nvidia.com/cuda/gpus)

### One node to multiple nodes

Expect a new communication path, placement problem, and failure boundary. Verify exact EFA/RDMA eligibility and transport selection. Do not assume a node's summed NIC bandwidth is available to every rank or that an all-reduce test predicts MoE all-to-all performance. Use the same collectives and message sizes that the application generates. [AWS EFA and NCCL setup](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html)

### Different CPU or storage platform

Record host architecture, vCPUs, NUMA placement, RAM, storage and network configuration alongside GPU details. Tokenization, checkpoint conversion, data loading, and native extensions remain part of the migration. Grace Arm systems require a separate image/dependency check; larger x86 GPU instances can also have different host characteristics. [AWS accelerated instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)

## Compare useful work, not just utilization

For serving, compare quality-qualified, SLO-passing completed requests and output tokens at a stable offered load. For training, compare cost and time to the same validated target, with effective batch and failure recovery disclosed. High utilization can accompany a slow or communication-bound workload; low utilization alone does not identify why.

At a fixed workload, define strong-scaling efficiency as `speedup / GPU-count multiplier`. When one-GPU execution is impossible, use the smallest valid baseline and state it. If the workload grows with GPU count, label the comparison weak scaling instead. Never combine the two into one unexplained “efficiency” number.

---

Next: [Run the hardware and end-to-end checks](10-validation-runbook.md)
