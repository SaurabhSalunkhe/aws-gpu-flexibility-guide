# 04 · Sharding and topology

[Guide home](../README.md) · Reviewed 2026-10-11

**Aggregate VRAM is not an automatically pooled allocation space.** Eight GPUs with 80 GiB each do not let an ordinary single-GPU process allocate a 640 GiB tensor. The runtime must explicitly place or shard tensors and move intermediate results. NVLink accelerates communication; it does not transparently rewrite the model into a distributed program.

| Strategy | What is distributed | What it is useful for | What still needs attention |
|---|---|---|---|
| Single GPU | Nothing | Simplest starting point when one replica fits and meets its SLO | Context/concurrency peaks and quality |
| Tensor parallelism (TP) | Operations and weight tensors within layers | Fitting one replica; sometimes reducing its execution latency | Frequent communication, divisible tensor/head dimensions, replicated state and backend support |
| Pipeline parallelism (PP) | Groups of layers | Fitting one replica across GPUs/nodes; sometimes useful on weaker interconnects | Unequal stage memory, embedding/head placement, pipeline bubbles and scheduling |
| Data parallelism (DP) | Independent replicas handling different data/requests | More aggregate serving capacity or training throughput | A dense replica still must fit; replication does not reduce its weight memory |
| Expert parallelism (EP) | MoE experts, with tokens routed to them | Distributing total expert weight storage and expert compute | All-to-all traffic, routing skew, replicated shared layers and expert load balancing |
| FSDP or ZeRO | Training parameters and/or gradients and optimizer state | Full-parameter training memory reduction | Transient all-gathers, activation peaks, communication and checkpoint save/load |

TP subdivides layer computation; sequence parallelism can additionally reduce certain activation replication. It is distinct from adding independent replicas. [PyTorch tensor parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)

For a dense serving model, start with one GPU if possible, then a supported intra-node TP/PP layout, then multi-node arrangements if necessary. vLLM documents PP as an option for uneven layer splits and for machines without NVLink, where it can avoid some TP communication overhead. Treat this as a configuration to benchmark, not a universal PP performance guarantee. [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

DP can be composed with TP: for example, two TP=4 replicas consume eight GPUs. For MoE, engine terminology can combine data-parallel attention with expert-parallel expert layers; do not apply the dense replica arithmetic mechanically. [vLLM data parallel deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)

An MoE's active parameters per token describe only part of its compute. Budget the **total resident expert parameters**, plus shared layers and routing buffers. EP requires a supported model/backend and an appropriate communication path; skewed routing can leave one rank as the bottleneck. [vLLM expert parallel deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)

### Per-rank sizing and divisibility

For a balanced supported TP × PP layout, `total_weights / (TP × PP)` is an optimistic starting point, not a final estimate. Add replicated tensors and inspect each stage. The largest rank determines feasibility.

- Query-head count must satisfy the selected engine's TP constraints. Current vLLM validation checks divisibility by TP; a 32-head example does not support arbitrary TP=3.
- KV sharding can stop scaling when TP exceeds KV-head count. Current vLLM duplicates KV heads so each rank retains at least one; dividing KV bytes by TP beyond this point understates memory unless a supported additional context-parallel method changes placement.
- KV-head division, head dimensions, quantization group sizes and expert layouts can impose further backend-specific restrictions.
- PP support is model-specific. Layer counts need not divide PP universally: some engines allow uneven splits. A simplified equal-stage estimator should reject or warn on an uneven case rather than claim the model is impossible to deploy.

These constraints must be checked against the pinned implementation. [vLLM model parallel validation and KV-head placement](https://docs.vllm.ai/en/stable/api/vllm/config/model/)

**Illustration:** the 70B BF16, eight-sequence, 8,192-token example has 130.39 GiB weights and 20 GiB total KV. Ideal TP=2 gives 65.19 GiB weights and 10 GiB KV per rank, already 75.19 GiB before per-rank overhead. It exceeds a 68 GiB planning budget for each 80 GiB GPU. “Two 80 GiB GPUs total 160 GiB” therefore does not establish fit. TP=4 lowers the ideal weight-plus-KV term to 37.60 GiB per rank, but runtime overhead and actual sharding still need measurement.

## Topology changes the answer

Capture `nvidia-smi topo -m`, device/NUMA placement, GPU count and interconnect details from the exact instance size. Separate intra-node NVLink/NVSwitch, intra-node PCIe, and inter-node networking. Optimize rank placement for the traffic pattern; do not assume every GPU pair has the same path.

EFA is an inter-instance network interface, not local VRAM and not a substitute for NVLink. Its supported features vary by exact instance. As reviewed, AWS lists G7e.8xlarge with RDMA read/write but without GPUDirect RDMA, while G7e.12xlarge, .24xlarge and .48xlarge list GPUDirect RDMA support. Recheck the live table for the chosen size. [AWS EFA support matrix](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)

For multi-node NVIDIA workloads, qualify the NVIDIA driver, CUDA, NCCL, EFA installer, Libfabric and AWS OFI NCCL integration together. Run the AWS-recommended NCCL tests across the intended nodes and message sizes, confirm the intended transport in logs, then test the application. An ordinary TCP fallback can make a memory-feasible layout unusable for latency. Placement, security group rules, NIC availability, host memory and container IPC settings also matter. [AWS EFA and NCCL setup](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html)

Prefer TP within a fast local GPU fabric when appropriate, and evaluate PP between nodes when one replica cannot fit in one node. EP's all-to-all pattern needs its own test. Avoid mixing unlike GPU types inside one synchronous TP group without explicit engine support and measurements; use heterogeneous capacity as separately qualified replica pools where practical.

AWS also limits EFA communication across AZs/VPCs and between the P4d/P4de/DL1 group and other instance types. Do not treat a heterogeneous fleet as one interchangeable EFA collective domain. [EFA limitations](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)

## A practical fallback design

Treat each tested model + precision + layout + instance family as a distinct serving pool. Keep a stable client API while routing only to pools that meet the same model, quality, and context contract. Two TP=4 replicas and one TP=8 replica have different scheduling and failure behavior even when both occupy eight GPUs. Keep one known-good profile for rollback.

On G7e, first compare independent replicas when the model fits on one GPU. When a replica needs multiple GPUs, benchmark supported TP and PP layouts on its PCIe P2P path. On an eight-GPU P5/P5e/P5en/P6 node, include local NVLink/NVSwitch layouts. Neither topology guarantees a particular SLO; measure the same request mix. Hardware basis: [AWS G7e](https://aws.amazon.com/ec2/instance-types/g7e/), [AWS P5](https://aws.amazon.com/ec2/instance-types/p5/), [AWS P6](https://aws.amazon.com/ec2/instance-types/p6/).

---

Next: [Qualify a migration](05-migration-playbook.md)
