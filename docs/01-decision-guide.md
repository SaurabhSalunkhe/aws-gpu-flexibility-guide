# 01 · Choose an alternative by constraint

[Guide home](../README.md) · Reviewed 2026-10-11

**Keep several qualified workload profiles, rather than one interchangeable list of GPU names.** An alternative is usable when the exact model and runtime meet the same quality, context, latency, throughput, and operational requirements.

## A useful first shortlist

These are candidates to test, not a performance ranking or a price recommendation. Hardware premises come from [G7e](https://aws.amazon.com/ec2/instance-types/g7e/), [P5](https://aws.amazon.com/ec2/instance-types/p5/), and [P6](https://aws.amazon.com/ec2/instance-types/p6/) specifications; the selection logic is engineering judgment.

| Current constraint | Candidate path | First question to resolve |
|---|---|---|
| A small model fits on one GPU, but G7e is unavailable | Compare fitting G6/G6e, single-GPU P5, or another qualified G7e pool | Does the destination still have enough per-device memory at the required live-token load? |
| G7e single-GPU inference works; throughput is too low | More independent replicas, then a different GPU | Is per-request latency already acceptable, and can traffic be load-balanced? |
| Weights alone exceed one device | Larger per-GPU memory, accepted quantization, or supported TP/PP | Which change preserves quality and the application contract? |
| Weights load, but long prompts or concurrency OOM | More memory, KV-aware limits, or supported sharding | Is the peak KV cache, prefill/workspace, or graph capture? |
| A multi-GPU model is communication-bound on G7e | Compare eight-GPU P5/P5e/P5en/P6 nodes | Does a faster local GPU fabric improve the actual request mix enough? |
| H100 memory is the limiting factor | H200, then B200/B300 where the stack is qualified | Can larger per-device memory remove a shard or increase useful concurrency? |
| GPU idle gaps persist despite a fitting model | Inspect CPU/NUMA, preprocessing, copies, storage and scheduler | Which stage leaves the GPU waiting, and did the destination change that path? |
| New GPU peak specifications look much faster | Verify exact kernel, precision, shape and quality support | Does the real code exercise the advertised capability? |
| Full-parameter training is the task | Training-state budget with TP/PP and FSDP/ZeRO as supported | What are optimizer, activation, all-gather, and checkpoint peaks on the busiest rank? |
| One replica cannot fit in one node | Supported multi-node layout | Does the real EFA/NCCL path sustain the required collective traffic? |

The [catalog](02-instance-catalog.md) also includes A100 P4d/P4de as additional fallback candidates that require benchmarking. Avoid silently reducing context, changing quantization, or replacing a model to make a candidate pass.

## Ask these questions in order

### 1. What contract must the alternative preserve?

Write down the exact model/checkpoint and tokenizer revision, weight and KV precision, maximum prompt plus output tokens, resident concurrency, request mix, quality bar, and latency target. Distinguish inference, adapter fine-tuning, and full training. A name such as “70B at 4-bit” is not a complete contract.

If an existing repository is available, the [local codebase assessor](../CODEBASE_ASSESSMENT.md) can locate static signals and prepare review questions. Its output does not resolve the actual deployment or replace runtime tests.

### 2. Is the exact software stack supported?

Check GPU compute capability, host architecture, driver/CUDA/runtime, attention and GEMM kernels, quantization format, and parallelism support. Rebuild architecture-specific artifacts where needed. “Blackwell” alone does not establish portability from G7e to B200/B300. See the [software migration checks](05-migration-playbook.md).

### 3. What limits the complete workload?

Profile compute, device-memory traffic, peer transfers, collectives, CPU/preprocessing, storage, and queueing separately. The hardware path and the path selected by the runtime can differ. Use [whole-workload portability](09-workload-portability.md) to form hypotheses and the [validation runbook](10-validation-runbook.md) to test them.

### 4. What is the memory bottleneck?

- If weights exceed the budget, reducing context alone cannot solve it.
- If KV dominates, live token count and concurrency matter more than parameter count alone.
- If temporary peaks dominate, look at prefill, graphs, logits, communication buffers, and allocator behavior.
- If training, add optimizer state, gradients, activations, and transient parameter materialization.

Use the [memory model](03-model-memory.md) and [estimator](08-estimator-reference.md) to reject implausible configurations quickly. Only real execution can establish the remaining cases.

### 5. Which layout changes the right constraint?

Start with one GPU when it works. For additional aggregate serving capacity, compare replicas. To fit one replica, compare larger memory, accepted quantization, TP, or PP. Choose based on the actual communication fabric and supported architecture. Data-parallel replicas do not divide inference memory. [Parallelism guide](04-parallelism.md)

### 6. Does the candidate pass all acceptance gates?

1. Correctness and quality on representative tasks
2. Peak per-rank memory, including long-context and startup/capture paths
3. Tail latency and successful SLO-passing throughput at the required load
4. Real regional offering, quota, capacity mechanism, and replacement capacity
5. Cost over a stated measurement window, including idle reserve and retries
6. Startup, drain, failure recovery, and rollback

Record failures as well as successes. A candidate can be the preferred throughput option but an unsuitable low-latency fallback. A capacity mechanism can be available commercially while a particular account or AZ cannot provision the desired shape.

## The migration traps worth saying aloud

- **G7e → single H100 can reduce memory:** nominal 96 GB becomes 80 GB. A faster compute path cannot compensate for a memory overrun.
- **The size suffix does not encode GPU count:** G7e `.12xlarge` has two GPUs, while G6e `.12xlarge` has four.
- **More CPU/host RAM does not mean more VRAM:** some larger sizes retain one GPU.
- **B300 and GB300 are different deployment choices:** do not turn an UltraServer product mention into an ordinary instance identifier.
- **Offering is not inventory:** a regional/AZ support check is not a capacity reservation.

These details are sourced and expanded in the [instance catalog](02-instance-catalog.md).

---

Next: [Inspect the exact instance](02-instance-catalog.md)
