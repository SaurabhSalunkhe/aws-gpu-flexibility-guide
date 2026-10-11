# 08 · Estimator reference

[Guide home](../README.md) · Version 0.1.0

The estimator is a dependency-free **arithmetic screen**. It never contacts AWS, downloads weights, allocates GPU memory, changes a deployment, or predicts performance. Use Python 3.10+ from the repository root.

```bash
python3 -m gpu_fit inference --help
python3 -m gpu_fit training --help
python3 -m unittest discover -s tests -v
```

## Shared input conventions

| Option | Meaning | Default / validation |
|---|---|---|
| `--parameters` | Total parameter count, not billions; `8e9` is accepted | Required positive whole number, at most 1e18 |
| `--gpu-memory-gib` | Capacity of one GPU in binary GiB | Required finite positive number; use an observed device value |
| `--tp`, `--pp` | Tensor ranks and pipeline stages | Positive integers; each defaults to 1 |
| `--headroom-fraction` | Portion of each device reserved from the budget | 0.15; finite, at least 0 and less than 1 |
| `--runtime-overhead-gib` | Explicit extra allocation allowance on each rank | 0; finite and nonnegative |
| `--weight-imbalance-factor` | Multiply the model-state term to screen an uneven busiest rank | 1; finite and at least 1 |
| `--json` | Return full inputs, components, warnings, and result | Off |

One GiB is 2³⁰ bytes. Parameter counts use decimal notation: 8B means 8,000,000,000 parameters. Source GB/GiB labels are not automatically normalized by the estimator. The [catalog](02-instance-catalog.md) deliberately retains conflicting source labels and is not used as an automatic capacity lookup.

The usable per-device budget is `gpu_memory_gib × (1 − headroom_fraction)`. The runtime allowance adds to demand; the reserve reduces capacity. Keep them distinct and avoid counting the same expected allocation twice. A 15% reserve is only the default scenario assumption.

## Inference inputs and formulas

Required: `--layers`, `--query-heads`, `--kv-heads`, `--head-dim`, `--context-tokens`, and `--concurrency`, all positive integers.

- Context means **resident prompt plus generated tokens per sequence**, not prompt length alone or a model's advertised context limit.
- Concurrency means resident sequences **per model replica**, not fleet-wide requests, tokens per second, or only the currently scheduled microbatch.
- `--weight-bits` accepts 4, 8, 16, or 32; default 16. It represents storage bits, not a format or a kernel selection.
- `--kv-bits` accepts 8, 16, or 32; default 16. It is independent of weight precision.
- `--weight-overhead-fraction` defaults to 0 and must be nonnegative. It multiplies the ideal weight payload to allow for metadata and unquantized tensors; replace it using checkpoint/runtime evidence.
- `--kv-layout sharded` divides KV across TP ranks. `replicated` conservatively puts every KV head for a pipeline stage on each of that stage's TP ranks. Partial replication is not modeled.

```text
weight_bytes = parameters × weight_bits / 8 × (1 + weight_overhead_fraction)
KV_bytes = 2 × layers × KV_heads × head_dim × KV_bits / 8
           × context_tokens × concurrency

weights_per_rank = weight_bytes / (TP × PP) × weight_imbalance_factor
KV_per_rank = KV_bytes / PP / (TP if sharded else 1)
demand_per_rank = weights_per_rank + KV_per_rank + runtime_allowance
```

The formulas assume uniform dense decoder MHA/GQA/MQA, equal key/value dimensions, equal layer-count pipeline stages, and a supported balanced tensor layout. The CLI validates query heads divisible by KV heads, query heads divisible by TP, and layers divisible by PP. Sharded KV additionally requires KV heads divisible by TP. These are **this screen's constraints**, not a claim that every other layout is impossible. Some engines support uneven stages or replicated KV; inspect their actual placement.

The imbalance factor affects weights only in inference. It does not repair a wrong KV partition or support uneven layer stages. Compute those separately rather than hiding them in a generous reserve.

### What inference mode leaves out

Prefill activation peaks, paged-cache block rounding and metadata, graph capture, attention/GEMM workspace, NCCL buffers, allocator fragmentation, checkpoint conversion copies, unsharded embeddings/heads, draft models, and adapter memory are not computed. Your allowances and measurements must cover them.

Do not apply this formula unchanged to MoE, MLA, heterogeneous/hybrid/state-space layers, sliding-window allocation policies, encoder-decoder cross-attention, multimodal caches, offload, or speculative decoding. It also does not verify model licensing, quantization quality, backend/driver support, topology, latency, or capacity.

## Training inputs and formulas

Training mode covers **full-parameter persistent-state accounting**, plus an explicit per-rank activation allowance. It does not predict the peak training step or validate architecture divisibility.

| Option | Default | Meaning |
|---|---:|---|
| `--weight-bits` | 16 | Parameter storage, 16 or 32 |
| `--gradient-bits` | 16 | Gradient storage, 16 or 32 |
| `--master-weight-bits` | 32 | Master weights, 0, 16, or 32 |
| `--optimizer-bits-per-parameter` | 64 | Sum of optimizer states: two FP32 Adam moments total 64 bits; nonnegative integer |
| `--activation-gib` | Required | Busiest-rank activation allowance, finite/nonnegative; not divided by parallelism |
| `--dp` | 1 | Size of the data-parallel state-sharding group |
| `--zero-stage` | 0 | ZeRO stage 0, 1, 2, or 3 |

The default selected state is 16 bytes per parameter. Set `--gradient-bits 32` for the 18-byte variant. Framework implementations can differ; neither setting is a universal optimizer recommendation.

All model-state components first divide by TP × PP and multiply by the imbalance factor. ZeRO then applies these DP divisors:

| Stage | Weights | Gradients | Master weights + optimizer states |
|---|---:|---:|---:|
| 0 | 1 | 1 | 1 |
| 1 | 1 | 1 | DP |
| 2 | 1 | DP | DP |
| 3 | DP | DP | DP |

Activations and runtime allowance remain per rank. Ordinary DP with stage 0 does not reduce state memory. The GPU count is TP × PP × DP, with DP representing the actual sharding group, not an arbitrary fleet size.

ZeRO-3/FSDP parameter materialization, prefetch, communication buckets, optimizer workspaces, checkpoint save/load, padding and largest unsharded/wrapped layers are omitted. The result remains a lower-bound screen even when the user supplies an activation estimate. LoRA/QLoRA and CPU/NVMe offload need separate accounting.

## Output and exit status

- `within_budget_screen`: estimated terms are less than or equal to the stated per-GPU budget. Missing components or unsupported execution can still make the model fail.
- `exceeds_budget_screen`: the supplied configuration exceeds its budget under the given assumptions. Change the assumptions only when the workload or implementation actually changes.
- Exit status **0** means a valid calculation, including an exceeded budget; it is not a deployment-ready flag.
- Exit status **2** means invalid or missing CLI inputs. NaN, infinity, invalid precision, negative values, and unsupported partition arithmetic are rejected.

JSON includes `schema_version`, `mode`, `decision`, `inputs`, `gpu_count`, per-GPU byte components, GiB totals/budget/margin, and warnings. Numerical outputs are deterministic for the same inputs. No model-fit Boolean is provided.

## Verification scope

Tests verify arithmetic, boundary decisions, unit conversion, GQA, quantization separation, TP/PP/ZeRO divisors, CLI errors, source-catalog structure, local links, and executable examples in the docs. CI is configured for Python 3.10, 3.12, and 3.13. See the actual workflow status for which versions have run; local testing alone does not establish that CI passed.

---

Next: [Inspect the whole workload](09-workload-portability.md)
