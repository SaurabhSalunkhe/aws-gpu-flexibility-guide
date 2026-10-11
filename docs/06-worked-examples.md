# 06 · Worked examples you can reproduce

[Guide home](../README.md) · Reviewed 2026-10-11

**Every number below is arithmetic, not a GPU benchmark or a certified deployment.** Capacities and overheads are explicitly hypothetical inputs. Replace them with actual per-device capacity and measured runtime behavior before a deployment decision.

## Common inputs

| Profile | Rounded parameters | Layers | Query heads | KV heads | Head dimension |
|---|---:|---:|---:|---:|---:|
| 8B GQA | 8,000,000,000 | 32 | 32 | 8 | 128 |
| 70B GQA | 70,000,000,000 | 80 | 64 | 8 | 128 |

The architecture shapes follow NVIDIA-published [8B](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8/blob/main/config.json) and [70B](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8/blob/main/config.json) Llama 3.1 configurations. Parameter counts here are deliberately rounded, not the exact checkpoint tensor count. Those FP8 checkpoint configuration files are architecture references; they do not make the BF16 examples actual deployments of those files.

All examples use binary GiB, a 15% per-device reserve, and 2 GiB of additional runtime allowance unless stated otherwise. The allowance is deliberately simple, not a recommended production value. It does not establish that omitted memory is covered.

### The raw quantities

| Quantity | 8B example | 70B example |
|---|---:|---:|
| BF16/FP16 ideal weights | 14.90 GiB | 130.39 GiB |
| Ideal 8-bit weights, before metadata | 7.45 GiB | 65.19 GiB |
| Ideal 4-bit weights, before metadata | 3.73 GiB | 32.60 GiB |
| BF16 KV per cached token | 128 KiB | 320 KiB |
| BF16 KV for one 8,192-token sequence | 1.00 GiB | 2.50 GiB |
| BF16 KV for eight 8,192-token sequences | 8.00 GiB | 20.00 GiB |

These follow the [weight and KV formulas](03-model-memory.md). For example, 70 billion parameters × 2 bytes is 140 billion bytes, or **130.39 GiB**. “140 GB of weights” and “140 GiB of weights” are different quantities.

## A. 8B inference: memory includes more than weights

Eight concurrent resident sequences, each with 8,192 prompt-plus-generated tokens; hypothetical 80 GiB device.

<!-- smoke-test: inference_8b -->
```bash
python3 -m gpu_fit inference \
  --parameters 8e9 --gpu-memory-gib 80 \
  --layers 32 --query-heads 32 --kv-heads 8 --head-dim 128 \
  --context-tokens 8192 --concurrency 8 \
  --runtime-overhead-gib 2
```

14.90 weights + 8.00 KV + 2.00 runtime = **24.90 GiB**, below the 68.00 GiB arithmetic budget. This is a candidate for a real load test. It says nothing about the tokens/second or latency of G7e versus H100.

Now increase resident load to sixteen 32,768-token sequences:

<!-- smoke-test: inference_8b_long -->
```bash
python3 -m gpu_fit inference \
  --parameters 8e9 --gpu-memory-gib 80 \
  --layers 32 --query-heads 32 --kv-heads 8 --head-dim 128 \
  --context-tokens 32768 --concurrency 16 \
  --runtime-overhead-gib 2
```

KV becomes 64 GiB. Total becomes **80.90 GiB**, exceeding the same 68.00 GiB budget even though the weights are unchanged. Context limits, concurrency, and KV policy are part of the instance decision.

## B. 70B inference: two GPUs do not establish fit

Use BF16 weights and KV, eight resident 8,192-token sequences, and ideal balanced TP=2. Each hypothetical GPU has 80 GiB, with a 68 GiB budget after reserve.

<!-- smoke-test: inference_70b_tp2 -->
```bash
python3 -m gpu_fit inference \
  --parameters 70e9 --gpu-memory-gib 80 \
  --layers 80 --query-heads 64 --kv-heads 8 --head-dim 128 \
  --context-tokens 8192 --concurrency 8 --tp 2 \
  --runtime-overhead-gib 2
```

Per GPU: 65.19 weights + 10.00 KV + 2.00 runtime = **77.19 GiB**, exceeding the budget. Adding the two nominal device capacities obscures this per-rank constraint.

Try TP=4 under the same assumptions:

<!-- smoke-test: inference_70b_tp4 -->
```bash
python3 -m gpu_fit inference \
  --parameters 70e9 --gpu-memory-gib 80 \
  --layers 80 --query-heads 64 --kv-heads 8 --head-dim 128 \
  --context-tokens 8192 --concurrency 8 --tp 4 \
  --runtime-overhead-gib 2
```

Per GPU: 32.60 weights + 5.00 KV + 2.00 runtime = **39.60 GiB**. This clears the arithmetic screen. Whether it is useful depends on real shard balance, collective traffic, kernels, and latency. The same TP setting on G7e PCIe and a P-family NVSwitch node is not a performance-equivalent experiment.

## C. Quantization changes weights, not KV automatically

Suppose an approved 4-bit format has a **hypothetical 12% weight-payload overhead** for metadata and higher-precision tensors. Keep BF16 KV and the same 70B resident-token load on one hypothetical 80 GiB GPU.

<!-- smoke-test: inference_70b_4bit -->
```bash
python3 -m gpu_fit inference \
  --parameters 70e9 --gpu-memory-gib 80 --weight-bits 4 \
  --weight-overhead-fraction 0.12 \
  --layers 80 --query-heads 64 --kv-heads 8 --head-dim 128 \
  --context-tokens 8192 --concurrency 8 --kv-bits 16 \
  --runtime-overhead-gib 2
```

36.51 adjusted weights + 20.00 KV + 2.00 runtime = **58.51 GiB**, below the 68.00 GiB budget. The 12% value is not a claim about any checkpoint format. Inspect actual storage and loading allocations, then validate quality and target-GPU kernel support. Four bits in an arithmetic formula do not identify INT4, FP4, NF4, or a serving backend.

## D. Larger per-device memory can remove sharding, with a narrow margin

Reduce the 70B example to one resident 8,192-token sequence and try a **hypothetical 160 GiB** device. This is a neutral arithmetic capacity, not an EC2 product specification.

<!-- smoke-test: inference_70b_single160 -->
```bash
python3 -m gpu_fit inference \
  --parameters 70e9 --gpu-memory-gib 160 \
  --layers 80 --query-heads 64 --kv-heads 8 --head-dim 128 \
  --context-tokens 8192 --concurrency 1 \
  --runtime-overhead-gib 2
```

The screen is **134.89 GiB** against 136.00 GiB after reserve: just 1.11 GiB of margin. An optimistic overhead assumption or an additional resident sequence can reverse the result. Use the [instance catalog's memory caveat](02-instance-catalog.md) before mapping this reasoning to H200/B200/B300.

## E. Full training has a different order of magnitude

Before activations, a chosen 16-byte state accounting gives **119.21 GiB for 8B** and **1,043.08 GiB for 70B**. The 18-byte variant with FP32 gradients gives **134.11 GiB** and **1,173.47 GiB**. Neither is a universal optimizer default.

For an 8B example, use four data-parallel ranks with ZeRO-3 and an invented 8 GiB activation allowance on each rank:

<!-- smoke-test: training_8b_zero3 -->
```bash
python3 -m gpu_fit training \
  --parameters 8e9 --gpu-memory-gib 80 \
  --dp 4 --zero-stage 3 --activation-gib 8 \
  --runtime-overhead-gib 2
```

Persistent state is 119.21 / 4 = 29.80 GiB per rank. With activations and runtime, the lower-bound screen is **39.80 GiB**. This excludes transient parameter all-gathers and other peaks; it is not a claim that the training job fits.

Change only gradient storage to FP32:

<!-- smoke-test: training_8b_zero3_fp32grad -->
```bash
python3 -m gpu_fit training \
  --parameters 8e9 --gpu-memory-gib 80 --gradient-bits 32 \
  --dp 4 --zero-stage 3 --activation-gib 8 \
  --runtime-overhead-gib 2
```

The screen rises to **43.53 GiB**. Production estimates must use the actual framework's state types and peak behavior; [DeepSpeed's memory guidance](https://deepspeed.readthedocs.io/en/latest/memory.html) includes largest-layer materialization and state assumptions beyond simple parameter-count division.

For a 70B example, TP=8 with an illustrative 10 GiB activation allowance per rank on hypothetical 160 GiB devices:

<!-- smoke-test: training_70b_tp8 -->
```bash
python3 -m gpu_fit training \
  --parameters 70e9 --gpu-memory-gib 160 \
  --tp 8 --activation-gib 10 --runtime-overhead-gib 2
```

The default 16-byte persistent state gives **142.39 GiB per rank**, above the 136 GiB budget. Architecture support, uneven state placement, activation scheduling, and any additional DP sharding need separate evaluation. A successful 70B inference configuration cannot be reused as a full-training memory budget.

---

Next: [Measure the actual workload](07-benchmarking.md)
