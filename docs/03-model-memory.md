# 03 · Model memory: inference and training

[Guide home](../README.md) · Reviewed 2026-10-11

## Start with the workload contract

Record these inputs before comparing instances:

- Inference, full-parameter training, or adapter fine-tuning
- Model ID and immutable revision; architecture; total parameters, including all MoE experts; exact checkpoint tensor dtypes and quantization metadata
- Weight precision, activation/compute precision and KV-cache precision separately
- Maximum input plus generated tokens per request; distribution of real input/output lengths; concurrent sequences; request arrival rate and burstiness
- Attention type and per-layer configuration: MHA, GQA, MQA, MLA, sliding-window, hybrid/state-space or multimodal
- Engine and attention/linear/MoE backends; TP, PP, DP and EP sizes; graph capture and speculative-decoding settings
- Quality acceptance set, TTFT and inter-token latency objectives, error budget, availability and restart target

A model's advertised maximum context is a feature ceiling. It is not a promise that that context length can be served at the desired concurrency on a particular GPU.

## Inference: budget every GPU

Use bytes internally. **1 GB = 1,000,000,000 bytes; 1 GiB = 1,073,741,824 bytes.** Convert both demand and capacity to the same unit. Do not silently read a vendor's “GB” label as GiB. Use the exact EC2 specification and confirm CUDA-visible device capacity; account for partitioning and other resident processes.

For GPU rank r, use this accounting identity:

```text
peak_required_r = resident_weights_r
                + live_KV_cache_r
                + peak_activations_and_workspace_r
                + runtime_and_communication_buffers_r
                + graph_and_auxiliary_model_memory_r
                + allocator_slack_r

peak_required_r <= device_capacity_r - operational_reserve_r
```

The terms must be non-overlapping. A planning fraction such as 90% is an explicit scenario assumption, not a universal safe setting or a guarantee that the engine will keep total device usage below that number.

### Weight storage is only a lower bound

```text
ideal_weight_bytes = parameter_count × storage_bits_per_parameter / 8
```

FP32 uses 4 bytes per parameter; BF16/FP16 use 2; an ideal 8-bit representation uses 1; an ideally packed 4-bit representation uses 0.5. Real quantized checkpoints also contain scales, zero points or block metadata, padding, and possibly higher-precision embeddings, output heads or other layers. Loading, conversion and execution can allocate additional copies. Measure the actual loaded representation.

Quantization reduces storage but changes the quality and implementation contract. “INT4,” “FP4,” “FP8” and “W4A16” are not interchangeable file formats or kernel promises. Validate the exact format, group/block size, scaling scheme, model architecture and chosen backend on the target GPU. The current vLLM summary matrix does not enumerate every Blackwell target; absence from that summary is not proof of non-support. Consult the release-specific implementation and test it. [vLLM quantization](https://docs.vllm.ai/en/stable/features/quantization/)

### KV cache grows with live tokens

For a conventional decoder with the same dense attention shape in every layer, equal key/value head dimensions, and no sharing or eviction:

```text
KV_bytes = 2 × layers × KV_heads × head_dimension × KV_bytes_per_element
           × sum(live_cached_tokens_per_sequence)
```

The factor 2 is for keys and values. Use the number of **KV heads**, not the number of query heads. GQA shares KV heads among groups of query heads; MQA has one KV head; MHA has one KV head per query head. [Hugging Face Llama configuration](https://huggingface.co/docs/transformers/v4.45.2/model_doc/llama)

This formula is derived from the key/value tensor dimensions. A paged cache rounds allocated tokens into blocks and adds metadata. For a simple no-sharing upper estimate with block size q, replace each sequence length T with `ceil(T / q) × q`. Prefix reuse, sliding windows, cache eviction and engine policy change the allocation. Cache both prompt tokens and generated tokens. [vLLM attention cache tensor shapes](https://docs.vllm.ai/en/v0.9.2/api/vllm/attention/backends/flash_attn.html)

For heterogeneous layers, sum the layer-specific terms. Do not apply this dense formula unchanged to MLA latent caches, state-space models, encoder-decoder cross-attention, or multimodal caches. Their layouts need model-specific accounting. [vLLM model configuration](https://docs.vllm.ai/en/stable/api/vllm/config/model/)

Weight quantization does not automatically quantize KV cache. FP8 KV storage can reduce this component relative to BF16, but scales, backend support and accuracy must be checked. A reduced cache footprint does not by itself guarantee a latency improvement. [vLLM quantized KV cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)

### Reserve for the rest of the execution

Prefill, decode and graph capture have different peaks. Include attention and GEMM workspace, temporary activations/logits, NCCL buffers, CUDA context, captured graphs, adapters, multimodal encoders and speculative draft models as applicable. Limiting context length, concurrently scheduled sequences and graph capture shapes can reduce memory pressure, with performance tradeoffs. [vLLM conserving memory](https://docs.vllm.ai/en/stable/configuration/conserving_memory/)

Do not equate `nvidia-smi` usage, live tensor bytes and allocator-reserved bytes. PyTorch exposes distinct allocated and reserved measures, and its allocator does not see every external allocation. Clearing an unused cache does not free live model tensors or solve an intrinsically oversized workload. [PyTorch memory APIs](https://github.com/pytorch/pytorch/blob/main/torch/cuda/memory.py)

With vLLM, read the initialization memory profile and KV-token capacity, then validate under load. `gpu_memory_utilization` is an engine allocation input; `kv_cache_memory_bytes` explicitly specifies a per-GPU KV budget and takes precedence over that automatic sizing path. Neither replaces a whole-device peak measurement. Check the pinned release's options rather than copying flags from another version. [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)

## Training: a different budget

For full-parameter training, add gradients, optimizer states and saved activations. One illustrative mixed-precision Adam accounting is 2 bytes for low-precision weights + 4 for master weights + 4 for gradients + 8 for two FP32 optimizer moments = **18 bytes per parameter, before activations and temporary buffers**. This is a configuration example, not a universal BF16 rule; implementations differ in master copies and gradient/optimizer precision. At this assumption, 8B parameters need 134.11 GiB and 70B need 1,173.47 GiB before activations. [Hugging Face training memory anatomy](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_memory_anatomy.md)

Plain distributed data parallelism replicates training state and does not make an oversized replica fit. FSDP shards parameters, gradients and optimizer states, with parameter materialization and communication during execution. Budget transient all-gathers and the largest wrapped unit as well as persistent shards; naive division of 18P by GPU count is insufficient. [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)

ZeRO-1 shards optimizer states; ZeRO-2 adds gradient sharding; ZeRO-3 also shards parameters. Higher sharding stages can reduce persistent memory while increasing communication. CPU or NVMe offload trades GPU residency for host/storage capacity and transfer cost. Check optimizer-step and checkpoint-save peaks as well as forward/backward. [Hugging Face DeepSpeed ZeRO](https://huggingface.co/docs/transformers/deepspeed)

Activation checkpointing saves fewer forward intermediates and recomputes them during backward, trading compute for memory. This is different from saving a model checkpoint to disk. Microbatch size, sequence length, attention implementation and checkpoint placement affect activation peaks; gradient accumulation can preserve a desired effective batch while changing per-step execution. [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint)

Adapter fine-tuning and QLoRA need a separate budget: the frozen base can use a quantized representation while trainable adapters need gradients and optimizer state. They still require activation/workspace memory and are not equivalent to either full training or inference. Do not reuse an inference-fit result as proof that fine-tuning will fit. [Hugging Face PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)

### How the estimator represents training

The CLI defaults to a different, explicit 16-byte example: 2-byte weights, 2-byte gradients, 4-byte master weights, and two 4-byte Adam moments. Use `--gradient-bits 32` to reproduce the 18-byte example above. Neither is a universal training default. Inspect the actual optimizer and parameter/gradient dtypes. The CLI models persistent states and your supplied activation allowance; it does not predict materialization or optimizer-step peaks. See the [worked training examples](06-worked-examples.md) and [estimator reference](08-estimator-reference.md).

---

Next: [Choose a parallelism strategy](04-parallelism.md)
