"""Auditable memory arithmetic, intentionally not a model-fit predictor.

Bytes use SI parameter counts and binary GiB (2**30 bytes). Calculations screen
one rank, not the sum of all device memory. See docs/08-estimator-reference.md.
"""

from dataclasses import asdict, dataclass
import math

GIB = 2**30


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _real(name, value, minimum=0.0, exclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (exclusive and value == minimum):
        operator = ">" if exclusive else ">="
        raise ValueError(f"{name} must be {operator} {minimum}")


def _common(config):
    _integer("parameters", config.parameters)
    _integer("tp", config.tp)
    _integer("pp", config.pp)
    _real("gpu_memory_gib", config.gpu_memory_gib, exclusive=True)
    _real("headroom_fraction", config.headroom_fraction)
    if config.headroom_fraction >= 1:
        raise ValueError("headroom_fraction must be < 1")
    _real("runtime_overhead_gib", config.runtime_overhead_gib)
    _real("weight_imbalance_factor", config.weight_imbalance_factor, minimum=1)


def _result(mode, config, components, warnings, gpu_count, extra):
    total = sum(components.values())
    budget = config.gpu_memory_gib * GIB * (1 - config.headroom_fraction)
    if not math.isfinite(total) or not math.isfinite(budget):
        raise ValueError("computed memory exceeds the supported numeric range")
    return {
        "schema_version": 1,
        "mode": mode,
        "decision": "within_budget_screen" if total <= budget else "exceeds_budget_screen",
        "interpretation": (
            "Arithmetic screen only. A within-budget result is not evidence that the model "
            "loads, runs, meets an SLO, or has capacity available."
        ),
        "inputs": asdict(config),
        "gpu_count": gpu_count,
        "components_per_gpu_bytes": components,
        "estimated_per_gpu_gib": total / GIB,
        "usable_budget_per_gpu_gib": budget / GIB,
        "margin_per_gpu_gib": (budget - total) / GIB,
        "warnings": warnings,
        **extra,
    }


@dataclass(frozen=True)
class InferenceConfig:
    parameters: int
    gpu_memory_gib: float
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    context_tokens: int
    concurrency: int
    weight_bits: int = 16
    kv_bits: int = 16
    tp: int = 1
    pp: int = 1
    kv_layout: str = "sharded"
    headroom_fraction: float = 0.15
    runtime_overhead_gib: float = 0.0
    weight_overhead_fraction: float = 0.0
    weight_imbalance_factor: float = 1.0


def estimate_inference(config):
    """Screen a dense, uniform-layer decoder; reject unsupported partition math."""
    _common(config)
    for name in ("layers", "query_heads", "kv_heads", "head_dim", "context_tokens", "concurrency"):
        _integer(name, getattr(config, name))
    _integer("weight_bits", config.weight_bits)
    _integer("kv_bits", config.kv_bits)
    if config.weight_bits not in (4, 8, 16, 32):
        raise ValueError("weight_bits must be one of 4, 8, 16, 32")
    if config.kv_bits not in (8, 16, 32):
        raise ValueError("kv_bits must be one of 8, 16, 32")
    _real("weight_overhead_fraction", config.weight_overhead_fraction)
    if config.query_heads % config.kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads for this GQA/MHA screen")
    if config.query_heads % config.tp:
        raise ValueError("query_heads must be divisible by tp for this screen")
    if config.layers % config.pp:
        raise ValueError("layers must be divisible by pp; uneven pipeline stages require a separate calculation")
    if config.kv_layout not in ("sharded", "replicated"):
        raise ValueError("kv_layout must be sharded or replicated")
    if config.kv_layout == "sharded" and config.kv_heads % config.tp:
        raise ValueError("kv_heads must be divisible by tp for sharded KV; use a supported layout or the conservative replicated screen")

    weights_total = config.parameters * (config.weight_bits / 8) * (1 + config.weight_overhead_fraction)
    kv_per_token = 2 * config.layers * config.kv_heads * config.head_dim * (config.kv_bits / 8)
    kv_total = kv_per_token * config.context_tokens * config.concurrency
    kv_tp_divisor = config.tp if config.kv_layout == "sharded" else 1
    components = {
        "weights": weights_total / (config.tp * config.pp) * config.weight_imbalance_factor,
        "kv_cache": kv_total / (kv_tp_divisor * config.pp),
        "runtime_overhead": config.runtime_overhead_gib * GIB,
    }
    warnings = [
        "Dense uniform-layer MHA/GQA/MQA only, with equal K/V head dimensions; no MLA, MoE, hybrid, multimodal, or sliding-window-specific accounting.",
        "Equal layer stages and balanced weight shards are assumed; real embeddings, heads, and stage peaks can be uneven.",
        "KV uses full context × resident concurrency per replica; query heads are not KV heads. Data-parallel replicas do not divide this memory.",
        "Prefill activations, CUDA graphs, attention workspaces, allocator fragmentation, and communication buffers need measured allowance.",
        "KV block rounding, scales, and cache metadata are not computed; include them in the allowance.",
        "Engine, checkpoint format, GPU kernel, precision, topology, and quality compatibility are not validated.",
        "GPU memory must be an observed per-device GiB budget, not an unverified marketing GB label.",
    ]
    if config.weight_bits < 16:
        warnings.append("Quantized weight payload excludes metadata and unquantized tensors unless your weight overhead input covers them.")
    if config.kv_bits < 16:
        warnings.append("Quantized KV needs supported kernels, scaling metadata, and a quality check; weight precision does not set KV precision.")
    if config.kv_layout == "replicated":
        warnings.append("Replicated mode conservatively places all stage KV heads on every TP rank; some engines use partial replication instead.")
    if config.runtime_overhead_gib == 0:
        warnings.append("Runtime overhead is zero; this omits non-weight/non-KV allocations beyond the separate headroom reserve.")
    return _result("inference", config, components, warnings, config.tp * config.pp, {
        "weights_total_bytes_before_sharding": weights_total,
        "kv_bytes_per_cached_token_before_sharding": kv_per_token,
        "kv_total_bytes_before_sharding": kv_total,
        "layers_per_pipeline_stage": config.layers // config.pp,
    })


@dataclass(frozen=True)
class TrainingConfig:
    parameters: int
    gpu_memory_gib: float
    activation_gib: float
    weight_bits: int = 16
    gradient_bits: int = 16
    master_weight_bits: int = 32
    optimizer_bits_per_parameter: int = 64
    tp: int = 1
    pp: int = 1
    dp: int = 1
    zero_stage: int = 0
    headroom_fraction: float = 0.15
    runtime_overhead_gib: float = 0.0
    weight_imbalance_factor: float = 1.0


def estimate_training(config):
    """Persistent-state lower-bound screen, not peak training-step prediction."""
    _common(config)
    _integer("dp", config.dp)
    _integer("zero_stage", config.zero_stage, minimum=0)
    if config.zero_stage > 3:
        raise ValueError("zero_stage must be one of 0, 1, 2, 3")
    _real("activation_gib", config.activation_gib)
    for name in ("weight_bits", "gradient_bits"):
        _integer(name, getattr(config, name))
        if getattr(config, name) not in (16, 32):
            raise ValueError(f"{name} must be 16 or 32 for this full-parameter training screen")
    _integer("master_weight_bits", config.master_weight_bits, minimum=0)
    if config.master_weight_bits not in (0, 16, 32):
        raise ValueError("master_weight_bits must be one of 0, 16, 32")
    _integer("optimizer_bits_per_parameter", config.optimizer_bits_per_parameter, minimum=0)

    model_divisor = config.tp * config.pp
    parameter_factor = config.parameters / model_divisor * config.weight_imbalance_factor
    components = {
        "weights": parameter_factor * config.weight_bits / 8 / (config.dp if config.zero_stage >= 3 else 1),
        "gradients": parameter_factor * config.gradient_bits / 8 / (config.dp if config.zero_stage >= 2 else 1),
        "master_weights": parameter_factor * config.master_weight_bits / 8 / (config.dp if config.zero_stage >= 1 else 1),
        "optimizer_states": parameter_factor * config.optimizer_bits_per_parameter / 8 / (config.dp if config.zero_stage >= 1 else 1),
        "activations": config.activation_gib * GIB,
        "runtime_overhead": config.runtime_overhead_gib * GIB,
    }
    warnings = [
        "Persistent-state lower bound plus user-supplied per-rank activations/runtime; NOT peak memory of a training step.",
        "Default 16-byte parameter state assumes 16-bit gradients; use gradient_bits=32 for an 18-byte variant. Actual optimizer implementations differ.",
        "Equal TP/PP state partitions are assumed; architecture divisibility and parallelism support are not checked in training mode.",
        "ZeRO-3/FSDP all-gathers, unsharded layers, prefetch, communication buckets, optimizer workspaces, and checkpoints can raise the peak.",
        "Activations must be supplied for the actual microbatch, sequence length, checkpointing, and schedule; they are not divided again.",
        "This is full-parameter training arithmetic, not a LoRA/QLoRA, offload, MoE, or FP8 training recipe.",
        "GPU memory must be an observed per-device GiB budget. Kernel support and throughput are not validated.",
    ]
    if config.zero_stage and config.dp == 1:
        warnings.append("dp=1 gives no ZeRO data-parallel sharding benefit.")
    if config.activation_gib == 0:
        warnings.append("Activations are zero; this is a state-only lower bound and cannot establish training feasibility.")
    return _result("training", config, components, warnings, config.tp * config.pp * config.dp, {
        "persistent_bytes_per_parameter_before_sharding": (
            config.weight_bits + config.gradient_bits + config.master_weight_bits + config.optimizer_bits_per_parameter
        ) / 8,
    })
