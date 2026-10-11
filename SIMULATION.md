# CPU-only analytical sensitivity laboratory

**This is an analytical model, NOT a GPU emulator, benchmark, or real-instance performance prediction.** No GPU kernels, model loads, cloud resources, cloud API calls, or paid tests are used. The included numbers are deliberately hypothetical, uncalibrated assumptions. They do not rank G7e, H100, H200, B200, or B300 instances.

The purpose is to ask better portability questions: when would more arithmetic throughput help, when would memory traffic matter more, and when could a different communication path erase the benefit of extra GPUs?

**A modeled layout is not necessarily deployable.** Every output row says that memory feasibility and software support are not evaluated. The small weight/KV fields here omit runtime allocations and are not a fit test. First use the [memory estimator](docs/08-estimator-reference.md) and compatibility checks, then use this laboratory for conditional sensitivities. Only [real workload measurements](docs/07-benchmarking.md) can qualify a replacement instance.

## Run and reproduce

Python 3.10+ and its standard library are sufficient. Run from this repository root:

```bash
python3 simulation.py --help
python3 simulation.py --output-prefix examples/simulation-results
python3 simulation.py --only decode_tp8_network --json
python3 -m unittest discover -s tests -p test_simulation.py -v
```

The first command with `--output-prefix` **replaces** its `.csv` and `.json` files. It evaluates 96 scenarios: 53 inference cases, 30 standalone collectives, and 13 training-DP cases.

- [Scenario inputs](examples/simulation_scenarios.json): every scalar and partition/traffic assumption can be edited. Copy this file before experimenting.
- [Result CSV](examples/simulation-results.csv): one labeled analytical row per scenario. Empty fields mean that metric does not apply to that scenario kind.
- [Result JSON](examples/simulation-results.json): full input file, provenance, scenario SHA-256, model version, disclaimer, and output rows.
- [Implementation](simulation.py) and [tests](tests/test_simulation.py).

To try another assumption file:

```bash
python3 simulation.py --scenarios my-scenarios.json --output-prefix my-results
```

Unknown case keys, invalid rates, impossible supported partition arithmetic, malformed containers, and duplicate scenario IDs are rejected. Values must be finite. A zero-byte standalone collective is defined as no operation; one rank needs no all-reduce. Rates must stay positive. A missing/zero utilization is not treated as a working path.

## What the included scenarios show

These are consequences of the chosen equations and inputs, not empirical findings about any real GPU. Millisecond values below are **modeled service-work durations**, with no queueing or SLO guarantee.

| Isolated question | Example output | Interpretation within this model |
|---|---|---|
| More compute or more memory bandwidth? | Decode reference: 17.577 ms; doubled compute: 17.577 ms; doubled memory bandwidth: 8.788 ms | Arithmetic capacity does not help a memory-bound roofline term. Prefill in this particular scenario is compute-bound and has the reverse sensitivity. |
| Is more TP always faster? | Network-path TP4: 10.583 ms; TP8: 16.137 ms | Extra ranks reduce ideal local work but add ring phases. At TP8, raw communication is 13.929 ms versus 2.208 ms local work. |
| Can overlap erase a slow link? | Shared-network prefill TP8: 17,039.093 ms without overlap; 16,047.985 ms at the optimistic full-overlap endpoint | Only 991.109 ms of local work exists to hide communication under. The remaining communication cannot disappear. |
| More resident requests or more active work? | Eight versus 32 resident decode sequences at the same active batch: both 17.577 ms; KV proxy rises from 8.001 to 32.004 GiB/rank | Residency affects this memory proxy; only active sequences generate per-step work. Actual scheduling/queueing effects are excluded. |
| What does PP filling change? | PP4 ideal zero-transfer bubble fraction: 75% with one microbatch, 15.789% with 16 | More independent microbatches amortize fill/drain. They do not shorten the first microbatch's service path and require more resident sequences. |
| Small or large collectives? | Eight-rank network ring: 8 KiB payload is latency-dominated; 256 MiB is bandwidth-dominated | Raising bandwidth barely helps tiny messages. Lower latency and fewer phases matter there. |
| What do independent inference replicas buy? | One versus eight replicas: same 17.577 ms service work per replica; eight times aggregate modeled token rate, eight times GPUs | This assumes eight times the offered work, enough independent requests, and memory feasibility on every replica. No inference gradient sync is added. |
| How much accumulated-training work can hide DP sync? | DP8 step: 2,033.387 ms with zero overlap; 2,009.387 ms at full allowed overlap | Four microbatches do not provide four backward windows for one final sync. The eligible final-microbatch window is only 24 ms. |

Increasing active decode batch in the example amortizes one weight read over more tokens, but raises step duration and KV traffic. Increasing context raises decode KV traffic and attention work; prefill attention work has a quadratic context component. The constant utilization assumptions deliberately omit the utilization improvements or regressions that actual shape changes can cause.

The GPU counts differ between some scenarios. An eight-replica result is not an eight-GPU latency speedup for one request. Token rates use input tokens for prefill and one next-token output per active sequence for decode; **do not compare those units as the same throughput metric.** No prices or cost-efficiency scores are produced.

## Equations, units, and assumptions

### 1. Local arithmetic and memory roofline

Use decimal rates: one TFLOP/s = 10^12 FLOP/s; one GB/s = 10^9 bytes/s. Memory-capacity displays use GiB = 2^30 bytes. A multiply plus add counts as two FLOPs.

For device inputs `peak_TFLOPs`, `compute_utilization`, `memory_peak_GBps`, and `memory_utilization`:

```
C = peak_TFLOPs * compute_utilization * 10^12       [FLOP/s]
B = memory_peak_GBps * memory_utilization * 10^9    [byte/s]
t_compute = F / C                                  [s]
t_memory  = M / B                                  [s]
t_local   = max(t_compute, t_memory)                [s]
AI = F / M                                        [FLOP/byte]
ridge = C / B                                     [FLOP/byte]
```

This is the roofline construction, expressed as time rather than throughput. It does not sum arithmetic and memory time. Treating an entire stage as one roofline region also ignores serial kernels with different bottlenecks; it is a coarse conditional envelope. [NERSC roofline documentation](https://docs.nersc.gov/tools/performance/roofline/)

The three generic profiles change only one ceiling at a time. Their 50% compute and 70% memory utilization are **invented inputs, not calibration**. Peak arithmetic must refer to the precision and dense/sparse mode actually used. Quantizing weights does not automatically select a valid low-precision arithmetic rate. Matrix multiplication's operation count and arithmetic-intensity framing are described by [NVIDIA's matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).

### 2. Dense decoder forward proxy

Let `P` be the supplied approximate number of dense matrix parameters, `L` layers, `H` hidden size, `K` KV heads, `D` head dimension, `b` sequences per microbatch, `S` context, `t` TP degree, and `p` PP stages.

- Prefill: `S` is prompt length; tokens = `b*S`; causal query-key pairs = `b*S*(S+1)/2`.
- Decode: `S` is **preexisting cached tokens**; tokens = `b`; pairs = `b*(S+1)`, including attention to the current token.

```
F_per_stage_rank = (2*P*tokens + 4*L*H*pairs) / (t*p)
weight_bytes_per_rank = P * bytes_per_weight / (t*p)
weight_traffic = weight_bytes_per_rank * weight_read_passes
KV_unit = 2 * (L/p) * (K/KV_shards) * D * bytes_per_KV_element
KV_read = KV_unit * b * S * kv_read_passes          [decode only]
KV_write = KV_unit * tokens                        [both modes]
other_traffic = other_activation_passes * tokens * H * (L/p)
                * bytes_per_activation / activation_shards
M = weight_traffic + KV_read + KV_write + other_traffic
```

These workload expressions are transparent **model assumptions**, not a profiled transformer implementation. `2*P*tokens` approximates linear-layer multiply-adds. The two attention matrix products contribute four FLOPs per hidden channel per query-key pair. Softmax, normalization, positional operations, sampling, embeddings/output peculiarities, reductions, launch costs, and other kernels are not independently modeled.

`weight_read_passes=1` assumes ideal reuse of each weight per microbatch, even at large prefill shapes; real tiles may reread it. `kv_read_passes` controls rereads of prior cache during decode. Current-token K/V is written once; no extra HBM read is charged for using it immediately. Prefill assumes fused attention and does not materialize a quadratic score matrix or separately charge all K/V rereads. Those are important optimistic simplifications.

`other_activation_passes` is an arbitrary count of additional hidden-sized read/write equivalents **excluding the weight and KV terms already counted**. The example value eight is not a known kernel traffic count. It is there so users can vary omitted traffic rather than quietly assume it is free. Arithmetic precision, KV bytes, weight bytes, and activation bytes are separate inputs.

Balanced PP requires `L % p == 0`. Query heads `H/D` must be integral and divisible by TP; KV heads must divide query heads. If `kv_tp_sharded=true`, KV heads must also divide evenly across TP. If false, the full KV set is charged on each TP rank. No claim is made that a particular engine supports every arithmetically allowed layout. There is no automatic padding, uneven stage placement, embedding/lm-head imbalance, or partial-KV-replication algorithm.

The two capacity fields cover only weights and a common-length KV proxy. Resident KV is `KV_unit * resident_sequences * S` for prefill and uses `S+1` for decode. The latter budgets the next-token length for every resident sequence, even if inactive. `resident_sequences` is **per replica**, must cover `batch * microbatches`, and is distinct from active batch. Full memory feasibility belongs to the separate estimator and a real load test.

### 3. Actual-path transport versus advertised bandwidth

Each transport has:

- `directional_rank_peak_GBps`: a **one-direction, per-rank** ceiling for the chosen path
- `utilization`: assumed fraction of that ceiling achieved by this transport/workload
- `sharing_factor`: additional contention multiplier, at least one
- `latency_us`: effective per-phase startup delay
- `actual_path`: explicit path assumption, such as local P2P, host-staged, or inter-node
- `provenance` and `advertised_description`: metadata retained in the input/output JSON

```
B_effective = directional_rank_peak_GBps * 10^9 * utilization / sharing_factor
```

**Advertised metadata is never fed to the equation.** Bidirectional aggregate NVLink bandwidth, total instance NIC bandwidth, and Gb/s Ethernet rates cannot be copied unmodified into a per-rank directional GB/s field. Divide bits/s by eight to get bytes/s, then resolve direction, rank allocation, path, and contention separately. Do not divide twice for sharing if a measured per-rank rate already includes it.

The example path names are conceptual only. A `pcie_local` case assumes P2P works; `host_staged_local` assumes an effective staged path whose rate and latency already include staging costs. It does not merely relabel a direct path. A `network` case does not prove EFA, GPUDirect RDMA, or an appropriate NCCL plugin is active. Transport diagnostics and application-level collective measurements are necessary; [NCCL performance troubleshooting](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/troubleshooting/performance_and_tuning.html) explains why topology, peer access, and network tests matter.

### 4. Ring collectives, TP, and node scaling

For an all-reduce of `n` bytes **on each rank** across `r` ranks:

```
t_ring = 2*(r-1)*alpha + [2*(r-1)/r] * n / B_effective
alpha = latency_us * 10^-6
```

The first term is the chosen ring's startup model; the second is its byte-volume model. NCCL can use trees, hierarchical collectives, multiple channels, and specialized protocols; this script does not select or reproduce them. For heterogeneous/inter-node cases it applies a single slow-path envelope to all ring phases, not an accurate graph of local and remote edges. `nodes` and `ranks_per_node` validate rank placement counts; they do **not** discover topology or automatically select the transport.

[NVIDIA NCCL tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) distinguish algorithm bandwidth `n/time` from bus bandwidth `algbw * 2*(r-1)/r` for all-reduce. This model expects a directional link/bus-style rate and applies that factor itself. **Do not put NCCL `algbw` into this field and apply the factor again.** Small-message `busbw` already includes latency in the measured time, so fitting both bandwidth and alpha from the same single point would also double-count overhead. Fit the intercept and slope of a suitable message-size region instead.

Inference TP charges `tp_collectives_per_layer * L/p` all-reduces per stage microbatch. Payload is `tokens * H * activation_bytes`, with no automatic division by TP for this ordinary row-parallel-style all-reduce. The example uses two forward collectives per layer, motivated by the classic [Megatron-LM decomposition](https://research.nvidia.com/labs/adlr/MegatronLM/). This is not universal: sequence/context/expert parallelism and newer implementations change the operations and payloads. The input count is user-controlled.

### 5. Overlap uses one finite budget

For local work `C`, summed communication `K`, overlap fraction `f`, and eligible local fraction `e`:

```
hidden = f * min(e*C, K)
exposed = K - hidden
total = C + exposed
```

`f=0` serializes the two; `f=1,e=1` yields `max(C,K)`. Communication is first summed, then hidden **once**. It cannot be independently hidden under the same compute time multiple times. Dependency constraints may allow far less overlap; the endpoints are sensitivity bounds, not promises from setting an engine flag. Examples use no TP overlap unless the case explicitly varies it. [Megatron Core's parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) distinguishes the parallelism dimensions and implementation-specific overlap features.

### 6. PP fill/drain and transfers

The forward-only deterministic scheduler models one serial service resource per stage and one serial transfer resource per stage boundary. Stage service includes local roofline time plus exposed TP. A microbatch can begin a stage only after its previous-stage transfer and that stage's prior microbatch have completed. Each transfer similarly waits for its link. Different links and stages can overlap.

For `p` identical stages of duration `c`, `m` independent microbatches, and zero transfer cost:

```
total = (m+p-1)*c
ideal bubble fraction = (p-1)/(m+p-1)
```

The familiar microbatch fill/drain effect motivates pipeline parallelism; see [GPipe](https://arxiv.org/html/1811.06965v5). Here, nonzero edge transfer time is `alpha + payload/B_effective`, separately scheduled. The reported ideal zero-link bubble is always labeled as a reference quantity; actual stage service occupancy comes from the schedule. It is **not GPU utilization**, because stage service includes communication waits.

The example sends a full activation payload at each PP boundary; `pp_payload_tp_sharded` can instead divide it by TP. This is an explicit layout assumption, not inferred support. Links are assumed dedicated and independent of TP/local-memory resources. Shared fabric, copy-engine contention, uneven layers, backpressure, and queue limits require a richer schedule or measurements. The scheduler does not implement backward passes, GPipe training, 1F1B, interleaved stages, or repeated autoregressive token feedback. In decode, its microbatches are distinct sets of sequences taking one step each.

### 7. Training DP versus independent serving replicas

`training_dp` accepts explicit per-rank FLOPs and HBM bytes per microbatch instead of claiming to derive an entire training graph. Its gradient payload is per rank; it splits that payload into equal buckets and adds each bucket's ring time. More buckets add alpha terms without changing total gradient wire volume.

The schedule assumes local accumulation with synchronization suppressed until the final microbatch. There is one bucketed gradient synchronization per step. **Only `eligible_final_microbatch_fraction` of the final microbatch's local service is eligible for overlap**, not that fraction of all accumulated work. The default 0.6 is an invented backward-window fraction; actual bucket readiness can reduce overlap further. `optimizer_s` is serial extra work and is zero in the examples, so those examples omit optimizer cost. They also omit gradient-buffer/optimizer-memory feasibility, FSDP/ZeRO parameter gathers, and cross-step pipelining.

The DP sweep is weak scaling: fixed per-rank work, more total samples and GPUs. It does not preserve global batch or training convergence. A strong-scaling experiment must explicitly change per-rank work and samples while holding total workload fixed, then account for utilization and synchronization changes.

Independent inference `replicas` multiply total processed tokens and GPU count. They do not divide model memory or per-replica service time, and they add no training gradient collective. This comparison is useful only if a local model replica is memory-feasible and enough independent traffic exists.

## What would real measurements calibrate?

Do not infer these values from model marketing names. For the exact instance size, software versions, model, precision, shape, and parallel layout:

1. Measure precision/shape-specific kernel arithmetic and device-memory ceilings; profile FLOPs, bytes, and time per important region. Fit separate utilization assumptions for prefill/decode, shapes, and TP sizes rather than one universal efficiency.
2. Inspect actual P2P/topology and communication logs. Measure point-to-point behavior and actual collective operations, message sizes, rank placements, buckets, and concurrency. Preserve NCCL algorithm/bus bandwidth distinctions.
3. Fit a latency/intercept and bandwidth/slope only over regions where the chosen algorithm remains appropriate. Check held-out message sizes and node counts. Algorithm switches invalidate a single alpha-beta fit.
4. Profile weight/KV/activation traffic, kernel launches, real shard imbalance, and current-token cache behavior. Replace proxy counts with measurements; do not hide omissions in arbitrary utilization factors.
5. Trace TP/PP/DP dependencies and overlap. Use bucket-ready times and backward windows; count shared-resource contention once. Validate that a purported overlap interval is actually concurrent useful work.
6. Measure the full serving/training workload, quality, memory peaks on every rank, queueing, p50/p95/p99, useful throughput, errors, and recovery. Only then consider prices and cost per SLO-passing output. Record uncertainty and repeat runs.

Use the [benchmark protocol](docs/07-benchmarking.md) for qualification. This laboratory does not launch any of those experiments. It helps prioritize which ones are most informative.

## Exclusions and interpretation limits

This is deliberately not complete hardware or application simulation. It does not execute CUDA/PTX, reproduce instruction scheduling or tensor-core kernels, validate compute capabilities, load weights, test numerical quality, enforce device memory capacity, verify engine features, or query live AWS capacity/prices.

It does not model MoE routing/all-to-all, context parallelism, CPU/NUMA/host bottlenecks, storage/model-loading time, quantization/dequantization kernels, cache hierarchy, memory fragmentation, attention variants, prefix sharing, speculative decoding, disaggregated serving, scheduler policies, timeouts, power, failures, or network congestion dynamics. Generic PP and DP cases do not combine into a complete distributed training engine.

Constant efficiencies and ideal balanced partitioning can be especially optimistic as per-rank matrix shapes shrink. The ring envelope can also be pessimistic relative to hierarchical algorithms. Therefore even the direction of a cross-platform ranking can change. Use scenario differences as hypotheses to test, never evidence that a named migration will pass its SLO.
