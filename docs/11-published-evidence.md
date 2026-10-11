# 11 · Published evidence: what transfers across GPUs, and what does not

[Guide home](../README.md) · Reviewed 2026-10-11. This is a literature review, not a benchmark run by this project. No paid GPU instances were launched. Historical results below are deliberately kept with their original hardware, software, workloads, and metric definitions.

## The practical answer

GPU portability has at least three separate questions: does the program execute correctly, can the model and runtime fit, and does the resulting service meet its quality, latency, throughput, and cost targets? Published experiments establish mechanisms and useful test cases. They do not supply a universal conversion factor from G7e to H100, H200, B200, or B300.

For a migration, use the studies below to decide what to measure:

- Check the exact compiled kernels and serving backend before comparing GPU specifications.
- Separate cold start, prefill, decode, and sustained mixed traffic. Record actual generated lengths, not just requested limits.
- Treat the parallelism plan as part of the configuration. Changing TP, PP, replication, sharding, or expert parallelism changes the workload placed on the links.
- Keep a quality gate alongside performance. A faster precision or quantization recipe is useful only if the application still passes its acceptance tests.
- Require an application trace for a claim about the active communication path. A physical link, a successful peer copy, and an application collective are different evidence.

These are engineering recommendations synthesized from the selected studies, not measured speedup predictions. Study numbers are not intended to be compared with one another.

## Evidence map

Use this table to choose a case study. The details below are the technical appendix; each result stays with its original conditions.

| Decision | Start with | Evidence boundary |
|---|---|---|
| Will the G7e software path work? | [1. G7e practitioner run](#1-actual-g7e-a-useful-practitioner-report-with-corrected-results) | Version-specific, self-reported; corrected workload and tuning claims |
| Can a published GPU score guide buying? | [2. MLPerf](#2-mlperf-reproducible-full-system-results-with-important-comparison-boundaries) | Whole-system recipe; preserve division, scenario, quality, and topology |
| Should TP cross nodes? | [3. Sarathi-Serve](#3-sarathi-serve-parallelism-and-scheduler-choices-change-the-network-penalty) | Measured Ethernet testbed, not a universal EFA result |
| Should prefill and decode be separate? | [4. DistServe](#4-distserve-disaggregation-depends-on-placement-and-the-slo) | Placement and KV-transfer cost; simulation identified separately |
| Why did a new GPU disappoint? | [5. FlashAttention-3](#5-flashattention-3-using-a-newer-gpus-instructions-requires-the-right-kernel) | Kernel-level evidence, not whole-model throughput |
| Why did sharded training stop scaling? | [6. FSDP](#6-fsdp-faster-compute-can-expose-communication-and-memory-management-stalls) | Overlap and memory pressure; dated implementation |
| How should TP/PP/DP be composed? | [7. Megatron-LM](#7-megatron-lm-training-scale-comes-from-a-joint-tpppdp-plan) | Weak scaling and iteration throughput, not completed training |
| Can lower precision remove a GPU boundary? | [8. SmoothQuant](#8-smoothquant-quantization-can-cross-a-fit-boundary-with-measured-quality-conditions) | Recipe-specific fit and quality, not generic low-bit accuracy |
| Which interconnect metric matters? | [9. NCCL](#9-demystifying-nccl-the-winning-protocol-depends-on-message-size-and-path) | Message-size curve, not advertised peak bandwidth |
| Why is cold readiness still slow? | [10. ServerlessLLM](#10-serverlessllm-storage-and-the-host-affect-useful-latency-before-inference-starts) | Checkpoint loading, not steady-state tokens/s |

**Retraction worth remembering:** the G7e report's opening tuning gain was a cold-versus-warm artifact. Use its corrected follow-up; do not repeat the superseded headline.

## Ten selected measured studies and benchmark series

### 1. Actual G7e: a useful practitioner report with corrected results

**Evidence class:** self-reported practitioner experiment; neither peer-reviewed nor an independent replication.

**Configuration:** MiniMax-M2.5 FP8 on g7e.48xlarge, eight RTX PRO 6000 Blackwell Server Edition GPUs; TP8/EP8; FP8 KV cache; SGLang 0.5.8.post1, Triton 3.5.1, PyTorch 2.9.1+cu128, driver 580.126.09. The report lists CUDA 13.0 but nvcc 12.8, so do not collapse these into one runtime version.

**Measured result:** the February 20 update recommends the seeded 500-request, requested-200-input/200-output, infinite-arrival-rate reference: **1,817.2 ± 2.4 aggregate output tokens/s**, with actual mean output 183.5 tokens. The original 3,822 ± 7 result generated only 101.4 tokens/request. A claimed tuning improvement was withdrawn after warm-to-warm comparison. In this version, FP8 backend selection also determined whether the model ran at all.

**Meaning:** fix workload seed, actual lengths, warmup, and backend when migrating. **Cannot infer:** an H100 comparison, current SGLang limitations, a universal EP8 requirement, or all-pairs PCIe P2P. The no-CUDA-graph experiment had overlapping runs; exclude its magnitude from headline comparisons. Claimed bottleneck explanations were not profiled.

Source: [SGLang issue #18870, including February 20 follow-up](https://github.com/sgl-project/sglang/issues/18870). [Author's raw-artifact collection](https://gist.github.com/stewtong/980e7b95d03443d81cdd1057582c4518).

### 2. MLPerf: reproducible full-system results, with important comparison boundaries

**Evidence class:** standardized benchmark submissions, usually measured and submitted by vendors; verification is not an independent laboratory rerun.

**Concrete example:** NVIDIA's MLPerf Inference v5.1 report identifies DeepSeek-R1 results 5.1-0071 and 5.1-0072 on **72-GPU GB200 NVL72 and GB300 NVL72**. Its derived per-GPU offline figures are 4,024 and 5,842 output tokens/s; server figures are 2,327 and 2,907. Server constraints include 2-second TTFT and 80-ms TPOT. Blackwell recipes use NVFP4 weights and FP8 KV cache. NVIDIA explicitly marks the accompanying H200 DeepSeek comparison **unverified by MLCommons**. Per-GPU throughput is derived, not MLPerf's primary metric.

**Meaning:** preserve submission ID, division, scenario, quality target, entire system, and implementation. **Cannot infer:** B300 EC2 performance from GB300 NVL72, a pure silicon gain, or independent single-GPU performance by dividing a distributed score. NVIDIA's separate disaggregation comparison changes DGX B200/GB200 topology and mixes open/closed results, so it does not isolate disaggregation.

Sources: [NVIDIA v5.1 Table 2 and Figure 1 footnotes](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/), [MLCommons results](https://docs.mlcommons.org/inference_results_v5.1/), [submission artifacts](https://github.com/mlcommons/inference_results_v5.1), [benchmark rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc).

**Additional audit point:** the v5.1 table also has an eight-RTX-PRO-6000 HPE submission, 5.1-0051. That establishes measured results on that GPU/server combination, not on AWS G7e. NVIDIA's v5.0 report says H100 Llama-2-70B throughput improved up to 1.5× over the preceding year through software changes. Thus even a fixed chip name does not fix its performance baseline. [NVIDIA v5.0, “Hopper continues to improve”](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0).

### 3. Sarathi-Serve: parallelism and scheduler choices change the network penalty

**Evidence class:** peer-reviewed OSDI 2024 systems paper, academic/Microsoft authors; author-run experiments and an artifact.

**Configuration:** Falcon-180B across two Azure NC96ads v4 nodes, four A100 80GB GPUs per node, pairwise NVLink within nodes, 100-Gbps Ethernet between nodes. Comparison: TP8 versus TP4/PP2. The implementation uses NCCL for TP and PP, FlashAttention, and a modified vLLM. Artifact appendix specifies CUDA 12.1; consult the paper's artifact branch for exact baseline revisions.

**Measured result:** Figure 13 reports approximately **2× higher median time-between-tokens** for cross-node TP8 than TP4/PP2 on decode-only batches. Under the strict 1-second P99 TBT target, the scheduler's hybrid configuration increases Falcon serving capacity **3.6× over its vLLM hybrid baseline**. These are distinct comparisons. Requests follow a Poisson trace with openchat_sharegpt4 lengths.

**Meaning:** a topology change can require a new TP/PP and batching plan. **Cannot infer:** that PP always wins, that an EFA fabric behaves like this Ethernet fabric, or that modern vLLM retains the 2024 baseline's scheduler behavior.

Source: [Sarathi-Serve, §5.3, Figure 13, Tables 1–3 and artifact appendix](https://www.usenix.org/system/files/osdi24-agrawal.pdf).

### 4. DistServe: disaggregation depends on placement and the SLO

**Evidence class:** peer-reviewed OSDI 2024 paper; measured main evaluation, simulated high-bandwidth placement ablation.

**Configuration:** four nodes, each eight A100 SXM 80GB GPUs with NVLink; inter-node bandwidth **25 Gbps**. OPT-13B/66B/175B in FP16, ShareGPT prompts, Poisson arrivals; TTFT/TPOT requirements vary by model. The implementation uses asynchronous CUDA copies intra-node and NCCL cross-node. Exact package revisions are not enumerated in the evaluation text.

**Measured result:** §6.2 reports **2.0–4.6× higher request rates than its vLLM baseline**, meeting both latency targets for over 90% of requests on the chatbot workload. This is goodput under specified constraints, not unconstrained tokens/s. Low-bandwidth placement keeps the relevant prefill/decode stages on the same machine to use NVLink for cache transfer.

**Meaning:** independent prefill/decode allocation can help, but transporting KV cache becomes part of the design. **Cannot infer:** a generic cross-node speedup. The high-cross-node-bandwidth §6.4 comparison was simulated, not run on that testbed; it is not evidence for measured EFA performance. Nor do OPT's capped contexts establish modern long-context behavior.

Source: [DistServe, §§5–6 and Table 1](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf).

### 5. FlashAttention-3: using a newer GPU's instructions requires the right kernel

**Evidence class:** primary research with academic and NVIDIA/industry coauthors; attention microbenchmarks.

**Configuration:** H100 SXM5 80GB, 700 W, clock fixed at 1,830 MHz; CUDA 12.3, PyTorch 2.3.0, cuDNN 9.1.1.17, CUTLASS 3.5, FlashAttention 2.5.8; 100 repetitions averaged. FP16 attention, sequence-length sweep around 512–16K, total approximately 16K tokens, hidden size 2,048, specified head dimensions and causal/noncausal variants.

**Measured result:** forward attention is roughly **1.5–2.0× faster than FlashAttention-2**, reaching **740 TFLOP/s** in the reported configurations. The paper ties improvements to Hopper-aware scheduling, warp specialization, and overlapping matrix multiplication with softmax.

**Meaning:** a functioning CUDA program may leave substantial architecture-specific capability unused. Record the dispatched attention implementation, shapes, and build. **Cannot infer:** a 2× whole-model speedup, a decode throughput, or the same gain on G7e/Blackwell. Its FP8 numerical-error tests are kernel-accuracy experiments, not an application's model-quality acceptance test.

Source: [FlashAttention-3 v2, §4 and Appendix C.1](https://arxiv.org/html/2407.08608v2).

### 6. FSDP: faster compute can expose communication and memory-management stalls

**Evidence class:** primary Meta/PyTorch engineering paper; implementation-author measurements.

**Configuration:** A100 80GB clusters with a reported 2-Tb/s RoCE network, up to 512 GPUs. Tests include minGPT-175B and T5-11B; large-model scaling uses BF16, Adam, activation checkpointing, full sharding, backward prefetching, and rate limiting. This is the 2023 FSDP implementation, not a measurement of today's FSDP2.

**Measured result:** backward prefetching improves GPT-175B throughput by approximately **18%** (Figure 6b). For T5-11B, per-GPU TFLOP/s falls about **7%** from 8 to 512 GPUs despite ample memory; the authors attribute this to communication outgrowing the ability to overlap it. Rate limiting is workload-dependent: it helps some memory-pressure cases and can hurt others.

**Meaning:** inspect the uncovered portion of all-gather/reduce-scatter time, allocation retries, and step time, rather than summing communication durations that overlap useful work. **Cannot infer:** that a smaller shard or faster GPU guarantees a shorter step, or that this RoCE result quantifies an EFA or PCIe-only migration.

Source: [PyTorch FSDP, §§5.1–5.4, Figures 6–8](https://arxiv.org/html/2304.11277v1).

### 7. Megatron-LM: training scale comes from a joint TP/PP/DP plan

**Evidence class:** primary research by NVIDIA/Microsoft/academic authors; measured iteration throughput plus separate projected training durations.

**Configuration:** Selene DGX A100 nodes: eight A100 80GB GPUs with NVLink/NVSwitch and eight 200-Gbps InfiniBand HCAs per node. Mixed-precision GPT workloads use PyTorch, NCCL, sequence length 2,048, interleaved pipeline scheduling, and scatter/gather communication optimizations. Exact package release numbers are not given in the cited evaluation.

**Measured result:** Table 1's **1,008-billion-parameter** configuration uses **3,072 GPUs, TP8, PP64, global batch 3,072**, reaching **163 TFLOP/s per GPU, 502 PFLOP/s aggregate** under the paper's FLOP-count convention. It is a weak-scaling series: model size and batch increase with GPU count. The paper's 84-day full-training figure is an extrapolation, not a completed training run.

**Meaning:** map frequent TP collectives to the strongest links, then assess pipeline boundaries, DP synchronization, and microbatch bubbles together. **Cannot infer:** these utilization numbers for FSDP, EC2 EFA, G7e, or a fixed-size strong-scaling experiment.

Source: [Megatron-LM, §§3–5, Table 1](https://arxiv.org/pdf/2104.04473).

### 8. SmoothQuant: quantization can cross a fit boundary, with measured quality conditions

**Evidence class:** peer-reviewed ICML 2023 paper, academic/industry authors; implementation-specific measurements.

**Configuration:** A100 80GB servers, OPT models, SmoothQuant-O3 W8A8 with CUTLASS INT8 kernels in PyTorch/FasterTransformer. Calibration uses 512 Pile sentences. Exact framework releases are not enumerated in the performance section.

**Measured result:** the PyTorch context-stage experiment reaches **1.51× speedup for OPT-30B, batch 4, sequence 256**. FasterTransformer context experiments fit OPT-66B on one GPU instead of two and OPT-175B on four instead of eight with similar or better latency. For MT-NLG-530B, the mean score across four reported tasks is **73.1% for both FP16 and INT8**, although individual task scores differ.

**Meaning:** quantization can remove a sharding boundary and its communication as well as reducing bytes. Measure the resulting deployment, including quality. **Cannot infer:** that all 8-bit methods are faster, that benchmark-average equality means unchanged outputs, or that W8A8 results prove FP4/NVFP4 quality on modern reasoning or long-context tasks. Weight precision, activation precision, KV-cache precision, and their kernels must be recorded separately.

Source: [SmoothQuant, §§5.1–5.4, Figures 8–9 and Tables 8–9](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf).

### 9. Demystifying NCCL: the winning protocol depends on message size and path

**Evidence class:** academic/vendor-coauthored primary analysis and measured microbenchmarks; uses NCCL 2.19.1 internals.

**Configuration:** CSCS Alps, 16 GH200 nodes, reported 150-GB/s intra-node interconnect and 25-GB/s-per-direction Slingshot network link. Figure 6 uses warmup and 20 runs per point, comparing Ring/Tree AllReduce and Simple/LL/LL128 protocols.

**Measured result:** LL/LL128 perform best on inter-node messages below **64 KiB**, whereas Simple wins as message sizes reach the gigabyte range. Intra-node LL128 approaches Simple for large messages, reported about **5% slower**, while retaining low-latency behavior.

**Meaning:** a maximum-bandwidth number does not characterize the small collectives that can dominate TP decode. Retain latency and bandwidth curves at the application's message sizes; do not replace NCCL's autotuning with a universal setting based on this paper.

**Cannot infer:** an EFA speedup, a measured G7e PCIe path, or current NCCL algorithm selection. The paper's network is Slingshot; its generalized RoCE explanation is not a separate RoCE experiment. The source-code analysis and ATLAHS simulation work are distinct from these measurements.

Source: [Demystifying NCCL v1, §V-E and Figure 6](https://arxiv.org/html/2507.04786v1).

### 10. ServerlessLLM: storage and the host affect useful latency before inference starts

**Evidence class:** peer-reviewed OSDI 2024 systems paper; cold-loading microbenchmarks and cluster experiments.

**Configuration for the cited result:** eight A5000 GPUs, dual AMD EPYC 7453 CPUs, 1 TB DDR4, two PCIe-4 NVMe SSDs in RAID0, measured storage throughput 12 GB/s. Checkpoints are FP16; 20 checkpoint copies are randomly accessed after clearing page/inode caches. Baselines are the paper's PyTorch and Safetensors implementations; exact package revisions are not stated in §7.2.

**Measured result:** Figure 6 shows Llama-2-70B mean loading times of **84 s (PyTorch), 48 s (Safetensors), and 10.3 s (ServerlessLLM)**. This experiment concerns checkpoint loading, not token generation. Direct I/O and chunked parallel loading are part of the implementation change.

**Meaning:** a GPU upgrade can fail an end-to-end latency or recovery target because loading remains dominated by storage, host memory, or transfer orchestration. Benchmark cold readiness separately from warmed serving. **Cannot infer:** these savings on EBS, S3, a different filesystem, or GPUDirect Storage; this is not a NUMA-isolation experiment.

Source: [ServerlessLLM, §§7.1–7.2 and Figure 6](https://www.usenix.org/system/files/osdi24-fu.pdf).

## Vendor claims worth auditing, rather than promoting to universal rankings

### AWS's G7e/G6e SageMaker comparison

AWS reports Qwen3-32B BF16 with approximately 1,000 input/560 output tokens, native vLLM and prefix caching. At concurrency 32, its tables report **592 aggregate output tokens/s and 33.2-s p50** on ml.g7e.2xlarge versus **686 and 26.0 s** on the stated ml.g6e.12xlarge baseline. Its published hourly assumptions produce lower G7e dollars/token even while that configuration is slower.

Important caveat: the benchmark calls the G6e baseline **four L40S GPUs/TP4**, but the same post's later table lists **two GPUs** for that instance size. The exact container revision, actual cache-hit rates, and latency definition need confirmation for reproduction. Treat its price figures as historical SageMaker assumptions, not current EC2 quotes. This is a capacity/consolidation example with an unresolved configuration inconsistency, not a controlled per-GPU hardware comparison or G7e/H100 evidence. [AWS benchmark and configuration tables](https://aws.amazon.com/blogs/machine-learning/accelerate-generative-ai-inference-on-amazon-sagemaker-ai-with-g7e-instances/).

### A recent EFA/DeepEP result is specific to its stack

AWS reports **40% greater aggregate RL rollout throughput** after enabling DeepEP over EFA in a deployment with 48 P5en instances, 16 for training and 32 for inference, using a super-sparse MoE model. Its version matrix specifies CUDA 13.0, PyTorch 2.12+cu130, NCCL 2.31, EFA 1.49, DeepEP 2.0, SGLang 0.5.17, and Miles 0.1.0.

This is a provider-reported application result with substantial stack context. It does not isolate EFA versus TCP, establish a generic NCCL gain, or demonstrate DeepEP on G7e. The cited headline lacks the same granular request-length/quality/latency controls as a migration acceptance test. It is useful evidence that expert-exchange implementation can materially affect an H200-class deployment, subject to reproducing the full workload. [AWS, 25 September 2026, Figure 5 and version matrix](https://aws.amazon.com/blogs/machine-learning/scaling-moe-reinforcement-learning-on-amazon-eks-with-efa-and-deepep-with-40-more-throughput/).

## What an actual-path claim must show

Physical capability and software selection need separate evidence. NVIDIA documents NCCL P2P transport over either NVLink or PCIe when supported; physical topology alone does not prove successful GPU peer memory access. [NCCL GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html).

On AWS, the OFI plugin connects NCCL to libfabric. An EFA device being present is weaker evidence than the application's log showing the EFA provider. Even that log is not proof of every collective using an optimal GPU-direct path or of useful compute/communication overlap. Preserve per-rank logs and correlate them with timings/traces. [AWS OFI NCCL](https://github.com/aws/aws-ofi-nccl), [AWS EKS/Ray training transport checks](https://aws.github.io/deep-learning-containers/ray-train/deployment/eks/).

For inference libraries, custom collectives can have additional dtype, shape, rank-count, topology, and backend gates. Inspect the pinned application version and trace the operation rather than assuming every collective goes through NCCL or uses NVLink. [vLLM's legacy custom-all-reduce implementation](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py). These sources describe diagnostic requirements; this review did not measure any local communication path.

## Search coverage, exclusions, and missing evidence

This review searched primary papers, USENIX/PMLR proceedings, MLCommons submissions and rules, NVIDIA/AWS benchmark publications, and original framework issue reports. Targeted queries included G7e with H100/H200/B200, RTX PRO 6000 serving, prefill/decode, TP/PP, FSDP, NCCL transport, and quantization. It is a selected literature review, not an exhaustive systematic review.

- **No sufficiently controlled, directly measured AWS G7e-to-AWS H100/H200/B200/B300 portability study was established in this search.** G7e-only runs, RTX-PRO results on another server, and multi-platform vendor submissions exist; none justifies filling that gap with a fabricated conversion ratio.
- Peer-reviewed older studies are retained for mechanisms, with their dated stacks. They are not current rankings of serving frameworks.
- A B200/H200 preprint was reviewed but excluded from quantitative recommendations: [Microbenchmarking NVIDIA's Blackwell Architecture v1](https://arxiv.org/html/2512.02189v1). It says code could not yet be shared for double-blind review. Its Tables VIII and IX report substantially different throughput for apparently matching Mixtral-8x7B FP8 batch-32/2048-token descriptions without clearly separating phase/metric; the reviewed methodology does not resolve the comparison. Clarification and accessible artifacts are needed before using it as a capacity-planning input.
- No isolated, matched NUMA experiment on the requested EC2 families was established. CPU placement, host memory, storage, and data-loader behavior therefore remain validation items; no numerical NUMA benefit is assumed.
- No selected experiment demonstrates that a peak memory-bandwidth ratio, Tensor Core FLOP ratio, or maximum NIC rate becomes the same ratio in application throughput. A roofline is a bottleneck hypothesis to test with measured arithmetic intensity, achieved bandwidth, kernel behavior, and exposed communication.

## How to use the evidence in a migration decision

Run two comparisons when resources are authorized: first hold software, precision, workload, and parallelism as constant as feasible to identify regressions; then tune each destination subject to the same quality and service-level targets. The first answers compatibility and attribution questions. The second answers which deployment is useful.

Report all of these before declaring a winner:

1. Exact instance size, GPU count, driver, runtime/container digest, model revision, kernels, and topology.
2. Weight/activation/KV formats, quantization recipe, calibration, and quality-test outcome.
3. Input/output length distributions, seed, stop-token policy, cache state/hit rate, concurrency or arrival process, and warmup.
4. TP/PP/DP/EP/FSDP configuration, rank placement, actual selected transport, and exposed communication stalls.
5. Cold-start/readiness time; TTFT; per-request TPOT/ITL with clear aggregation; throughput; P95/P99; errors/OOMs; and SLO-compliant goodput.
6. For training: global/microbatch, accumulation, sequence length, optimizer, checkpointing, step time, convergence/quality, and strong versus weak scaling.
7. Cost at the same accepted workload and latency/quality target, including all participating instances and the utilization assumption. Dollars per generated token with no service-level or quality gate can reward an unusable configuration.

A published result should determine the next experiment, not substitute for it.


---

Next: [Explore analytical scenarios](../SIMULATION.md) or [run the target-hardware validation plan](10-validation-runbook.md).
