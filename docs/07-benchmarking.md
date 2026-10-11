# 07 · Benchmark and acceptance protocol

[Guide home](../README.md) · Reviewed 2026-10-11

This checklist is a proposed qualification procedure; it does not report measurements already performed.

### Freeze and record

- [ ] Exact Region/AZ, instance type/size, GPU inventory and usable memory, pricing model and date
- [ ] AMI, CPU ISA, driver/CUDA/NCCL/EFA/OFI versions, container digest and serving-engine commit/release
- [ ] Model/tokenizer revisions, license/access eligibility, precision and quantization recipe
- [ ] TP/PP/DP/EP, rank mapping, context/concurrency limits, graph settings, KV dtype, prefix-cache and speculative-decoding state
- [ ] Reproducible prompt set and request-arrival trace with no customer secrets committed to the repository

### Validate functionality and quality first

- [ ] Startup, checkpoint download/load, compilation and graph-capture success
- [ ] Short and longest-supported prompts, short and long outputs, stop behavior, streaming, structured output/tools and modalities as used
- [ ] Quality regression tests on representative tasks after changing GPU, engine, kernels or precision
- [ ] Correct tokenizer/chat template and output limits; no silent fallback precision or reduced context

### Measure the production request mix

- [ ] Separate cold-start from warm steady-state measurements
- [ ] Sweep request rate and concurrency; include bursts, long-context tails and mixed prompt/output lengths
- [ ] Measure client-observed TTFT, inter-token latency (ITL), end-to-end latency and their relevant tail percentiles
- [ ] Report request throughput and input/output token throughput separately; report failures, timeouts, rejected requests and output completeness
- [ ] Record all-GPU memory peaks, actual KV capacity, preemptions/recomputation, utilization, communication and CPU/RAM pressure
- [ ] State cache-hit assumptions; use both cache-cold and representative cache-warm cases rather than relying only on repeated identical prompts

TTFT includes the time until the first returned token; ITL measures gaps between subsequent tokens. Mean time per output token can hide occasional long stalls, so do not substitute it for an ITL-tail objective. vLLM's benchmark tooling can report latency percentiles and capacity under configured latency limits. [vLLM performance benchmarking](https://docs.vllm.ai/en/latest/benchmarking/dashboard/)

Define an SLO-passing request precisely, including required quality and successful completion. Measure **goodput**, the rate of completed requests satisfying the selected latency objectives, rather than maximizing raw tokens/second while a queue grows. [NVIDIA goodput definition](https://developer.download.nvidia.com/triton/vLLM-x-Triton-meetup-External.pdf)

### Compare cost and resilience

For a fixed workload and quality bar, compute a transparent experiment metric:

```text
cost_per_million_successful_output_tokens =
    total_service_cost_during_measurement
    / output_tokens_from_successful_SLO_passing_requests
    × 1,000,000
```

State currency, measurement window, whether idle/failover reserve is included, and included compute/storage/network/control-plane costs. Do not count failed or repeated retry output twice. Also report cost per successful request when output lengths vary materially; token cost alone can reward overly verbose responses. These are proposed accounting definitions, not quoted cloud prices.

- [ ] Sustain the target load without a growing queue, memory creep or rising failure rate
- [ ] Test replica restart, unhealthy-rank failure, drain/rollout and recovery time
- [ ] Check actual regional offerings, quota, purchasing/capacity arrangements and replacement capacity
- [ ] Preserve a known-good image/model/configuration and test rollback
- [ ] Save raw benchmark results, workload assumptions and rejected candidates so the decision is reproducible

**Acceptance statement template:** “On [exact instance and stack], [model revision and precision] supports [request mix/context/concurrency] at [quality bar] with [TTFT/ITL percentiles], [goodput], [failure rate], [peak per-rank memory], and [cost basis]. Results apply to this tested configuration.”

For executable diagnostic templates and an explicit P50/P95/P99, warmup, repetition, and scaling plan, use the [target-hardware validation runbook](10-validation-runbook.md).

## Repository templates

- Copy [benchmark-plan.md](../templates/benchmark-plan.md) for the experiment contract and acceptance decision.
- Use [benchmark-results.csv](../templates/benchmark-results.csv) for measurements. It has a header only; no synthetic measurements are presented as observations.
- Store prompts and raw traces in an approved location. Do not commit customer data, credentials, account identifiers, or private commercial terms.

For training, additionally compare validated loss/quality, tokens or samples processed per second, effective batch, step-time distributions, peak memory during optimizer steps and checkpointing, checkpoint size/time, recovery time, and cost to reach a fixed quality target. Comparing steps per second across different effective batches or convergence behavior can be misleading.

---

Next: [Use the estimator correctly](08-estimator-reference.md)
