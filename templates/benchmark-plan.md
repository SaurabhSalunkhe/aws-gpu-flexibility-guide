# GPU alternative qualification plan

This is an unfilled experiment template. No results are implied.

## Decision and acceptance gates

- Workload / owner:
- Baseline profile:
- Candidate profile:
- Why this alternative is needed:
- Allowed changes (model, precision, context, API behavior):
- Quality acceptance set and threshold:
- Maximum context (prompt + output):
- Target request rate, resident concurrency, and burst behavior:
- TTFT / ITL / end-to-end latency thresholds and percentiles:
- Failure, rejection, and timeout thresholds:
- Minimum SLO-passing goodput:
- Maximum cost under a specified cost basis:
- Startup / recovery / rollback targets:
- Stop conditions, experiment budget, and approval owner:

## Immutable experiment record

- Date, time window, Region/AZ, exact instance size, and purchase/capacity mechanism:
- GPU model/count, measured GiB per GPU, topology and NUMA/rank mapping:
- Host CPU architecture, RAM, network/EFA and storage:
- AMI ID and container digest:
- Driver, CUDA, framework, engine release/commit:
- Attention/GEMM/quantization backend and compiled kernel targets:
- NCCL, Libfabric, EFA installer, AWS OFI NCCL versions and verified transport:
- Model/tokenizer revisions, chat template, and access/license eligibility:
- Weight/compute/KV precision, quantization format, group size and scales:
- TP / PP / DP / EP / context parallelism and placement:
- Cache, graph capture, speculative decoding, adapters and offload settings:
- Workload generator version, random seed, trace location and checksum:
- Pricing date/currency and included cost categories:

## Test matrix

- [ ] Exact model load, short generation, and application-feature smoke tests
- [ ] Quality equivalence before measuring speed
- [ ] Cold startup, compilation/graph capture, and warmed steady state separately
- [ ] Representative prompt/output lengths, maximum-context paths, and mixed tails
- [ ] Cache-cold and representative cache-warm workloads; disclose hit rates
- [ ] Request-rate and concurrency sweep through target load plus burst conditions
- [ ] Peak memory on every rank, including prefill, decode, optimizer/checkpoint where applicable
- [ ] Collective tests on the actual rank/network topology
- [ ] Sustained load with queue depth, failure rate, preemptions and memory trend
- [ ] Drain, rank/replica failure, restart, recovery, and rollback
- [ ] Offering, quota, purchase path, and replacement-capacity checks

## Training additions, if applicable

- Optimizer/state precision and sharding, microbatch and effective batch:
- Sequence length, activation checkpointing and schedule:
- Tokens/s or samples/s, step-time tails, loss/quality trajectory:
- Peak forward/backward/optimizer-step memory:
- Checkpoint save/load size, duration, and memory peak:
- Cost and time to a fixed quality target:

## Results and decision

Use [benchmark-results.csv](benchmark-results.csv) for measurements. Keep raw traces and logs in an approved location; never commit customer prompts, credentials, private account details, or negotiated commercial terms.

- Raw evidence location:
- Quality result:
- Worst-rank memory peak and remaining reserve:
- SLO-passing throughput and latency percentiles:
- Cost per successful request and per million successful output tokens:
- Operational and capacity findings:
- Rejected configurations and reasons:
- Decision: untested / rejected / more evidence required / qualified
- Qualified workload boundaries:
- Canary plan, rollback trigger and known-good profile:
- Reviewer and date:

Acceptance statement: On [exact instance and stack], [model revision and precision] supports [request mix/context/concurrency] at [quality bar] with [latency percentiles], [goodput], [failure rate], [peak per-rank memory], and [cost basis]. Results apply only to this tested configuration.
