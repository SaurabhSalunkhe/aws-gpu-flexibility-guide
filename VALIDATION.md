# Validation record

Reviewed 2026-10-11. This record distinguishes repository checks from target-hardware work.

## Completed locally

Environment: Python 3.12.14, CPU-only execution. No third-party Python packages, GPUs, AWS credentials, cloud API calls, or paid instances were needed.

- 63 standard-library unit/integration tests passed.
- All nine distinct documented memory scenarios were executed, including the repeated README example; exact component totals, budgets and decisions matched expected values.
- All 96 analytical scenarios were evaluated and checked-in CSV/JSON outputs were reproduced.
- Input validation, finite-value handling, units, GQA, TP/PP, ZeRO, collective latency/bandwidth, pipeline scheduling, overlap bounds, replica scaling, and scenario provenance were tested.
- Local Markdown file links and heading links, catalog structure, and source-ledger coverage were checked.
- Python modules compiled successfully.
- A separate technical review checked the instance mappings/source conflicts, memory and parallelism arithmetic, diagnostic templates, scoped study claims, and analytical model assumptions.

Reproduce from the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q gpu_fit tests simulation.py
python3 simulation.py --output-prefix examples/simulation-results
```

The final command replaces the generated example CSV/JSON. Their metadata and every row identify them as uncalibrated analytical results. They are separate from the empty real-measurement template at [templates/benchmark-results.csv](templates/benchmark-results.csv).

## Configured, but not established by this local run

GitHub Actions is configured for Python 3.10, 3.12 and 3.13 with read-only repository permissions. Consult the actual run for the exact published commit before claiming CI passed. Local Python 3.12 testing alone does not verify the other versions or hosted Actions execution.

## Not performed

- Loading or executing a model on G7e, H100, H200, B200, B300, or any other GPU
- Running the topology, P2P, NCCL, EFA, profiler, or end-to-end commands in the target-hardware runbook
- Verifying customer-account quotas, available capacity, or current commercial pricing
- Measuring a model-quality score, TTFT, TPOT/ITL, service throughput, scaling efficiency, recovery time, or cost on AWS
- Independently rerunning the external papers or vendor/practitioner benchmark submissions

Published measurements retain their original conditions in [the evidence review](docs/11-published-evidence.md). Offline arithmetic and simulations establish only their stated calculations. The [validation runbook](docs/10-validation-runbook.md) describes the remaining real-hardware qualification work.
