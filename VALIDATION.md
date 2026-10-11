# Validation record

Reviewed 2026-10-11. This record distinguishes repository checks from target-hardware work.

## Completed locally

Environment: Python 3.12.14, CPU-only execution. No third-party Python packages, GPUs, AWS credentials, cloud API calls, or paid instances were needed.

- 103 standard-library unit/integration tests passed: 22 memory-formula, 7 CLI, 6 documentation/catalog, 28 analytical-simulation, and 40 codebase-assessor tests.
- All nine distinct documented memory scenarios were executed, including the repeated README example; exact component totals, budgets and decisions matched expected values.
- All 96 analytical scenarios were evaluated and checked-in CSV/JSON outputs were reproduced.
- Input validation, finite-value handling, units, GQA, TP/PP, ZeRO, collective latency/bandwidth, pipeline scheduling, overlap bounds, replica scaling, and scenario provenance were tested.
- Local Markdown file links and heading links, catalog structure, and source-ledger coverage were checked.
- The assessor’s synthetic fixture produced 26 deterministic static findings; its JSON/Markdown reports reproduced exactly. Read-only/no-execution, privacy, symlink/hardlink, ignore-policy, bounded-coverage and output-safety cases passed. No customer repository was supplied or assessed.
- Python modules compiled successfully.
- A separate technical review checked the instance mappings/source conflicts, memory and parallelism arithmetic, diagnostic templates, scoped study claims, analytical model assumptions, and the codebase assessor’s static-analysis/security boundaries.

Reproduce from the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q gpu_fit gpu_assess tests simulation.py
python3 simulation.py --output-prefix examples/simulation-results
```

The final command replaces the generated example CSV/JSON. Their metadata and every row identify them as uncalibrated analytical results. They are separate from the empty real-measurement template at [templates/benchmark-results.csv](templates/benchmark-results.csv).

## Hosted CI and version scope

The original 63-test baseline passed GitHub Actions on Python 3.10, 3.12 and 3.13 at commit `ef285696e474f03899b5dd41de9836361b7e73c1`. The expanded 103-test tree was verified locally on Python 3.12.14; consult the exact published commit’s Actions run for hosted results. The workflow retains read-only repository permissions and the same three-version matrix. Local testing alone does not establish the other Python versions or hosted execution.

## Not performed

- Assessing a real customer codebase, resolving its deployed dependencies/artifacts, or certifying its portability

- Loading or executing a model on G7e, H100, H200, B200, B300, or any other GPU
- Running the topology, P2P, NCCL, EFA, profiler, or end-to-end commands in the target-hardware runbook
- Verifying customer-account quotas, available capacity, or current commercial pricing
- Measuring a model-quality score, TTFT, TPOT/ITL, service throughput, scaling efficiency, recovery time, or cost on AWS
- Independently rerunning the external papers or vendor/practitioner benchmark submissions

Published measurements retain their original conditions in [the evidence review](docs/11-published-evidence.md). Offline arithmetic and simulations establish only their stated calculations. The [validation runbook](docs/10-validation-runbook.md) describes the remaining real-hardware qualification work.
