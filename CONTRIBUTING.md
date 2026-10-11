# Updating this guide

1. Cite AWS, NVIDIA, the engine/framework, or the model publisher directly. Date checks. Separate a product launch announcement from current documentation and account-level observations.
2. Preserve the source's units. When sources disagree, document the discrepancy rather than choosing whichever number makes a model appear to fit.
3. Keep hardware specifications, arithmetic examples, hypotheses, and measured results distinct. Do not add benchmark claims without the stack, workload, raw measurements, and methodology.
4. Treat catalog additions as research candidates. Do not imply quota, capacity, pricing, model licensing, or runtime compatibility is verified.
5. Add or update tests when changing formulas or CLI behavior. Mark runnable estimator examples with the existing smoke-test comment convention so the test suite executes the documented command.
6. Run `python3 -m unittest discover -s tests -v` and `python3 -m compileall -q gpu_fit tests simulation.py` from the root. No packages or cloud access are needed.
7. Review the docs and `docs/SOURCES.md` together. Keep secrets, private account details, model access tokens, and customer prompt data out of commits.

The current tests validate arithmetic and repository consistency. They cannot establish GPU fit, real-world performance, current prices, or live AWS capacity. A license has not yet been selected for this repository.

When changing analytical simulations, regenerate the checked-in examples with `python3 simulation.py --output-prefix examples/simulation-results` and verify deterministic CSV/JSON output. Keep generic profiles uncalibrated unless a documented, authorized measurement set supports calibration; never silently turn them into instance predictions.
