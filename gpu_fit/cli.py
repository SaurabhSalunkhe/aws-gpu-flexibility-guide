"""Dependency-free command-line interface; never calls AWS or loads a model."""

import argparse
from decimal import Decimal, InvalidOperation
import json
import sys

from .model import InferenceConfig, TrainingConfig, estimate_inference, estimate_training


def parameter_count(value):
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("parameters must be a positive integer, e.g. 8e9") from exc
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise argparse.ArgumentTypeError("parameters must be a finite positive integer, e.g. 8e9")
    # Avoid pathological integer conversion and float-overflow arithmetic.
    if number > Decimal("1e18"):
        raise argparse.ArgumentTypeError("parameters must be <= 1e18")
    return int(number)


def build_parser():
    parser = argparse.ArgumentParser(description="Per-GPU memory arithmetic screen, never a guarantee of fit or performance.")
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("inference", "training"):
        sub = modes.add_parser(mode, help=f"{mode} memory screen")
        sub.add_argument("--parameters", type=parameter_count, required=True, help="total parameter count; e.g. 8e9, not 8 for an 8B model")
        sub.add_argument("--gpu-memory-gib", type=float, required=True, help="observed memory of one GPU in binary GiB")
        sub.add_argument("--tp", type=int, default=1, help="tensor-parallel ranks per pipeline stage (default: 1)")
        sub.add_argument("--pp", type=int, default=1, help="pipeline stages (default: 1)")
        sub.add_argument("--weight-bits", type=int, default=16)
        sub.add_argument("--headroom-fraction", type=float, default=0.15, help="fraction of each GPU reserved (default: 0.15)")
        sub.add_argument("--runtime-overhead-gib", type=float, default=0, help="extra allowance per rank; distinct from reserve (default: 0)")
        sub.add_argument("--weight-imbalance-factor", type=float, default=1, help="multiply per-rank model state to screen uneven partitions; must be >= 1")
        sub.add_argument("--json", action="store_true", help="emit full machine-readable inputs, components, and warnings")
        if mode == "inference":
            for arg in ("layers", "query-heads", "kv-heads", "head-dim"):
                sub.add_argument(f"--{arg}", type=int, required=True)
            sub.add_argument("--context-tokens", type=int, required=True, help="resident prompt + generated tokens per sequence; allocation rounding is not modeled")
            sub.add_argument("--concurrency", type=int, required=True, help="resident sequences per model replica, not fleet-wide requests or request rate")
            sub.add_argument("--kv-bits", type=int, default=16)
            sub.add_argument("--kv-layout", choices=("sharded", "replicated"), default="sharded")
            sub.add_argument("--weight-overhead-fraction", type=float, default=0, help="metadata/unquantized tensor allowance as fraction of ideal weight payload")
        else:
            sub.add_argument("--activation-gib", type=float, required=True, help="estimated or measured activations on the busiest rank; not divided by TP/PP/DP")
            sub.add_argument("--gradient-bits", type=int, default=16)
            sub.add_argument("--master-weight-bits", type=int, default=32)
            sub.add_argument("--optimizer-bits-per-parameter", type=int, default=64, help="both FP32 Adam moments total 64 bits (default)")
            sub.add_argument("--dp", type=int, default=1, help="data-parallel sharding group size (default: 1)")
            sub.add_argument("--zero-stage", type=int, choices=(0, 1, 2, 3), default=0)
    return parser


def main(argv=None):
    parser = build_parser()
    values = vars(parser.parse_args(argv))
    mode = values.pop("mode")
    output_json = values.pop("json")
    try:
        if mode == "inference":
            result = estimate_inference(InferenceConfig(**values))
        else:
            result = estimate_training(TrainingConfig(**values))
    except (ValueError, OverflowError) as exc:
        parser.error(str(exc))
    if output_json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(f"{mode.upper()} MEMORY SCREEN | {result['decision']}")
        print(f"GPUs in assumed layout: {result['gpu_count']}")
        print(f"Estimated per GPU: {result['estimated_per_gpu_gib']:.2f} GiB")
        print(f"Usable per-GPU budget after reserve: {result['usable_budget_per_gpu_gib']:.2f} GiB")
        print(f"Margin per GPU: {result['margin_per_gpu_gib']:.2f} GiB")
        print("Components per GPU:")
        for name, size in result["components_per_gpu_bytes"].items():
            print(f"  {name}: {size / (2**30):.2f} GiB")
        print(result["interpretation"])
        for warning in result["warnings"]:
            print(f"CAUTION: {warning}")
    # Exceeding a budget is a valid calculation, not a program failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
