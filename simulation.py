#!/usr/bin/env python3
"""CPU-only analytical sensitivity model. NOT a GPU emulator or benchmark.

All example inputs are invented scenario assumptions, not calibrated hardware
profiles. See SIMULATION.md for equations, units, exclusions, and calibration.
Python 3.10+, standard library only. This module never calls a GPU, cloud API,
network endpoint, subprocess, or package installer.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

MODEL_VERSION = "1.0"
EVIDENCE = "uncalibrated_analytical_scenario_not_measured_performance"
DISCLAIMER = ("Analytical model, NOT a GPU emulator, benchmark, or real-instance "
              "performance prediction. Inputs are assumptions; inspect provenance.")


def number(name: str, value: Any, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, not {value!r}")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def integer(name: str, value: Any, *, zero: bool = False) -> int:
    number(name, value, zero=zero)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def fraction(name: str, value: Any, *, zero: bool = True) -> float:
    number(name, value, zero=zero)
    if value > 1:
        raise ValueError(f"{name} must be at most 1")
    return float(value)


def keys(data: dict, allowed: set[str], required: set[str], context: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be an object")
    if set(data) - allowed:
        raise ValueError(f"{context}: unknown keys {sorted(set(data) - allowed)}")
    if required - set(data):
        raise ValueError(f"{context}: missing keys {sorted(required - set(data))}")


@dataclass(frozen=True)
class Device:
    peak_TFLOPs: float
    memory_peak_GBps: float
    compute_utilization: float
    memory_utilization: float
    provenance: str

    def __post_init__(self) -> None:
        number("peak_TFLOPs", self.peak_TFLOPs)
        number("memory_peak_GBps", self.memory_peak_GBps)
        fraction("compute_utilization", self.compute_utilization, zero=False)
        fraction("memory_utilization", self.memory_utilization, zero=False)
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("device provenance must be a nonempty string")

    @property
    def compute_FLOPs(self) -> float:
        return self.peak_TFLOPs * self.compute_utilization * 1e12

    @property
    def memory_bytes_s(self) -> float:
        return self.memory_peak_GBps * self.memory_utilization * 1e9


@dataclass(frozen=True)
class Transport:
    directional_rank_peak_GBps: float
    utilization: float
    latency_us: float
    sharing_factor: float
    actual_path: str
    provenance: str
    advertised_description: str = "Not supplied; never used as an equation input"

    def __post_init__(self) -> None:
        number("directional_rank_peak_GBps", self.directional_rank_peak_GBps)
        fraction("utilization", self.utilization, zero=False)
        number("latency_us", self.latency_us, zero=True)
        number("sharing_factor", self.sharing_factor)
        if self.sharing_factor < 1:
            raise ValueError("sharing_factor must be at least 1")
        for name in ("actual_path", "provenance", "advertised_description"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"transport {name} must be a nonempty string")

    @property
    def bytes_s(self) -> float:
        return self.directional_rank_peak_GBps * 1e9 * self.utilization / self.sharing_factor


def roofline(flops: float, memory_bytes: float, device: Device) -> dict:
    """Idealized overlap of arithmetic and device-memory work, never their sum."""
    number("flops", flops, zero=True)
    number("memory_bytes", memory_bytes, zero=True)
    compute = flops / device.compute_FLOPs
    memory = memory_bytes / device.memory_bytes_s
    return {
        "compute_s": compute, "memory_s": memory, "local_s": max(compute, memory),
        "local_regime": "balanced" if compute == memory else ("compute" if compute > memory else "memory"),
        "arithmetic_intensity_FLOP_per_byte": flops / memory_bytes if memory_bytes else None,
        "ridge_FLOP_per_byte": device.compute_FLOPs / device.memory_bytes_s,
    }


def ring_all_reduce(payload_bytes: float, ranks: int, transport: Transport) -> dict:
    """Flat homogeneous ring with 2(ranks-1) phases; payload is bytes PER rank.

    Transport bandwidth is effective directional link/bus bandwidth, NOT algbw.
    Reduction compute, algorithm switches, hierarchy, and routing are excluded.
    A zero-byte payload is defined as no operation, hence no launch latency.
    """
    number("payload_bytes", payload_bytes, zero=True)
    integer("ranks", ranks)
    factor = 2 * (ranks - 1) / ranks
    if ranks == 1 or payload_bytes == 0:
        return {"latency_s": 0.0, "bandwidth_s": 0.0, "total_s": 0.0,
                "wire_bytes_per_rank": 0.0, "ring_byte_factor": factor}
    latency = 2 * (ranks - 1) * transport.latency_us * 1e-6
    wire = factor * payload_bytes
    bandwidth = wire / transport.bytes_s
    return {"latency_s": latency, "bandwidth_s": bandwidth,
            "total_s": latency + bandwidth, "wire_bytes_per_rank": wire,
            "ring_byte_factor": factor}


def overlap_time(local_s: float, communication_s: float, overlap: float,
                 eligible_local_fraction: float = 1.0) -> dict:
    """Spend a single overlap budget; full overlap never goes below max(C,K)."""
    number("local_s", local_s, zero=True)
    number("communication_s", communication_s, zero=True)
    fraction("overlap", overlap)
    fraction("eligible_local_fraction", eligible_local_fraction)
    hidden = overlap * min(local_s * eligible_local_fraction, communication_s)
    exposed = communication_s - hidden
    return {"total_s": local_s + exposed, "hidden_s": hidden, "exposed_s": exposed}


def pipeline_schedule(stage_s: list[float], microbatches: int,
                      transfer_s: list[float] | None = None) -> dict:
    """Deterministic forward-only FIFO pipeline with one serialized link per edge.

    A stage processes one microbatch at a time; a link transfers one at a time.
    Links may overlap stage compute and other links. TP work belongs in stage_s.
    This is not a training/1F1B scheduler or a CUDA execution simulator.
    """
    integer("microbatches", microbatches)
    if not stage_s:
        raise ValueError("pipeline must have at least one stage")
    for value in stage_s:
        number("stage_s", value, zero=True)
    if transfer_s is None:
        transfer_s = [0.0] * (len(stage_s) - 1)
    if len(transfer_s) != len(stage_s) - 1:
        raise ValueError("one transfer time is required per pipeline edge")
    for value in transfer_s:
        number("transfer_s", value, zero=True)
    stages_ready = [0.0] * len(stage_s)
    links_ready = [0.0] * len(transfer_s)
    first = None
    for _ in range(microbatches):
        arrival = 0.0
        for stage, duration in enumerate(stage_s):
            finish = max(arrival, stages_ready[stage]) + duration
            stages_ready[stage] = finish
            if stage < len(transfer_s):
                arrival = max(finish, links_ready[stage]) + transfer_s[stage]
                links_ready[stage] = arrival
        if first is None:
            first = finish
    total = stages_ready[-1]
    util = microbatches * sum(stage_s) / (len(stage_s) * total) if total else 0.0
    return {"total_s": total, "first_microbatch_s": first,
            "mean_stage_service_fraction": util,
            "ideal_balanced_zero_link_bubble_fraction": (len(stage_s) - 1) / (microbatches + len(stage_s) - 1)}


MODEL_FIELDS = {"parameters", "layers", "hidden_size", "kv_heads", "head_dim",
                "weight_bytes", "kv_bytes", "activation_bytes", "provenance"}
INFERENCE_FIELDS = {"id", "kind", "group", "description", "device", "model", "mode",
    "batch", "context_tokens", "resident_sequences", "tp", "pp", "replicas", "microbatches",
    "tp_transport", "pp_transport", "tp_overlap", "tp_collectives_per_layer",
    "weight_read_passes", "kv_read_passes", "other_activation_passes",
    "activation_tp_sharded", "kv_tp_sharded", "pp_payload_tp_sharded"}
INFERENCE_REQUIRED = {"id", "kind", "device", "model", "mode", "batch", "context_tokens",
    "resident_sequences", "tp", "pp", "replicas", "microbatches", "tp_transport", "pp_transport",
    "tp_overlap", "tp_collectives_per_layer", "weight_read_passes", "kv_read_passes",
    "other_activation_passes", "activation_tp_sharded", "kv_tp_sharded", "pp_payload_tp_sharded"}


def inference_case(case: dict, models: dict, devices: dict, transports: dict) -> dict:
    keys(case, INFERENCE_FIELDS, INFERENCE_REQUIRED, "inference case")
    model = models[case["model"]]
    keys(model, MODEL_FIELDS, MODEL_FIELDS, "model")
    for name in MODEL_FIELDS - {"provenance"}:
        (integer if name in {"layers", "hidden_size", "kv_heads", "head_dim"} else number)(name, model[name])
    if not isinstance(model["provenance"], str) or not model["provenance"].strip():
        raise ValueError("model provenance must be a nonempty string")
    for name in ("batch", "context_tokens", "resident_sequences", "tp", "pp", "replicas", "microbatches"):
        integer(name, case[name])
    for name in ("activation_tp_sharded", "kv_tp_sharded", "pp_payload_tp_sharded"):
        if type(case[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    for name in ("weight_read_passes", "kv_read_passes"):
        number(name, case[name])
        if case[name] < 1:
            raise ValueError(f"{name} must be at least 1")
    number("other_activation_passes", case["other_activation_passes"], zero=True)
    integer("tp_collectives_per_layer", case["tp_collectives_per_layer"], zero=True)
    fraction("tp_overlap", case["tp_overlap"])
    if case["mode"] not in {"prefill", "decode"}:
        raise ValueError("mode must be prefill or decode")
    P, L, H = model["parameters"], model["layers"], model["hidden_size"]
    K, D = model["kv_heads"], model["head_dim"]
    b, S, t, p, m = (case[x] for x in ("batch", "context_tokens", "tp", "pp", "microbatches"))
    if L % p or H % t or H % D:
        raise ValueError("balanced proxy requires layers % pp = hidden_size % tp = hidden_size % head_dim = 0")
    if (H // D) % t:
        raise ValueError("query heads (hidden_size / head_dim) must divide evenly across tp")
    if K > H // D or (H // D) % K:
        raise ValueError("kv_heads must divide query heads (hidden_size / head_dim)")
    if case["kv_tp_sharded"] and K % t:
        raise ValueError("ideal KV sharding requires kv_heads divisible by tp; set kv_tp_sharded=false to replicate")
    if case["resident_sequences"] < b * m:
        raise ValueError("resident_sequences must cover batch * microbatches per replica")
    tokens = b * S if case["mode"] == "prefill" else b
    pairs = b * S * (S + 1) / 2 if case["mode"] == "prefill" else b * (S + 1)
    # 2 FLOPs per multiply-add; QK and AV together cost 4 * hidden * query-key pairs.
    flops = (2 * P * tokens + 4 * L * H * pairs) / (t * p)
    kv_shards = t if case["kv_tp_sharded"] else 1
    activation_shards = t if case["activation_tp_sharded"] else 1
    weights = P * model["weight_bytes"] / (t * p)
    weight_traffic = weights * case["weight_read_passes"]
    kv_unit = 2 * (L / p) * (K / kv_shards) * D * model["kv_bytes"]
    kv_read = kv_unit * b * S * case["kv_read_passes"] if case["mode"] == "decode" else 0.0
    kv_write = kv_unit * tokens
    other = (case["other_activation_passes"] * tokens * H * (L / p)
             * model["activation_bytes"] / activation_shards)
    traffic = weight_traffic + kv_read + kv_write + other
    roof = roofline(flops, traffic, devices[case["device"]])
    tp_link = transports[case["tp_transport"]]
    pp_link = transports[case["pp_transport"]]
    payload = tokens * H * model["activation_bytes"]
    collective = ring_all_reduce(payload, t, tp_link)
    collective_count = case["tp_collectives_per_layer"] * L / p if t > 1 else 0
    comm = collective["total_s"] * collective_count
    stage = overlap_time(roof["local_s"], comm, case["tp_overlap"])
    pp_payload = payload / (t if case["pp_payload_tp_sharded"] else 1)
    pp_transfer = pp_link.latency_us * 1e-6 + pp_payload / pp_link.bytes_s if p > 1 else 0.0
    pipeline = pipeline_schedule([stage["total_s"]] * p, m, [pp_transfer] * (p - 1))
    local_pipeline = pipeline_schedule([roof["local_s"]] * p, m)
    total = pipeline["total_s"]
    total_tokens = tokens * m * case["replicas"]
    return {
        "analytical_total_ms": total * 1e3,
        "analytical_first_microbatch_ms": pipeline["first_microbatch_s"] * 1e3,
        "analytical_tokens_per_s": total_tokens / total,
        "token_unit": "input_prefill_tokens" if case["mode"] == "prefill" else "one_output_token_per_active_sequence",
        "gpu_count": t * p * case["replicas"],
        "batch_per_microbatch": b, "context_tokens": S,
        "resident_sequences_per_replica": case["resident_sequences"],
        "tp": t, "pp": p, "replicas": case["replicas"], "microbatches": m,
        "analytical_stage_local_ms": roof["local_s"] * 1e3,
        "analytical_stage_compute_ms": roof["compute_s"] * 1e3,
        "analytical_stage_memory_ms": roof["memory_s"] * 1e3,
        "local_regime": roof["local_regime"],
        "arithmetic_intensity_FLOP_per_byte": roof["arithmetic_intensity_FLOP_per_byte"],
        "analytical_flops_per_stage_rank_microbatch": flops,
        "analytical_memory_bytes_per_stage_rank_microbatch": traffic,
        "analytical_weight_traffic_bytes": weight_traffic,
        "analytical_kv_read_bytes": kv_read, "analytical_kv_write_bytes": kv_write,
        "analytical_other_activation_bytes": other,
        "analytical_weights_GiB_per_rank": weights / 2**30,
        "analytical_resident_KV_GiB_per_rank": kv_unit * case["resident_sequences"] * (S + (case["mode"] == "decode")) / 2**30,
        "analytical_tp_raw_ms_per_stage_microbatch": comm * 1e3,
        "analytical_tp_hidden_ms_per_stage_microbatch": stage["hidden_s"] * 1e3,
        "analytical_tp_exposed_ms_per_stage_microbatch": stage["exposed_s"] * 1e3,
        "tp_communication_dominates_local": comm > roof["local_s"],
        "analytical_tp_latency_ms_per_stage_microbatch": collective["latency_s"] * collective_count * 1e3,
        "analytical_tp_bandwidth_ms_per_stage_microbatch": collective["bandwidth_s"] * collective_count * 1e3,
        "analytical_pp_transfer_ms_per_edge_microbatch": pp_transfer * 1e3,
        "analytical_communication_exposure_ms": (total - local_pipeline["total_s"]) * 1e3,
        "analytical_stage_service_fraction": pipeline["mean_stage_service_fraction"],
        "ideal_balanced_zero_link_bubble_fraction": pipeline["ideal_balanced_zero_link_bubble_fraction"],
        "inference_replica_gradient_sync_ms": 0.0,
        "tp_effective_directional_rank_GBps": tp_link.bytes_s / 1e9,
        "tp_actual_path_assumption": tp_link.actual_path,
        "pp_actual_path_assumption": pp_link.actual_path,
    }


def training_dp_case(case: dict, devices: dict, transports: dict) -> dict:
    required = {"id", "kind", "device", "transport", "replicas", "microbatches", "local_flops_per_microbatch",
        "local_memory_bytes_per_microbatch", "gradient_bytes_per_rank", "gradient_buckets", "overlap",
        "eligible_final_microbatch_fraction", "samples_per_microbatch", "optimizer_s"}
    keys(case, required | {"group", "description"}, required, "training_dp case")
    for name in ("replicas", "microbatches", "gradient_buckets", "samples_per_microbatch"):
        integer(name, case[name])
    number("gradient_bytes_per_rank", case["gradient_bytes_per_rank"], zero=True)
    number("optimizer_s", case["optimizer_s"], zero=True)
    roof = roofline(case["local_flops_per_microbatch"], case["local_memory_bytes_per_microbatch"], devices[case["device"]])
    local = roof["local_s"] * case["microbatches"]
    link = transports[case["transport"]]
    collective = ring_all_reduce(case["gradient_bytes_per_rank"] / case["gradient_buckets"], case["replicas"], link)
    comm = collective["total_s"] * case["gradient_buckets"]
    fraction("eligible_final_microbatch_fraction", case["eligible_final_microbatch_fraction"])
    # Accumulate with no_sync until the final microbatch. Earlier work cannot hide
    # a sync that has not started; only the final backward window is eligible.
    eligible = case["eligible_final_microbatch_fraction"] / case["microbatches"]
    overlap = overlap_time(local, comm, case["overlap"], eligible)
    total = overlap["total_s"] + case["optimizer_s"]
    if total == 0:
        raise ValueError("training step must have positive modeled duration")
    return {"analytical_total_ms": total * 1e3,
        "analytical_local_ms": local * 1e3, "analytical_dp_raw_ms": comm * 1e3,
        "analytical_dp_hidden_ms": overlap["hidden_s"] * 1e3,
        "analytical_dp_exposed_ms": overlap["exposed_s"] * 1e3,
        "analytical_dp_overlap_window_ms": local * eligible * 1e3,
        "analytical_samples_per_s": case["samples_per_microbatch"] * case["microbatches"] * case["replicas"] / total,
        "dp_communication_dominates_local": comm > local, "local_regime": roof["local_regime"],
        "replicas": case["replicas"], "microbatches": case["microbatches"],
        "gpu_count": case["replicas"],
        "analytical_dp_latency_ms": collective["latency_s"] * case["gradient_buckets"] * 1e3,
        "analytical_dp_bandwidth_ms": collective["bandwidth_s"] * case["gradient_buckets"] * 1e3,
        "effective_directional_rank_GBps": link.bytes_s / 1e9,
        "actual_path_assumption": link.actual_path}


def collective_case(case: dict, transports: dict) -> dict:
    required = {"id", "kind", "transport", "ranks", "nodes", "ranks_per_node", "payload_bytes"}
    keys(case, required | {"group", "description"}, required, "collective case")
    integer("nodes", case["nodes"])
    integer("ranks_per_node", case["ranks_per_node"])
    if case["nodes"] * case["ranks_per_node"] != case["ranks"]:
        raise ValueError("nodes * ranks_per_node must equal ranks")
    link = transports[case["transport"]]
    result = ring_all_reduce(case["payload_bytes"], case["ranks"], link)
    return {"analytical_total_ms": result["total_s"] * 1e3,
        "analytical_collective_latency_ms": result["latency_s"] * 1e3,
        "analytical_collective_bandwidth_ms": result["bandwidth_s"] * 1e3,
        "collective_regime": "latency" if result["latency_s"] > result["bandwidth_s"] else "bandwidth",
        "analytical_wire_bytes_per_rank": result["wire_bytes_per_rank"],
        "ring_byte_factor": result["ring_byte_factor"], "ranks": case["ranks"],
        "nodes": case["nodes"], "ranks_per_node": case["ranks_per_node"],
        "effective_directional_rank_GBps": link.bytes_s / 1e9,
        "actual_path_assumption": link.actual_path}


def finite_outputs(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite result: reduce numeric input magnitude")
    if isinstance(value, dict):
        for item in value.values():
            finite_outputs(item)
    if isinstance(value, list):
        for item in value:
            finite_outputs(item)


def run_scenarios(data: dict, only: str | None = None) -> list[dict]:
    required = {"schema_version", "evidence", "assumptions", "devices", "transports", "models", "cases"}
    keys(data, required, required, "scenario file")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["evidence"] != EVIDENCE:
        raise ValueError(f"schema_version must be 1 and evidence must be {EVIDENCE!r}")
    for section in ("devices", "transports", "models"):
        if not isinstance(data[section], dict) or not all(isinstance(v, dict) for v in data[section].values()):
            raise ValueError(f"{section} must be an object of objects")
    if not isinstance(data["cases"], list) or not all(isinstance(v, dict) for v in data["cases"]):
        raise ValueError("cases must be a list of objects")
    if not isinstance(data["assumptions"], list) or not data["assumptions"] or not all(isinstance(a, str) and a for a in data["assumptions"]):
        raise ValueError("assumptions must be a nonempty list")
    devices = {name: Device(**values) for name, values in data["devices"].items()}
    transports = {name: Transport(**values) for name, values in data["transports"].items()}
    results, seen = [], set()
    for case in data["cases"]:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("scenario IDs must be unique nonempty strings")
        seen.add(case_id)
        if only and case_id != only:
            continue
        try:
            if case["kind"] == "inference":
                result = inference_case(case, data["models"], devices, transports)
            elif case["kind"] == "training_dp":
                result = training_dp_case(case, devices, transports)
            elif case["kind"] == "collective":
                result = collective_case(case, transports)
            else:
                raise ValueError(f"unknown kind {case['kind']!r}")
        except (KeyError, TypeError, OverflowError, ValueError) as exc:
            raise ValueError(f"scenario {case_id}: {exc}") from exc
        result = {"id": case_id, "kind": case["kind"], "group": case.get("group", ""),
                  "evidence": EVIDENCE, "layout_feasibility": "not_evaluated_require_separate_memory_and_stack_checks",
                  "latency_scope": "idealized_service_work_only_no_queueing_or_SLO_prediction", **result}
        finite_outputs(result)
        results.append(result)
    if not results:
        raise ValueError("no matching scenarios")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=DISCLAIMER)
    parser.add_argument("--scenarios", type=Path, default=Path(__file__).parent / "examples/simulation_scenarios.json")
    parser.add_argument("--output-prefix", type=Path, help="Write PREFIX.csv and PREFIX.json, replacing existing files")
    parser.add_argument("--only", help="Run one scenario ID")
    parser.add_argument("--json", action="store_true", help="Print complete input/provenance/results JSON to stdout")
    args = parser.parse_args(argv)
    try:
        raw = args.scenarios.read_bytes()
        data = json.loads(raw)
        results = run_scenarios(data, args.only)
        report = {"model_version": MODEL_VERSION, "disclaimer": DISCLAIMER, "evidence": EVIDENCE,
                  "scenario_sha256": hashlib.sha256(raw).hexdigest(), "inputs": data, "results": results}
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output_prefix:
            args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
            args.output_prefix.with_suffix(".json").write_text(encoded)
            fields = list(dict.fromkeys(key for row in results for key in row))
            with args.output_prefix.with_suffix(".csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(results)
        if args.json:
            print(encoded, end="")
        else:
            print(DISCLAIMER)
            print(f"{len(results)} analytical scenarios evaluated on CPU. No GPU execution or cloud calls.")
            for row in results:
                print(f"{row['id']}: modeled {row['analytical_total_ms']:.6g} ms")
            if args.output_prefix:
                print(f"Wrote {args.output_prefix.with_suffix('.csv')} and {args.output_prefix.with_suffix('.json')}")
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
