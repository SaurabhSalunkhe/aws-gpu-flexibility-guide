from dataclasses import replace
import math
import unittest

from gpu_fit.model import GIB, InferenceConfig, TrainingConfig, estimate_inference, estimate_training


def inference(**changes):
    base = InferenceConfig(
        parameters=8_000_000_000, gpu_memory_gib=80,
        layers=32, query_heads=32, kv_heads=8, head_dim=128,
        context_tokens=8192, concurrency=8, runtime_overhead_gib=2,
    )
    return replace(base, **changes)


def training(**changes):
    return replace(TrainingConfig(parameters=8_000_000_000, gpu_memory_gib=80, activation_gib=8), **changes)


class InferenceTests(unittest.TestCase):
    def test_weights_are_decimal_parameters_not_binary_billions(self):
        result = estimate_inference(inference())
        self.assertEqual(result["weights_total_bytes_before_sharding"], 16_000_000_000)
        self.assertAlmostEqual(result["components_per_gpu_bytes"]["weights"] / GIB, 14.901161193847656)

    def test_gqa_kv_uses_kv_heads(self):
        result = estimate_inference(inference())
        self.assertEqual(result["kv_bytes_per_cached_token_before_sharding"], 128 * 1024)
        self.assertEqual(result["kv_total_bytes_before_sharding"], 8 * GIB)

    def test_mha_is_four_times_example_gqa(self):
        result = estimate_inference(inference(kv_heads=32))
        self.assertEqual(result["kv_total_bytes_before_sharding"], 32 * GIB)

    def test_70b_shape_kv(self):
        result = estimate_inference(inference(parameters=70_000_000_000, layers=80, query_heads=64))
        self.assertEqual(result["kv_bytes_per_cached_token_before_sharding"], 320 * 1024)
        self.assertEqual(result["kv_total_bytes_before_sharding"], 20 * GIB)

    def test_context_and_concurrency_scale_independently(self):
        base = estimate_inference(inference())["kv_total_bytes_before_sharding"]
        self.assertEqual(estimate_inference(inference(context_tokens=16384))["kv_total_bytes_before_sharding"], 2 * base)
        self.assertEqual(estimate_inference(inference(concurrency=16))["kv_total_bytes_before_sharding"], 2 * base)

    def test_quantized_weights_do_not_quantize_kv(self):
        base = estimate_inference(inference())
        quant = estimate_inference(inference(weight_bits=4, weight_overhead_fraction=0.12))
        self.assertAlmostEqual(quant["weights_total_bytes_before_sharding"], 4_480_000_000)
        self.assertEqual(quant["kv_total_bytes_before_sharding"], base["kv_total_bytes_before_sharding"])

    def test_explicit_kv_precision(self):
        result = estimate_inference(inference(kv_bits=8))
        self.assertEqual(result["kv_total_bytes_before_sharding"], 4 * GIB)
        self.assertTrue(any("Quantized KV" in item for item in result["warnings"]))

    def test_tensor_and_pipeline_partition_components(self):
        base = estimate_inference(inference())["components_per_gpu_bytes"]
        result = estimate_inference(inference(tp=2, pp=2))
        for key in ("weights", "kv_cache"):
            self.assertEqual(result["components_per_gpu_bytes"][key], base[key] / 4)
        self.assertEqual(result["components_per_gpu_bytes"]["runtime_overhead"], 2 * GIB)
        self.assertEqual(result["gpu_count"], 4)
        self.assertEqual(result["layers_per_pipeline_stage"], 16)

    def test_replicated_kv_does_not_divide_by_tp(self):
        result = estimate_inference(inference(tp=16, pp=2, kv_layout="replicated"))
        self.assertEqual(result["components_per_gpu_bytes"]["kv_cache"], 4 * GIB)

    def test_uneven_weight_allowance(self):
        base = estimate_inference(inference())
        result = estimate_inference(inference(weight_imbalance_factor=1.2))
        self.assertEqual(result["components_per_gpu_bytes"]["weights"], base["components_per_gpu_bytes"]["weights"] * 1.2)
        self.assertEqual(result["components_per_gpu_bytes"]["kv_cache"], base["components_per_gpu_bytes"]["kv_cache"])

    def test_headroom_is_reserve_not_added_to_estimate(self):
        result = estimate_inference(inference(headroom_fraction=0.25))
        self.assertEqual(result["usable_budget_per_gpu_gib"], 60)
        self.assertAlmostEqual(result["estimated_per_gpu_gib"], 24.901161193847656)

    def test_exact_budget_boundary(self):
        config = inference(parameters=GIB // 2, layers=1, query_heads=1, kv_heads=1, head_dim=1,
                           context_tokens=1, concurrency=1, runtime_overhead_gib=0, headroom_fraction=0)
        exact = 1 + 4 / GIB
        self.assertEqual(estimate_inference(replace(config, gpu_memory_gib=exact))["decision"], "within_budget_screen")
        self.assertEqual(estimate_inference(replace(config, gpu_memory_gib=1))["decision"], "exceeds_budget_screen")

    def test_total_memory_is_not_automatically_pooled(self):
        config = inference(parameters=70_000_000_000, layers=80, query_heads=64)
        result = estimate_inference(replace(config, tp=2))
        self.assertGreater(result["estimated_per_gpu_gib"], result["usable_budget_per_gpu_gib"])
        self.assertEqual(result["decision"], "exceeds_budget_screen")

    def test_invalid_inputs(self):
        invalid = {
            "parameters": (0, -1, 1.2, True), "gpu_memory_gib": (0, -1, math.nan, math.inf),
            "layers": (0, 31), "query_heads": (0, 33), "kv_heads": (0, 7, 64),
            "head_dim": (0, -1), "context_tokens": (0, -1), "concurrency": (0, -1),
            "weight_bits": (3, 16.0), "kv_bits": (4, 16.0), "tp": (0, 3, 16),
            "pp": (0, 3, 64), "kv_layout": ("unknown",),
            "headroom_fraction": (-0.01, 1, math.nan, math.inf),
            "runtime_overhead_gib": (-1, math.nan), "weight_overhead_fraction": (-1, math.inf),
            "weight_imbalance_factor": (0.9, math.nan),
        }
        # A positive odd layer count is valid for PP=1; test it with PP=2 instead.
        invalid["layers"] = (0, -1)
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    estimate_inference(inference(**{name: value}))
        with self.assertRaises(ValueError):
            estimate_inference(inference(layers=31, pp=2))


class TrainingTests(unittest.TestCase):
    def test_default_adam_state_is_16_bytes(self):
        result = estimate_training(training())
        states = result["components_per_gpu_bytes"]
        self.assertEqual(result["persistent_bytes_per_parameter_before_sharding"], 16)
        self.assertEqual(sum(states[key] for key in ("weights", "gradients", "master_weights", "optimizer_states")), 128_000_000_000)

    def test_fp32_gradients_give_18_byte_variant(self):
        result = estimate_training(training(gradient_bits=32))
        self.assertEqual(result["persistent_bytes_per_parameter_before_sharding"], 18)
        self.assertEqual(result["components_per_gpu_bytes"]["gradients"], 32_000_000_000)

    def test_zero_stage_partition_rules(self):
        base = estimate_training(training())["components_per_gpu_bytes"]
        for stage in range(4):
            result = estimate_training(training(dp=4, zero_stage=stage))["components_per_gpu_bytes"]
            for key, first_stage in (("weights", 3), ("gradients", 2), ("master_weights", 1), ("optimizer_states", 1)):
                self.assertEqual(result[key], base[key] / (4 if stage >= first_stage else 1))
            self.assertEqual(result["activations"], 8 * GIB)

    def test_data_parallel_alone_does_not_reduce_states(self):
        base = estimate_training(training())
        result = estimate_training(training(dp=8))
        self.assertEqual(result["estimated_per_gpu_gib"], base["estimated_per_gpu_gib"])
        self.assertEqual(result["gpu_count"], 8)

    def test_model_parallelism_does_not_divide_supplied_activations(self):
        result = estimate_training(training(tp=2, pp=2, dp=2, zero_stage=3, runtime_overhead_gib=2))
        self.assertEqual(result["components_per_gpu_bytes"]["weights"], 2_000_000_000)
        self.assertEqual(result["components_per_gpu_bytes"]["activations"], 8 * GIB)
        self.assertEqual(result["components_per_gpu_bytes"]["runtime_overhead"], 2 * GIB)
        self.assertEqual(result["gpu_count"], 8)

    def test_no_master_or_optimizer_states_allowed(self):
        result = estimate_training(training(master_weight_bits=0, optimizer_bits_per_parameter=0))
        self.assertEqual(result["persistent_bytes_per_parameter_before_sharding"], 4)

    def test_zero_activation_warns(self):
        result = estimate_training(training(activation_gib=0, dp=1, zero_stage=3))
        self.assertTrue(any("Activations are zero" in item for item in result["warnings"]))
        self.assertTrue(any("no ZeRO" in item for item in result["warnings"]))

    def test_invalid_inputs(self):
        invalid = {
            "activation_gib": (-1, math.nan), "dp": (0, -1, 1.2), "zero_stage": (-1, 4, True),
            "weight_bits": (4, 16.0), "gradient_bits": (8, 16.0),
            "master_weight_bits": (8, 32.0), "optimizer_bits_per_parameter": (-1, 1.5),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    estimate_training(training(**{name: value}))


if __name__ == "__main__":
    unittest.main()
