"""Synthetic text-only fixture. Deliberately incomplete; never execute it."""
import torch
from torch.utils.cpp_extension import CUDAExtension

# Intentional fixed configuration: compare with proposed deployment layout.
tensor_parallel_size = 8
pipeline_parallel_size = 2
world_size = 16
precision = "bf16"
quantization = "fp8"


def illustrative_step(batch):
    device = "cuda:7"
    output = batch.to(device).cpu()
    return output


if __name__ == "__main__":
    raise RuntimeError("This synthetic fixture must not be executed.")
