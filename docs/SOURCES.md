# Source ledger and refresh policy

[Guide home](../README.md) · Sources reviewed **2026-10-11**

## How to read the evidence

- AWS product pages establish the advertised instance platform; live EC2/EFA documentation supplies size-specific details. Launch posts establish launch timing, not today’s inventory.
- Model-publisher configurations establish example architecture shapes. Rounded parameter counts and invented overhead/capacity inputs remain illustrative arithmetic.
- Engine/framework documentation establishes behavior for its referenced version; live `stable`, `latest`, and `main` links can change. Pin releases and immutable revisions in production.
- Hardware specifications do not establish model fit, workload speed, available capacity, price, or a deployment recommendation. No customer AWS API calls or GPU benchmark runs were made for this guide.

## Known unresolved source differences

1. AWS memory labels differ between product pages and EC2 specifications, including GB/GiB and B200/B300 aggregate values. The [catalog](02-instance-catalog.md) preserves the differences.
2. P4 product documentation and the EFA support matrix disagree on GPUDirect RDMA. The structured catalog retains both source statements and marks the resolved capability unknown.
3. NVIDIA’s generic HGX memory specifications describe a hardware platform, not guaranteed application-allocatable memory on EC2.

## Refresh before deployment

Recheck the exact size, Region, offering, quota and purchase mechanism; then measure the actual device, topology and software stack. For a repository refresh, update both the prose and structured catalog, retain an explicit review date, and run the tests. Do not substitute a new source label silently when it changes a sizing conclusion.

The [machine-readable ledger](../data/source-ledger.json) records each URL and the files using it. The list below groups primary references for manual review.

## AWS hardware and operations

- [AWS G7e](https://aws.amazon.com/ec2/instance-types/g7e/)
- [AWS P5](https://aws.amazon.com/ec2/instance-types/p5/)
- [AWS P6](https://aws.amazon.com/ec2/instance-types/p6/)
- [AWS B300 launch](https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-ec2-p6-b300-instances-nvidia-blackwell-ultra-gpus-available/)
- [G6](https://aws.amazon.com/ec2/instance-types/g6/)
- [G6e](https://aws.amazon.com/ec2/instance-types/g6e/)
- [P4](https://aws.amazon.com/ec2/instance-types/p4/)
- [B200 launch specifications](https://aws.amazon.com/blogs/aws/new-amazon-ec2-p6-b200-instances-powered-by-nvidia-blackwell-gpus-to-accelerate-ai-innovations/)
- [B300 launch specifications](https://aws.amazon.com/blogs/aws/accelerate-large-scale-ai-applications-with-the-new-amazon-ec2-p6-b300-instances/)
- [EC2 instance-type specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [EFA support matrix](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [AWS architecture discussion](https://aws.amazon.com/blogs/machine-learning/aws-ai-infrastructure-with-nvidia-blackwell-two-powerful-compute-solutions-for-the-next-frontier-of-ai/)
- [AWS EFA and NCCL setup](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html)
- [AWS GPU instance setup](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/gpu-instances-started.html)
- [G7e launch](https://aws.amazon.com/about-aws/whats-new/2026/01/amazon-g7e-instances-generally-available/)
- [current EC2 regions](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-regions.html)
- [Capacity Blocks](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)
- [AWS offerings API](https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instance-type-offerings.html)
- [AWS NVIDIA driver requirements](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/public-nvidia-driver.html)
- [DescribeInstanceTypes](https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instance-types.html)
- [AWS benchmark and configuration tables](https://aws.amazon.com/blogs/machine-learning/accelerate-generative-ai-inference-on-amazon-sagemaker-ai-with-g7e-instances/)
- [AWS, 25 September 2026, Figure 5 and version matrix](https://aws.amazon.com/blogs/machine-learning/scaling-moe-reinforcement-learning-on-amazon-eks-with-efa-and-deepep-with-40-more-throughput/)

## Models, frameworks, published studies, and GPU software

- [NERSC roofline documentation](https://docs.nersc.gov/tools/performance/roofline/)
- [NVIDIA's matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [NCCL performance troubleshooting](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/troubleshooting/performance_and_tuning.html)
- [NVIDIA NCCL tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- [Megatron-LM decomposition](https://research.nvidia.com/labs/adlr/MegatronLM/)
- [Megatron Core's parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [GPipe](https://arxiv.org/html/1811.06965v5)
- [NVIDIA workstation GPU table](https://docs.nvidia.com/vgpu/sizing/virtual-workstation/latest/gpus-vws.html)
- [NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/)
- [NVIDIA HGX components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)
- [vLLM quantization](https://docs.vllm.ai/en/stable/features/quantization/)
- [Hugging Face Llama configuration](https://huggingface.co/docs/transformers/v4.45.2/model_doc/llama)
- [vLLM attention cache tensor shapes](https://docs.vllm.ai/en/v0.9.2/api/vllm/attention/backends/flash_attn.html)
- [vLLM model configuration](https://docs.vllm.ai/en/stable/api/vllm/config/model/)
- [vLLM quantized KV cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)
- [vLLM conserving memory](https://docs.vllm.ai/en/stable/configuration/conserving_memory/)
- [PyTorch memory APIs](https://github.com/pytorch/pytorch/blob/main/torch/cuda/memory.py)
- [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)
- [Hugging Face training memory anatomy](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_memory_anatomy.md)
- [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [Hugging Face DeepSpeed ZeRO](https://huggingface.co/docs/transformers/deepspeed)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint)
- [Hugging Face PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
- [PyTorch tensor parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [vLLM data parallel deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)
- [vLLM expert parallel deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
- [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus)
- [vLLM GPU installation and Arm builds](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)
- [TensorRT support and engine portability](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- [TensorRT-LLM hardware support](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/supported-hardware.md)
- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [8B](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8/blob/main/config.json)
- [70B](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8/blob/main/config.json)
- [DeepSpeed's memory guidance](https://deepspeed.readthedocs.io/en/latest/memory.html)
- [vLLM performance benchmarking](https://docs.vllm.ai/en/latest/benchmarking/dashboard/)
- [NVIDIA goodput definition](https://developer.download.nvidia.com/triton/vLLM-x-Triton-meetup-External.pdf)
- [NVIDIA NCCL troubleshooting](https://github.com/NVIDIA/nccl/blob/master/docs/userguide/source/troubleshooting/performance_and_tuning.rst)
- [AWS OFI NCCL](https://github.com/aws/aws-ofi-nccl)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth)
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [SGLang issue #18870, including February 20 follow-up](https://github.com/sgl-project/sglang/issues/18870)
- [Author's raw-artifact collection](https://gist.github.com/stewtong/980e7b95d03443d81cdd1057582c4518)
- [NVIDIA v5.1 Table 2 and Figure 1 footnotes](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/)
- [MLCommons results](https://docs.mlcommons.org/inference_results_v5.1/)
- [submission artifacts](https://github.com/mlcommons/inference_results_v5.1)
- [benchmark rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- [NVIDIA v5.0, “Hopper continues to improve”](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0)
- [Sarathi-Serve, §5.3, Figure 13, Tables 1–3 and artifact appendix](https://www.usenix.org/system/files/osdi24-agrawal.pdf)
- [DistServe, §§5–6 and Table 1](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)
- [FlashAttention-3 v2, §4 and Appendix C.1](https://arxiv.org/html/2407.08608v2)
- [PyTorch FSDP, §§5.1–5.4, Figures 6–8](https://arxiv.org/html/2304.11277v1)
- [Megatron-LM, §§3–5, Table 1](https://arxiv.org/pdf/2104.04473)
- [SmoothQuant, §§5.1–5.4, Figures 8–9 and Tables 8–9](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf)
- [Demystifying NCCL v1, §V-E and Figure 6](https://arxiv.org/html/2507.04786v1)
- [ServerlessLLM, §§7.1–7.2 and Figure 6](https://www.usenix.org/system/files/osdi24-fu.pdf)
- [NCCL GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
- [AWS EKS/Ray training transport checks](https://aws.github.io/deep-learning-containers/ray-train/deployment/eks/)
- [vLLM's legacy custom-all-reduce implementation](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py)
- [Microbenchmarking NVIDIA's Blackwell Architecture v1](https://arxiv.org/html/2512.02189v1)
- [nvidia l4](https://www.nvidia.com/en-us/data-center/l4/)

## Repository validation tooling

- [GitHub checkout action](https://github.com/actions/checkout)
- [GitHub setup-python action](https://github.com/actions/setup-python)
