# 05 · Migration playbook

[Guide home](../README.md) · Reviewed 2026-10-11

A fallback is useful only when it can take the same workload safely. Work through these gates for each candidate; record evidence in the [benchmark plan](../templates/benchmark-plan.md). These instructions do not provision resources or authorize spending.

## Gate 1: preserve the workload contract

Freeze model/tokenizer revisions, prompt formatting, output limits, API behavior, quality bar, and request mix. Record current context, concurrency, and latency objectives. A smaller context limit or different quantization is a service change that needs explicit acceptance.

## Gate 2: qualify the software stack

The name “Blackwell” does not establish kernel compatibility. NVIDIA currently lists RTX PRO 6000 Blackwell Server Edition at compute capability **12.0**, B200 at **10.0**, B300 at **10.3**, and H100/H200 at **9.0**. These are GPU instruction targets, not a simple performance ranking. [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus)

- **CPU ISA and platform:** verify x86-64 versus Arm64 for the image, Python wheels, native extensions and launch tools. Grace-based systems can require an Arm build; a B200/B300 GPU name alone does not identify the host CPU. [vLLM GPU installation and Arm builds](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- **Driver and CUDA:** match the target instance and the actual container/runtime requirements. A host's `nvidia-smi` CUDA compatibility display is not proof of the toolkit bundled in the image. Use a supported AMI/container combination and record versions. [AWS GPU instance setup](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/gpu-instances-started.html)
- **Compiled kernels:** native cubins and generic PTX have different compatibility rules; architecture-specific accelerated targets have additional limits. A Hopper `compute_90a` kernel is not automatically valid on Blackwell. Check every compiled extension, not just whether PyTorch imports. [NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)
- **Serialized engines and caches:** retain the source checkpoint and build recipe. TensorRT engines are not generally portable across GPU architectures unless the relevant hardware-compatibility mode was used; version and platform constraints still apply. Rebuild and re-tune for the target rather than copying an opaque engine and assuming equivalence. [TensorRT support and engine portability](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- **Model and precision:** TensorRT-LLM's supported-hardware list includes H100/H200/B200/B300, but that does not certify every model, feature, quantized format or backend combination. Check the selected release and perform a real generation test, then quality evaluation. [TensorRT-LLM hardware support](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/supported-hardware.md)

A safe migration preserves the checkpoint/tokenizer/chat template and application contract first, changes one major variable at a time, and reruns quality and load tests after each precision or parallelism change.

## Gate 3: verify the exact AWS candidate

- The launch geography is not the current geography. G7e launched in Northern Virginia and Ohio in January 2026; the current regional table also lists it elsewhere. B300 launched in Oregon, but the current region and Capacity Blocks tables list additional regions. Use the live support tables and the customer's account/API results rather than copying launch-only region lists. [G7e launch](https://aws.amazon.com/about-aws/whats-new/2026/01/amazon-g7e-instances-generally-available/), [current EC2 regions](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-regions.html), [Capacity Blocks](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)
- Region support does not imply every Availability Zone offers the size, nor that requested capacity is available now. Run `describe-instance-type-offerings` for exact allowed sizes and zones; then check quota and the intended capacity mechanism. An offering result is not an inventory reservation. [AWS offerings API](https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instance-type-offerings.html)
- G7e's GA announcement supports On-Demand, Spot, and Savings Plans. The B300 launch identifies Capacity Blocks and Savings Plans and sends on-demand reservation requests to the account manager. Current procurement choices must be checked directly; a Savings Plan is not proof of physical capacity. [G7e GA](https://aws.amazon.com/about-aws/whats-new/2026/01/amazon-g7e-instances-generally-available/), [B300 GA](https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-ec2-p6-b300-instances-nvidia-blackwell-ultra-gpus-available/)
- Build fallback pools by compatible workload profile. A scheduler should never silently substitute one 24 GB L4 for one 96 GB RTX PRO, or an Arm UltraServer for an x86 node, just because both are GPU instances. Separate GPU count, minimum per-device memory, host RAM, architecture, topology, tested software image, and region constraints.
- Before switching: pin and test the driver/CUDA/framework/container combination; rebuild architecture-specific extensions; verify GPU enumeration, peer topology and collectives; test the actual model and precision; repeat load, restart and checkpoint tests; then canary behind the same API and preserve rollback. Merely changing the instance-type string does not validate compatibility.
- AWS currently specifies extra Fabric Manager installation requirements for P6-B200 and P6-B300. Prefer a supported, current DLAMI/container deployment path over copying a driver installation recipe from a different generation. [AWS NVIDIA driver requirements](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/public-nvidia-driver.html)

### Read-only verification commands

These commands are examples for the customer's configured AWS CLI. They were not executed against a customer account during this research.

```bash
# Confirm exact GPU count, host memory and API-reported device memory in the selected Region.
aws ec2 describe-instance-types --region us-east-1 \
  --instance-types g7e.8xlarge p5.4xlarge p5.48xlarge p5en.48xlarge p6-b200.48xlarge p6-b300.48xlarge \
  --query 'InstanceTypes[].{Type:InstanceType,Arch:ProcessorInfo.SupportedArchitectures,vCPU:VCpuInfo.DefaultVCpus,HostMiB:MemoryInfo.SizeInMiB,GPUs:GpuInfo.Gpus,TotalGPUMiB:GpuInfo.TotalGpuMemoryInMiB,EFA:NetworkInfo.EfaSupported}'

# Discover which AZ IDs offer one candidate. This does not check free capacity.
aws ec2 describe-instance-type-offerings --region us-east-1 \
  --location-type availability-zone-id \
  --filters Name=instance-type,Values=p5en.48xlarge \
  --query 'InstanceTypeOfferings[].{Type:InstanceType,AZ:Location}'

# On an already provisioned test instance, record actual device memory and peer topology.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi topo -m
```

API field source: [DescribeInstanceTypes](https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instance-types.html). Physical verification source: [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

## Gate 4: measure, canary, and retain rollback

1. Load the exact checkpoint on the target. Exercise maximum supported context, graph capture, and the longest output path. Record every rank’s peak memory.
2. Run relevant collective tests and inspect transport logs. A successful multi-node launch can still be falling back to a slower path.
3. Run the same quality set and request trace as the baseline. Record [SLO-passing goodput and cost](07-benchmarking.md), including failures and retries.
4. Exercise startup, replica drain, worker/rank failure, restart, and checkpoint restore where applicable.
5. Promote a small canary only after acceptance. Keep the baseline image, checkpoint, and configuration available and define a rollback trigger.

Do not make automatic fallback a list of GPU family names. Make it a list of already-qualified workload profiles with explicit memory, topology, software, and capacity constraints.

---

Next: [Run the arithmetic examples](06-worked-examples.md)
