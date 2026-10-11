# 02 · AWS GPU instance catalog

[Guide home](../README.md) · Reviewed 2026-10-11

## Key mappings

- **G7e is not an H100 family.** It uses NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs with nominal 96 GB GDDR7 per GPU. Its multi-GPU peer path is **GPUDirect P2P over PCIe**. AWS does not describe it as an NVSwitch-connected HBM platform. [AWS G7e](https://aws.amazon.com/ec2/instance-types/g7e/)
- **P5 is H100; P5e and P5en are H200.** The single-GPU `p5.4xlarge` is meaningfully different from the eight-GPU `p5.48xlarge`: the smaller instance has no intra-instance peer GPU and does not support GPUDirect RDMA. [AWS P5](https://aws.amazon.com/ec2/instance-types/p5/)
- **B300 is already an EC2 product**, not merely a prospective GPU: `p6-b300.48xlarge` became generally available on November 18, 2025. Do not conflate B300, GB300, and G7e simply because all use Blackwell-generation technology. [AWS B300 launch](https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-ec2-p6-b300-instances-nvidia-blackwell-ultra-gpus-available/)

## Focused candidate catalog

GPU memory below uses the stated product-page nominal GB labels for G6/G6e/G7e/P4/P5. Blackwell P6 memory has conflicting public labels, so the exact reported values and caveat follow the table. Total GPU memory is a sum across devices; it is **not** one automatically pooled allocation. Host RAM is separate and is stated in GiB.

| Candidate | AWS size(s) | GPU count | GPU memory per device / total | Intra-node topology | Network ceiling | CPU / host RAM |
|---|---|---:|---|---|---|---|
| G6 / L4 | `g6.xlarge` through `g6.16xlarge` single-GPU sizes; `.12xlarge` and `.24xlarge`; `.48xlarge` | 1 / 4 / 8 | 24 GB GDDR6 / 24, 96, 192 GB | PCIe; no NVLink fabric | Up to 100 Gbps; size-specific EFA | AMD EPYC 7R13, x86-64; 16–768 GiB |
| G6e / L40S | single-GPU sizes through `.16xlarge`; `.12xlarge` and `.24xlarge`; `.48xlarge` | 1 / 4 / 8 | 48 GB GDDR6 / 48, 192, 384 GB | PCIe; NVIDIA specifies no NVLink | Up to 400 Gbps; size-specific EFA | AMD EPYC 7R13, x86-64; 32–1,536 GiB |
| G7e / RTX PRO 6000 Blackwell Server Edition | `.2xlarge`, `.4xlarge`, `.8xlarge`; `.12xlarge`; `.24xlarge`; `.48xlarge` | 1 / 2 / 4 / 8 | 96 GB GDDR7 / 96, 192, 384, 768 GB | GPUDirect P2P over PCIe | Up to 1,600 Gbps; multi-GPU EFAv4 + GPUDirect RDMA | Intel Emerald Rapids, x86-64; 64–2,048 GiB |
| P4d / A100 | `p4d.24xlarge` | 8 | 40 GB HBM2 / 320 GB | NVLink/NVSwitch, 600 GB/s advertised GPU peer link | 400 Gbps ENA/EFA; RDMA source conflict below | Intel 8275CL, x86-64; 96 vCPU, 1,152 GiB |
| P4de / A100 | `p4de.24xlarge` | 8 | 80 GB HBM2e / 640 GB | NVLink/NVSwitch, 600 GB/s advertised GPU peer link | 400 Gbps ENA/EFA; RDMA source conflict below | Intel 8275CL, x86-64; 96 vCPU, 1,152 GiB |
| P5 / H100, small | `p5.4xlarge` | 1 | 80 GB HBM3 / 80 GB | No intra-instance GPU peer | 100 Gbps EFA; **no GPUDirect RDMA** | AMD EPYC 7R13, x86-64; 16 vCPU, 256 GiB |
| P5 / H100, eight-GPU | `p5.48xlarge` | 8 | 80 GB HBM3 / 640 GB | NVLink/NVSwitch, 900 GB/s advertised GPU interconnect | 3,200 Gbps EFAv2; GPUDirect RDMA | AMD EPYC 7R13, x86-64; 192 vCPU, 2,048 GiB |
| P5e / H200 | `p5e.48xlarge` | 8 | 141 GB HBM3e / 1,128 GB | NVLink/NVSwitch, 900 GB/s advertised GPU interconnect | 3,200 Gbps EFAv2; GPUDirect RDMA | AMD EPYC 7R13, x86-64; 192 vCPU, 2,048 GiB |
| P5en / H200 | `p5en.48xlarge` | 8 | 141 GB HBM3e / 1,128 GB | NVLink/NVSwitch; PCIe Gen5 CPU/GPU path | 3,200 Gbps EFAv3; GPUDirect RDMA | Intel Sapphire Rapids, x86-64; 192 vCPU, 2,048 GiB |
| P6-B200 / B200 | `p6-b200.48xlarge` | 8 | HBM3e; AWS docs: 179 GiB per device, 1,432 GiB aggregate; see caveat | Fifth-gen NVLink/NVSwitch; 1,800 GB/s peer interconnect | 3,200 Gbps EFAv4; GPUDirect RDMA | Intel Emerald Rapids, x86-64; 192 vCPU, 2,048 GiB |
| P6-B300 / B300 | `p6-b300.48xlarge` | 8 | HBM3e; AWS docs: 268 GiB per device; aggregate labels conflict; see caveat | NVLink/NVSwitch; 1,800 GB/s peer interconnect | 6,400 Gbps EFAv4 plus dedicated 300 Gbps ENA | Intel Emerald Rapids, x86-64; 192 vCPU, 4,096 GiB |

Sources by family: [G6](https://aws.amazon.com/ec2/instance-types/g6/), [G6e](https://aws.amazon.com/ec2/instance-types/g6e/), [G7e](https://aws.amazon.com/ec2/instance-types/g7e/), [P4](https://aws.amazon.com/ec2/instance-types/p4/), [P5](https://aws.amazon.com/ec2/instance-types/p5/), [P6](https://aws.amazon.com/ec2/instance-types/p6/), [B200 launch specifications](https://aws.amazon.com/blogs/aws/new-amazon-ec2-p6-b200-instances-powered-by-nvidia-blackwell-gpus-to-accelerate-ai-innovations/), [B300 launch specifications](https://aws.amazon.com/blogs/aws/accelerate-large-scale-ai-applications-with-the-new-amazon-ec2-p6-b300-instances/). CPU models and EFA size flags: [EC2 instance-type specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html). GPU memory technology: [NVIDIA workstation GPU table](https://docs.nvidia.com/vgpu/sizing/virtual-workstation/latest/gpus-vws.html), [NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/).

### Interconnect-source warning

AWS’s P4 product table advertises GPUDirect RDMA, while the current EFA matrix lists GDR as unsupported for P4d/P4de. This unresolved source conflict is preserved as `null` in the structured catalog; verify the supported path with AWS and the actual software stack before a network-sensitive migration. [AWS P4](https://aws.amazon.com/ec2/instance-types/p4/), [EFA support matrix](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)

### Memory-label warning: preserve units and validate the actual machine

AWS pages do not consistently use GB and GiB. At this check, the EC2 specifications page labels G6 as 22 GiB per GPU versus the G6 product page's nominal 24 GB; G6e similarly appears as 44 GiB versus nominal 48 GB. G7e is listed as 96 GiB in EC2 specifications and 96 GB in the product page. Do not apply an unqualified GB↔GiB conversion to every family.

For P6-B200, the product-page table says 1,432 GB while its prose and launch article say 1,440 GB. EC2 specifications say 1,432 GiB (8 × 179 GiB). For P6-B300, the product table and launch article say 2,144 GB; EC2 specifications say 2,148 GiB with 8 × 268 GiB in parentheses. The parenthetical multiplication is 2,144, so that page has an unresolved internal discrepancy. These are reported source labels, not independently measured allocatable capacities. [AWS P6](https://aws.amazon.com/ec2/instance-types/p6/), [EC2 specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)

NVIDIA's generic HGX table lists B200 at 180 GB and B300 at 288 GB HBM3e per GPU. That describes NVIDIA's platform, not a promise of EC2 application-usable memory. Keep generic hardware specifications separate from AWS instance specifications. Use `DescribeInstanceTypes`, then `nvidia-smi --query-gpu=name,memory.total --format=csv` and a real model-load test on the selected AMI to establish deployment limits. Record driver reservations, runtime overhead, KV-cache and activation headroom. [NVIDIA HGX components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)

## G7e sizing details that affect migration

All sizes use the same nominal 96 GB GPU. Increasing CPU size without increasing GPU count does not increase per-GPU VRAM.

| Size | GPUs | Nominal aggregate GPU GB | vCPU | Host GiB | Network Gbps | EFA |
|---|---:|---:|---:|---:|---:|---|
| `g7e.2xlarge` | 1 | 96 | 8 | 64 | 50 | No |
| `g7e.4xlarge` | 1 | 96 | 16 | 128 | 50 | No |
| `g7e.8xlarge` | 1 | 96 | 32 | 256 | 100 | Yes |
| `g7e.12xlarge` | 2 | 192 | 48 | 512 | 400 | Yes |
| `g7e.24xlarge` | 4 | 384 | 96 | 1,024 | 800 | Yes |
| `g7e.48xlarge` | 8 | 768 | 192 | 2,048 | 1,600 | Yes |

Do not infer GPU count from the size suffix: `g6e.12xlarge` has four GPUs, while `g7e.12xlarge` has two. G6/G6e `.16xlarge` remain single-GPU instances. EFA also varies by size: G6 `.8xlarge` and larger listed sizes support EFA; G6e `.8xlarge`, `.12xlarge`, `.16xlarge`, `.24xlarge`, and `.48xlarge` do. EFA support does **not** by itself establish GPUDirect RDMA support. [G7e product sizing](https://aws.amazon.com/ec2/instance-types/g7e/), [EC2 network specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)

## The separate UltraServer lane

`p6e-gb200.36xlarge` has four Blackwell GPUs and NVIDIA Grace Arm CPUs, rather than an Intel/AMD x86 host. AWS lists 144 vCPUs, 960 GiB host RAM, and 740 aggregate GPU memory (GB on the product table; GiB in EC2 specifications). It is available only as part of an UltraServer. The documented UltraServer forms contain 36 or 72 GPUs in an NVLink domain; those are not ordinary four-GPU stand-alone EC2 substitutes. Their aggregate GPU-memory figures also must not be confused with per-device memory. [AWS P6 product table](https://aws.amazon.com/ec2/instance-types/p6/)

A move here includes an Arm container/AMI/dependency compatibility check and a cluster topology/deployment redesign. AWS itself distinguishes x86 P6-B200 as the familiar eight-GPU migration path from the larger Grace Blackwell UltraServer platform. [AWS architecture discussion](https://aws.amazon.com/blogs/machine-learning/aws-ai-infrastructure-with-nvidia-blackwell-two-powerful-compute-solutions-for-the-next-frontier-of-ai/)

The current P6 product page also describes P6e-GB300, but the checked EC2 instance specifications table did not provide a corresponding launchable size. Do not invent an instance identifier, region list, or general-availability claim from a product-page mention alone. Confirm the required purchase/deployment path with AWS before adding it to an executable fallback catalog.

## Machine-readable inventory

[data/instance-catalog.json](../data/instance-catalog.json) preserves 31 concrete instance sizes (30 x86-64 sizes and one Arm64 UltraServer component) and their source labels. Null means unverified or deliberately not normalized. It is a research inventory, not a launch allowlist, availability feed, or input to the estimator. Fractional GPU sizes and every other EC2 accelerator are outside this focused inventory.

Use the [migration checks](05-migration-playbook.md) to validate your exact account, Region, size, and machine before deployment.

---

Next: [Budget the model](03-model-memory.md)
