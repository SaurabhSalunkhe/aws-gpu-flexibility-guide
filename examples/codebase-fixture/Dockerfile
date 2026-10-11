# Synthetic migration fixture. Not a runnable or recommended image.
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV NCCL_P2P_DISABLE=1
ENV NCCL_SOCKET_IFNAME=eth0
CMD ["python", "workload.py"]
