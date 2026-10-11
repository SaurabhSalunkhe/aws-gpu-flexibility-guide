# Synthetic alternate GPU build targets demonstrate explicit architecture review.
set(CMAKE_CUDA_ARCHITECTURES "90;100")
set(ILLUSTRATIVE_FLAGS "-gencode=arch=compute_90a,code=sm_90a")
