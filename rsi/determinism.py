"""Deterministic GPU kernels for every real-model gradient the study records.

GPU pilot 2026-10-01 (Qwen3-1.7B with the shared LoRA adapter, bf16, SDPA
attention, RTX 4070 Laptop): the summed gradient of the same 16 examples,
computed twice in one process, differed by up to 2.0e-4 per element with the
default kernels and was bitwise equal with deterministic kernels, at about 4%
more time.  The study compares R and S through h^T Delta theta, so that
run-to-run noise is removed rather than reported around.

cuBLAS reads CUBLAS_WORKSPACE_CONFIG when it creates its handle, so this must
run before the first matrix product on the GPU; callers call it before loading
the model.  A value the user already set is kept and recorded.
"""
import os

WORKSPACE = ":4096:8"


def enable_determinism():
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", WORKSPACE)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return {"deterministic_algorithms": True, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
