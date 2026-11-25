"""Latency measurement for the hardware-aware axis.

Honest scoping: simulated quantization does not change the executed kernels, so
INT8/INT4 arms cannot show a real speedup here. Latency is therefore measured
only for dtypes that actually execute (FP32, FP16), and memory footprint for the
low-bit arms is computed analytically in quantize.weight_footprint_mb.
"""

import platform
import time

import torch


def device_info(device):
    return {
        "device": str(device),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
    }


@torch.no_grad()
def measure_latency(model, device, batch_size=64, dtype=torch.float32,
                    warmup=5, iters=20, input_shape=(3, 32, 32)):
    """Mean per-batch inference latency in milliseconds."""
    model = model.to(device=device, dtype=dtype).eval()
    x = torch.randn(batch_size, *input_shape, device=device, dtype=dtype)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        model(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    sync()
    ms = (time.perf_counter() - t0) / iters * 1000
    model.float()
    return float(ms)
