"""Simulated ("fake") weight quantization.

Everything stays in FP32 storage; values are snapped to the grid an INT8 / INT4
kernel would use. This is the standard way PTQ privacy studies are run, and it is
what lets the whole project run on a laptop with no INT4 kernels available.

Schemes
-------
fp32 : untouched baseline
fp16 : half-precision round trip (real dtype, so it is also a latency arm)
int8 : per-output-channel symmetric, 8-bit
int4 : group-wise asymmetric, 4-bit, group size g along the input dimension

By convention (and following common PTQ practice) the stem conv and the final
classifier can be kept at full precision via `skip_first_last=True`.
"""

import copy

import torch
import torch.nn as nn

QUANTIZABLE = (nn.Conv2d, nn.Linear)


# ----------------------------------------------------------------------------- tensor ops

def fp16_roundtrip(w: torch.Tensor) -> torch.Tensor:
    return w.half().float()


def quant_int8_per_channel(w: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-channel int8. w: [out, ...]"""
    out = w.shape[0]
    flat = w.reshape(out, -1).float()
    scale = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    q = torch.clamp(torch.round(flat / scale), -127, 127)
    return (q * scale).reshape(w.shape)


def quant_intN_group(w: torch.Tensor, bits: int = 4, group: int = 128) -> torch.Tensor:
    """Asymmetric group-wise quantization along the flattened input dimension."""
    out = w.shape[0]
    flat = w.reshape(out, -1).float()
    n = flat.shape[1]
    g = min(group, n)
    pad = (-n) % g
    if pad:
        flat = torch.cat([flat, torch.zeros(out, pad, device=w.device)], dim=1)
    grouped = flat.reshape(out, -1, g)

    qmax = 2 ** bits - 1
    lo = grouped.amin(dim=2, keepdim=True)
    hi = grouped.amax(dim=2, keepdim=True)
    scale = ((hi - lo) / qmax).clamp(min=1e-12)
    zp = torch.round(-lo / scale)
    q = torch.clamp(torch.round(grouped / scale) + zp, 0, qmax)
    deq = (q - zp) * scale

    deq = deq.reshape(out, -1)
    if pad:
        deq = deq[:, :n]
    return deq.reshape(w.shape)


# ----------------------------------------------------------------------------- model level

def quantize_model(model: nn.Module, scheme: str = "int8", group: int = 128,
                   skip_first_last: bool = True) -> nn.Module:
    """Returns a deep copy of `model` with quantized weights. Biases/BN untouched."""
    m = copy.deepcopy(model)
    if scheme == "fp32":
        return m

    layers = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, QUANTIZABLE)]
    skip = set()
    if skip_first_last and layers:
        skip = {layers[0][0], layers[-1][0]}

    for name, mod in layers:
        if name in skip:
            continue
        with torch.no_grad():
            w = mod.weight.data
            if scheme == "fp16":
                mod.weight.data = fp16_roundtrip(w)
            elif scheme == "int8":
                mod.weight.data = quant_int8_per_channel(w)
            elif scheme == "int4":
                mod.weight.data = quant_intN_group(w, bits=4, group=group)
            elif scheme == "int3":
                mod.weight.data = quant_intN_group(w, bits=3, group=group)
            else:
                raise ValueError(f"unknown scheme: {scheme}")
    return m


BITS = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4, "int3": 3}


def weight_footprint_mb(model: nn.Module, scheme: str, group: int = 128,
                        skip_first_last: bool = True) -> float:
    """Analytic weight memory: quantized layers at their bit width (+ scale/zero-point
    overhead), everything else at FP32. Simulated quantization does not shrink the
    in-memory tensors, so footprint must be computed, not measured."""
    bits = BITS[scheme]
    layers = [(n, mod) for n, mod in model.named_modules() if isinstance(mod, QUANTIZABLE)]
    skip = set()
    if skip_first_last and layers:
        skip = {layers[0][0], layers[-1][0]}

    quant_names = {n for n, _ in layers if n not in skip}
    total_bits = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, QUANTIZABLE):
            continue
        n_w = mod.weight.numel()
        if name in quant_names:
            total_bits += n_w * bits
            if scheme in ("int4", "int3"):
                n_groups = max(1, n_w // max(1, group))
                total_bits += n_groups * 2 * 16  # scale + zero point, fp16
            elif scheme == "int8":
                total_bits += mod.weight.shape[0] * 16  # one scale per channel
        else:
            total_bits += n_w * 32
    for p_name, p in model.named_parameters():
        if p_name.endswith("weight") and any(p_name.startswith(n + ".") or p_name == n + ".weight"
                                             for n, _ in layers):
            continue
        total_bits += p.numel() * 32
    return total_bits / 8 / 1e6


if __name__ == "__main__":
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    for fn, tag in [(lambda t: quant_int8_per_channel(t), "int8"),
                    (lambda t: quant_intN_group(t, 4, 32), "int4/g32")]:
        wq = fn(w)
        err = (w - wq).abs().mean().item()
        levels = len(torch.unique(wq[0]))
        print(f"{tag:10s} mean|Δw|={err:.5f}  distinct levels in row 0 = {levels}")
