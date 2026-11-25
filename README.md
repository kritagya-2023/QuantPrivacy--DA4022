# Quantization and Privacy — DA4022 Course Project (Direction 15)

How does membership inference risk change as a trained model is pushed from FP32
through FP16, INT8 and INT4? Everything here runs on a laptop (Apple Silicon MPS
or CPU) using simulated quantization, so no INT4 kernels or GPUs are required.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 2 — mid-project (due 15 Oct)

```bash
python run_midpoint.py --smoke                 # 1 min: checks the wiring, numbers are noise
python run_midpoint.py --epochs 30             # the real run: ~20 min on an M-series Mac
python run_midpoint.py --epochs 30 --no-augment --out results_noaug   # stronger leakage signal
python run_midpoint.py --epochs 30 --seed 1 --out results_seed1       # noise floor
python run_midpoint.py --epochs 30 --seed 2 --out results_seed2
python make_figs.py --results results/results.json --out results/fig_midpoint.png
python make_midpoint_slide.py                  # -> DA4022_Midproject_Slide.pptx
```

`DA4022_Midproject_Slide.pptx` in this repo was generated from
`results/results.example.json`, which holds **illustrative placeholder numbers so the
layout could be checked**. Run the two commands above before submitting — the slide
reads every figure from `results.json`, so it rebuilds itself with your real results.

## Phase 3 — final 

```bash
python run_final.py --stage grid                        # minutes
python run_final.py --stage shadows --shadows 16        # hours — start this first
python run_final.py --stage lira                        # minutes, needs the shadows
python run_final.py --stage qft                         # ~10 min
python run_final.py --stage persample                   # seconds, needs grid
```

Then write up `report/final_report.tex`.

## What each file does

| File | Role |
|---|---|
| `data.py` | CIFAR-10 and the fixed member / non-member / shadow split |
| `models.py` | CIFAR ResNet-18 |
| `train.py` | training loop, per-sample loss/probability scoring, device pick |
| `quantize.py` | simulated FP16 / INT8 (per-channel) / INT4 (group-wise) + analytic footprint |
| `attacks.py` | loss, confidence, modified-entropy and LiRA attacks; AUC and TPR@low-FPR |
| `benchmark.py` | latency measurement (FP32/FP16 only — see note below) |
| `run_midpoint.py` | Phase 2 experiment |
| `run_final.py` | Phase 3 stages |
| `make_figs.py`, `make_midpoint_slide.py` | figure and one-slide report, built from `results.json` |

## Why `--no-augment` matters

With random crop and flip on, the augmented target reaches ~83% test accuracy but leaks
very little: loss-attack AUC ~0.58 and TPR@1%FPR *below* the 1% chance line. At that
signal level the FP32-to-INT4 differences are ~0.001 and indistinguishable from seed
noise, so no claim about quantization and privacy can be supported either way.
Training without augmentation memorizes harder and should push AUC to ~0.65-0.75, which
is what makes the precision comparison measurable. Run both and report the contrast --
"quantization's privacy effect is only visible in models that memorize" is itself a
finding worth stating.

Always run 2-3 seeds before believing any gap between arms.


1. **Simulated quantization does not speed anything up.** Weights are snapped to the
   INT8/INT4 grid but still stored and multiplied as FP32, so the low-bit arms have no
   real latency gain. Footprint is therefore computed analytically
   (`weight_footprint_mb`) and latency is measured only for FP32 and FP16, which
   execute as real dtypes. Say this in the report rather than implying INT4 timings.
2. **A drop in attack AUC is not automatically a privacy gain.** Flattening confidence
   scores can lower an attack's average success while leaving the most memorized
   examples just as exposed — which is exactly why `--stage persample` and the
   TPR@1%FPR metric are in here, and why they should lead the results section.

## Self-tests

```bash
python attacks.py     # attacks on synthetic scores: separable case vs. null case
python quantize.py    # quantization grid sanity check
```
