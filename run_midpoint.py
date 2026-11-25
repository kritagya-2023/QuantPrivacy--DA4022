"""Phase 2 (mid-project) experiment.

Scope kept deliberately small so it finishes in one sitting on a laptop:
  train one target model -> quantize to FP16 / INT8 / INT4 -> run the three
  score-based attacks at every precision -> dump results.json + figure.

LiRA, the QAT arm and the per-sample vulnerability analysis are Phase 3.

    python run_midpoint.py --epochs 30
    python run_midpoint.py --smoke        # 60-second wiring check, meaningless numbers
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import attacks
import benchmark
import data as data_mod
from models import build_model
from quantize import quantize_model, weight_footprint_mb
from train import accuracy, pick_device, score_model, train_model

SCHEMES = ["fp32", "fp16", "int8", "int4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--group", type=int, default=128, help="INT4 group size")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed (weight init + batch order). The member/non-member "
                         "split is fixed separately so runs stay comparable.")
    ap.add_argument("--no-augment", action="store_true",
                    help="train without random crop/flip -> stronger memorization -> a "
                         "leakage signal large enough to compare precisions against.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    device = pick_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device: {device} | seed {args.seed} | augment {not args.no_augment}")

    if args.smoke:
        member_idx, nonmember_idx, _, train_ds, eval_ds, test_ds = data_mod.smoke_splits(256)
        args.epochs = 1
    else:
        member_idx, nonmember_idx, _, train_ds, eval_ds, test_ds = data_mod.get_splits(
            augment=not args.no_augment)

    train_loader = data_mod.loader(train_ds, member_idx, args.batch_size, shuffle=True)
    member_loader = data_mod.loader(eval_ds, member_idx, 256)
    nonmember_loader = data_mod.loader(eval_ds, nonmember_idx, 256)
    test_loader = data_mod.loader(test_ds, None, 256)

    model = build_model()
    model = train_model(model, train_loader, epochs=args.epochs, lr=args.lr,
                        device=device, label="target")
    torch.save(model.state_dict(), out_dir / "target_fp32.pt")

    results = {"config": vars(args), "hardware": benchmark.device_info(device), "arms": {}}

    for scheme in SCHEMES:
        qm = quantize_model(model, scheme=scheme, group=args.group)
        m = score_model(qm, member_loader, device)
        n = score_model(qm, nonmember_loader, device)
        t = score_model(qm, test_loader, device)

        arm = {
            "train_acc": accuracy(m),
            "test_acc": accuracy(t),
            "generalization_gap": accuracy(m) - accuracy(t),
            "mean_loss_member": float(np.mean(m["losses"])),
            "mean_loss_nonmember": float(np.mean(n["losses"])),
            "footprint_mb": weight_footprint_mb(model, scheme, group=args.group),
            "attacks": attacks.run_attack_suite(m, n),
        }
        if scheme in ("fp32", "fp16"):
            dt = torch.float32 if scheme == "fp32" else torch.float16
            try:
                arm["latency_ms"] = benchmark.measure_latency(qm, device, dtype=dt)
            except Exception as exc:                      # half unsupported on some CPUs
                arm["latency_ms"] = None
                arm["latency_note"] = str(exc)[:120]

        np.savez_compressed(out_dir / f"scores_{scheme}.npz",
                            member_losses=m["losses"], nonmember_losses=n["losses"],
                            member_ptrue=m["probs"][np.arange(len(m["labels"])), m["labels"]],
                            nonmember_ptrue=n["probs"][np.arange(len(n["labels"])), n["labels"]])
        results["arms"][scheme] = arm
        print(f"  {scheme:5s} test acc {arm['test_acc']:.4f} | "
              f"loss-attack AUC {arm['attacks']['loss']['auc']:.4f} | "
              f"TPR@1%FPR {arm['attacks']['loss']['tpr@1%fpr']:.4f} | "
              f"{arm['footprint_mb']:.2f} MB")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
