"""Phase 3 (final) experiments. Run the stages in order; each writes into results/.

    python run_final.py --stage grid        # full precision grid incl. INT4 group sizes
    python run_final.py --stage shadows --shadows 16   # slowest stage, run overnight
    python run_final.py --stage lira        # LiRA scores from the cached shadows
    python run_final.py --stage qft         # quantize-then-finetune arm
    python run_final.py --stage persample   # who stays vulnerable across precisions

Stage `shadows` is the only expensive one (S x the target's training cost). Start it
early — everything else is minutes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

import attacks
import data as data_mod
from models import build_model
from quantize import quantize_model, weight_footprint_mb
from train import accuracy, pick_device, score_model, train_model

OUT = Path("results")
GRID = [("fp32", None), ("fp16", None), ("int8", None),
        ("int4", 128), ("int4", 32), ("int3", 128)]


def _loaders(bs=128):
    m_idx, n_idx, s_idx, train_ds, eval_ds, test_ds = data_mod.get_splits()
    return dict(
        m_idx=m_idx, n_idx=n_idx, s_idx=s_idx,
        train=data_mod.loader(train_ds, m_idx, bs, shuffle=True),
        member=data_mod.loader(eval_ds, m_idx, 256),
        nonmember=data_mod.loader(eval_ds, n_idx, 256),
        test=data_mod.loader(test_ds, None, 256),
        train_ds=train_ds, eval_ds=eval_ds,
    )


def _load_target(device):
    model = build_model()
    model.load_state_dict(torch.load(OUT / "target_fp32.pt", map_location="cpu"))
    return model.to(device)


# --------------------------------------------------------------------- stages

def stage_grid(args, L, device):
    model = _load_target(device)
    res = {"arms": {}}
    for scheme, group in GRID:
        tag = scheme if group is None else f"{scheme}_g{group}"
        qm = quantize_model(model, scheme=scheme, group=group or 128)
        m = score_model(qm, L["member"], device)
        n = score_model(qm, L["nonmember"], device)
        t = score_model(qm, L["test"], device)
        res["arms"][tag] = {
            "train_acc": accuracy(m), "test_acc": accuracy(t),
            "footprint_mb": weight_footprint_mb(model, scheme, group=group or 128),
            "attacks": attacks.run_attack_suite(m, n),
        }
        np.savez_compressed(OUT / f"final_scores_{tag}.npz",
                            member_losses=m["losses"], nonmember_losses=n["losses"],
                            member_ptrue=m["probs"][np.arange(len(m["labels"])), m["labels"]],
                            nonmember_ptrue=n["probs"][np.arange(len(n["labels"])), n["labels"]])
        print(f"{tag:10s} acc {res['arms'][tag]['test_acc']:.4f}  "
              f"AUC {res['arms'][tag]['attacks']['loss']['auc']:.4f}  "
              f"{res['arms'][tag]['footprint_mb']:.2f} MB")
    json.dump(res, open(OUT / "final_grid.json", "w"), indent=2)


def stage_shadows(args, L, device):
    """Train S shadow models, each on a random half of the evaluation set."""
    eval_idx = np.concatenate([L["m_idx"], L["n_idx"]])
    rng = np.random.default_rng(2024)
    membership = np.zeros((len(eval_idx), args.shadows), dtype=bool)
    eval_loader = data_mod.loader(L["eval_ds"], eval_idx, 256)

    for s in range(args.shadows):
        inmask = rng.random(len(eval_idx)) < 0.5
        membership[:, s] = inmask
        sub = eval_idx[inmask]
        sm = build_model()
        sm = train_model(sm, data_mod.loader(L["train_ds"], sub, args.batch_size, shuffle=True),
                         epochs=args.shadow_epochs, device=device, label=f"shadow{s}")
        sc = score_model(sm, eval_loader, device)
        p_true = sc["probs"][np.arange(len(eval_idx)), sc["labels"]]
        np.save(OUT / f"shadow_phi_{s}.npy", attacks.logit_confidence(p_true))
        del sm
    np.save(OUT / "shadow_membership.npy", membership)
    print(f"trained {args.shadows} shadow models")


def stage_lira(args, L, device):
    membership = np.load(OUT / "shadow_membership.npy")
    S = membership.shape[1]
    phi = np.stack([np.load(OUT / f"shadow_phi_{s}.npy") for s in range(S)], axis=1)

    model = _load_target(device)
    eval_idx = np.concatenate([L["m_idx"], L["n_idx"]])
    is_member = np.concatenate([np.ones(len(L["m_idx"])), np.zeros(len(L["n_idx"]))]).astype(bool)
    eval_loader = data_mod.loader(L["eval_ds"], eval_idx, 256)

    out = {}
    for scheme, group in GRID:
        tag = scheme if group is None else f"{scheme}_g{group}"
        qm = quantize_model(model, scheme=scheme, group=group or 128)
        sc = score_model(qm, eval_loader, device)
        p_true = sc["probs"][np.arange(len(eval_idx)), sc["labels"]]
        target_phi = attacks.logit_confidence(p_true)

        # per-example OUT distribution; masked mean/std over shadows where it was out
        scores = np.zeros(len(eval_idx))
        for i in range(len(eval_idx)):
            out_phi = phi[i, ~membership[i]]
            if len(out_phi) < 2:
                out_phi = phi[i]
            scores[i] = (target_phi[i] - out_phi.mean()) / max(out_phi.std(), 1e-6)
        out[tag] = attacks.evaluate_attack(scores[is_member], scores[~is_member])
        np.save(OUT / f"lira_scores_{tag}.npy", scores)
        print(f"LiRA {tag:10s} AUC {out[tag]['auc']:.4f}  TPR@1% {out[tag]['tpr@1%fpr']:.4f}")
    json.dump(out, open(OUT / "final_lira.json", "w"), indent=2)


def stage_qft(args, L, device):
    """Quantize-then-finetune: does recovering accuracy also restore leakage?"""
    base = _load_target(device)
    res = {}
    for scheme, group in [("int4", 128), ("int3", 128)]:
        qm = quantize_model(base, scheme=scheme, group=group)
        qm = train_model(qm, L["train"], epochs=args.qft_epochs, lr=0.01,
                         device=device, label=f"qft-{scheme}")
        qm = quantize_model(qm, scheme=scheme, group=group)   # re-snap after finetuning
        m = score_model(qm, L["member"], device)
        n = score_model(qm, L["nonmember"], device)
        t = score_model(qm, L["test"], device)
        res[f"{scheme}_qft"] = {"test_acc": accuracy(t), "train_acc": accuracy(m),
                                "attacks": attacks.run_attack_suite(m, n)}
        print(f"{scheme}+finetune acc {accuracy(t):.4f} "
              f"AUC {res[f'{scheme}_qft']['attacks']['loss']['auc']:.4f}")
    json.dump(res, open(OUT / "final_qft.json", "w"), indent=2)


def stage_persample(args, L, device):
    """Does INT4 protect everyone equally, or only on average?"""
    ref = np.load(OUT / "final_scores_fp32.npz")
    ref_scores = np.concatenate([-ref["member_losses"], -ref["nonmember_losses"]])
    is_member = np.concatenate([np.ones(len(ref["member_losses"])),
                                np.zeros(len(ref["nonmember_losses"]))]).astype(bool)
    k = max(1, int(0.01 * is_member.sum()))
    ref_top = set(np.argsort(-np.where(is_member, ref_scores, -np.inf))[:k])

    out = {}
    for f in sorted(OUT.glob("final_scores_*.npz")):
        tag = f.stem.replace("final_scores_", "")
        d = np.load(f)
        s = np.concatenate([-d["member_losses"], -d["nonmember_losses"]])
        top = set(np.argsort(-np.where(is_member, s, -np.inf))[:k])
        out[tag] = {
            "spearman_vs_fp32": float(spearmanr(ref_scores, s).statistic),
            "top1pct_overlap": len(ref_top & top) / k,
        }
        print(f"{tag:10s} rho {out[tag]['spearman_vs_fp32']:.3f}  "
              f"top-1% overlap {out[tag]['top1pct_overlap']:.3f}")
    json.dump(out, open(OUT / "final_persample.json", "w"), indent=2)


STAGES = {"grid": stage_grid, "shadows": stage_shadows, "lira": stage_lira,
          "qft": stage_qft, "persample": stage_persample}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--shadows", type=int, default=16)
    ap.add_argument("--shadow-epochs", type=int, default=20)
    ap.add_argument("--qft-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    device = pick_device()
    print(f"device: {device} | stage: {args.stage}")
    STAGES[args.stage](args, _loaders(args.batch_size), device)
