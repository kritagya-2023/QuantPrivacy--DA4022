"""Membership inference attacks and metrics.

Deliberately depends on numpy/sklearn only: it consumes per-sample *scores*
(losses, probabilities), never a model, so it can be unit-tested without torch.

Attacks
-------
loss        : score = -CE loss                       (Yeom et al.)
confidence  : score = p(true class)                  (Shokri et al., thresholded)
entropy     : score = -modified prediction entropy   (Song & Mittal)
lira        : offline LiRA — target's logit-confidence standardized against the
              distribution of shadow models that did NOT train on that sample
              (Carlini et al.); used in Phase 3.

Reporting
---------
AUC hides worst-case exposure, so TPR at low FPR is reported alongside it.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-12


# ----------------------------------------------------------------------------- scores

def loss_score(losses: np.ndarray) -> np.ndarray:
    return -np.asarray(losses, dtype=np.float64)


def confidence_score(probs_true: np.ndarray) -> np.ndarray:
    return np.asarray(probs_true, dtype=np.float64)


def modified_entropy_score(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Negative modified prediction entropy (Song & Mittal, 2021).

    probs: [N, C] softmax outputs. labels: [N] true class indices.
    """
    probs = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0 - EPS)
    n, _ = probs.shape
    idx = np.arange(n)
    p_true = probs[idx, labels]

    term = -probs * np.log(1.0 - probs)            # j != y
    term[idx, labels] = 0.0
    ment = term.sum(axis=1) - (1.0 - p_true) * np.log(p_true)
    return -ment


def logit_confidence(probs_true: np.ndarray) -> np.ndarray:
    """phi = log(p / (1-p)) — the LiRA statistic; roughly Gaussian across models."""
    p = np.clip(np.asarray(probs_true, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1.0 - p))


def lira_offline_score(target_phi: np.ndarray, shadow_phi_out: np.ndarray) -> np.ndarray:
    """target_phi: [N]. shadow_phi_out: [N, S] statistics from shadow models that
    did NOT contain the sample. Score = (phi - mu_out) / sigma_out."""
    mu = shadow_phi_out.mean(axis=1)
    sd = shadow_phi_out.std(axis=1).clip(min=1e-6)
    return (np.asarray(target_phi) - mu) / sd


# ----------------------------------------------------------------------------- metrics

def tpr_at_fpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.interp(target_fpr, fpr, tpr))


def evaluate_attack(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> dict:
    y = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    s = np.concatenate([member_scores, nonmember_scores])
    return {
        "auc": float(roc_auc_score(y, s)),
        "tpr@1%fpr": tpr_at_fpr(y, s, 0.01),
        "tpr@0.1%fpr": tpr_at_fpr(y, s, 0.001),
        "advantage": float(max(tpr_at_fpr(y, s, f) - f for f in (0.001, 0.01, 0.1))),
    }


def run_attack_suite(member: dict, nonmember: dict) -> dict:
    """member/nonmember are dicts with keys: losses [N], probs [N, C], labels [N]."""
    out = {}
    out["loss"] = evaluate_attack(loss_score(member["losses"]),
                                  loss_score(nonmember["losses"]))
    p_true_m = member["probs"][np.arange(len(member["labels"])), member["labels"]]
    p_true_n = nonmember["probs"][np.arange(len(nonmember["labels"])), nonmember["labels"]]
    out["confidence"] = evaluate_attack(confidence_score(p_true_m),
                                        confidence_score(p_true_n))
    out["entropy"] = evaluate_attack(
        modified_entropy_score(member["probs"], member["labels"]),
        modified_entropy_score(nonmember["probs"], nonmember["labels"]))
    return out


if __name__ == "__main__":
    # self-test: members are given systematically lower loss, so every attack
    # should land well above chance; identical distributions should sit at ~0.5.
    rng = np.random.default_rng(0)
    n, c = 4000, 10

    def synth(loss_mean):
        losses = np.abs(rng.normal(loss_mean, 0.5, n))
        logits = rng.normal(0, 1, (n, c))
        labels = rng.integers(0, c, n)
        logits[np.arange(n), labels] += 6.0 - losses
        e = np.exp(logits - logits.max(1, keepdims=True))
        return {"losses": losses, "probs": e / e.sum(1, keepdims=True), "labels": labels}

    sep = run_attack_suite(synth(0.05), synth(1.2))
    same = run_attack_suite(synth(0.6), synth(0.6))
    for k in sep:
        print(f"{k:11s} separable AUC={sep[k]['auc']:.3f} "
              f"tpr@1%={sep[k]['tpr@1%fpr']:.3f} | null AUC={same[k]['auc']:.3f}")
