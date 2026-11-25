"""Training and per-sample scoring."""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(model, train_loader, epochs=30, lr=0.1, weight_decay=5e-4,
                device=None, log_every=5, label="target"):
    device = device or pick_device()
    model = model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(epochs):
        running, seen, correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)
        sched.step()
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"[{label}] epoch {ep+1:3d}/{epochs}  loss {running/seen:.4f}  "
                  f"train acc {correct/seen:.4f}  ({time.time()-t0:.0f}s)")
    return model


@torch.no_grad()
def score_model(model, data_loader, device=None):
    """Per-sample cross-entropy losses, softmax probabilities and labels."""
    device = device or pick_device()
    model = model.to(device).eval()
    losses, probs, labels = [], [], []
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(F.cross_entropy(out, y, reduction="none").cpu().numpy())
        probs.append(F.softmax(out, dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return {
        "losses": np.concatenate(losses),
        "probs": np.concatenate(probs).astype(np.float64),
        "labels": np.concatenate(labels),
    }


def accuracy(scored):
    return float((scored["probs"].argmax(1) == scored["labels"]).mean())
