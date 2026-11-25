"""CIFAR-10 loading with a controlled member / non-member split.

Split policy (fixed by SEED so every run is comparable):
    indices[0        : N_MEMBER]              -> members     (target model trains on these)
    indices[N_MEMBER : 2*N_MEMBER]            -> non-members (never seen by the target)
    indices[2*N_MEMBER:]                      -> shadow pool (used only for LiRA in Phase 3)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 1337
N_MEMBER = 12500  # => 12500 members + 12500 non-members + 25000 shadow pool

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_splits(root="./data", download=True, augment=True):
    """Returns (member_idx, nonmember_idx, shadow_idx, train_ds, eval_ds, test_ds).

    train_ds and eval_ds are the *same* underlying data; train_ds has augmentation
    (used for fitting) unless augment=False, eval_ds never does (used for scoring, so
    attack features are deterministic).

    augment=False removes random crop/flip. Augmentation is a strong regularizer that
    suppresses memorization, so the augmented target barely leaks and the precision arms
    become indistinguishable. The MIA literature trains targets without it for exactly
    this reason; keep both arms and report the contrast.
    """
    train_ds = datasets.CIFAR10(root, train=True, download=download,
                                transform=_transforms(augment))
    eval_ds = datasets.CIFAR10(root, train=True, download=False,
                               transform=_transforms(False))
    test_ds = datasets.CIFAR10(root, train=False, download=download,
                               transform=_transforms(False))

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(train_ds))
    member_idx = idx[:N_MEMBER]
    nonmember_idx = idx[N_MEMBER:2 * N_MEMBER]
    shadow_idx = idx[2 * N_MEMBER:]
    return member_idx, nonmember_idx, shadow_idx, train_ds, eval_ds, test_ds


def loader(ds, indices=None, batch_size=128, shuffle=False, workers=2):
    if indices is not None:
        ds = Subset(ds, list(indices))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=False)


def smoke_splits(n=512):
    """Tiny random-tensor stand-in so the pipeline can be exercised in seconds."""
    x = torch.randn(2 * n, 3, 32, 32)
    y = torch.randint(0, 10, (2 * n,))
    ds = torch.utils.data.TensorDataset(x, y)
    return np.arange(n), np.arange(n, 2 * n), np.arange(n), ds, ds, ds
