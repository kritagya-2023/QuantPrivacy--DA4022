"""CIFAR-style ResNet-18 (3x3 stem, no maxpool) — ~11M params, trains fine on MPS."""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, width=64):
        super().__init__()
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, 2, 1)
        self.layer2 = self._make_layer(width * 2, 2, 2)
        self.layer3 = self._make_layer(width * 4, 2, 2)
        self.layer4 = self._make_layer(width * 8, 2, 2)
        self.fc = nn.Linear(width * 8, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def build_model(num_classes=10, width=64):
    return ResNet18(num_classes=num_classes, width=width)
