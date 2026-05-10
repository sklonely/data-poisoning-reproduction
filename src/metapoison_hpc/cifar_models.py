from collections import OrderedDict

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)),
                ('bn', nn.BatchNorm2d(planes)),
            ]))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, num_blocks=(3, 3, 3), num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(BasicBlock(self.in_planes, planes, stride=block_stride))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.fc(self.features(x))


def build_model(name, num_classes=10):
    name = name.lower()
    if name in {'resnet20', 'resnet'}:
        return CifarResNet((3, 3, 3), num_classes=num_classes)
    if name == 'resnet32':
        return CifarResNet((5, 5, 5), num_classes=num_classes)
    raise ValueError(f'Unsupported architecture: {name}')
