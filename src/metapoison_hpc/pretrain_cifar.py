"""Train a clean CIFAR-10 classifier to convergence.

Used as the frozen feature extractor for the Feature Collision baseline (Phase D / Fig 3),
and as the pretrained init for fine-tuning victims in the same phase.

Saves a checkpoint dict {'state_dict', 'epoch', 'valid_acc', 'arch'}.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .cifar_models import build_model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def build_loaders(data_root, batch_size, num_workers, augment):
    train_ops = [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    if augment:
        train_ops = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    train_tf = transforms.Compose(train_ops)
    valid_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    train = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    valid = datasets.CIFAR10(data_root, train=False, download=True, transform=valid_tf)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, valid_loader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='./data')
    p.add_argument('--arch', default='resnet20')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=2e-4)
    p.add_argument('--schedule', default='60,80')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--output', required=True, help='Path to save checkpoint .pt')
    p.add_argument('--no-augment', action='store_true',
                   help='Disable training augmentation. Default ON for the pretrained classifier '
                        '(this is OK because Phase D evaluation does not use this model directly '
                        'as a victim — it is a feature extractor / init).')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, valid_loader = build_loaders(args.data_root, args.batch_size,
                                               args.num_workers, augment=not args.no_augment)
    model = build_model(args.arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(s) for s in args.schedule.split(',') if s], gamma=0.1)

    history = []
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        valid_acc = evaluate(model, valid_loader, device)
        history.append({'epoch': epoch, 'valid_acc': valid_acc})
        print(f'epoch {epoch} valid_acc {valid_acc:.4f}')
        if valid_acc > best_acc:
            best_acc = valid_acc

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'arch': args.arch,
        'epoch': args.epochs,
        'valid_acc': history[-1]['valid_acc'],
        'best_valid_acc': best_acc,
        'history': history,
        'args': vars(args),
    }, out)
    print(json.dumps({'final_valid_acc': history[-1]['valid_acc'],
                      'best_valid_acc': best_acc, 'output': str(out)}, indent=2))


if __name__ == '__main__':
    main()
