"""Fine-tune victim trainer for the Phase D / Fig 3 comparison.

Eats either:
  - a feature_collision.py output pkl (FC baseline), or
  - a MetaPoison poisondataset-N.pkl + a chosen base set (MetaPoison fine-tuning mode)

Loads a pretrained CIFAR-10 classifier, replaces only the final FC layer (so it learns
fresh class boundaries on the poisoned data), and trains for a small number of epochs
(paper used 10). Reports ASR on x_target.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .cifar_models import build_model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class PoisonedFinetuneSet(torch.utils.data.Dataset):
    """Combines a clean CIFAR-10 train subset with the poisons (labeled as base_class)."""

    def __init__(self, data_root, poisons_uint8, poison_label, clean_per_class=500, seed=0):
        train = datasets.CIFAR10(data_root, train=True, download=True, transform=None)
        targets = np.array(train.targets)
        rng = np.random.default_rng(seed)
        clean_indices = []
        for c in range(10):
            idx = np.where(targets == c)[0]
            clean_indices.extend(rng.choice(idx, size=clean_per_class, replace=False).tolist())
        clean_indices = np.array(clean_indices, dtype=np.int64)

        clean_images = train.data[clean_indices]                          # [N, 32, 32, 3] uint8
        clean_labels = targets[clean_indices]                             # [N]

        poison_labels = np.full((len(poisons_uint8),), poison_label, dtype=np.int64)
        self.images = np.concatenate([clean_images, poisons_uint8], axis=0)
        self.labels = np.concatenate([clean_labels, poison_labels], axis=0)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(self.images[idx].astype(np.uint8)), int(self.labels[idx])


def evaluate_target(model, x_target_uint8, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    x = torch.stack([transform(img.astype(np.uint8)) for img in x_target_uint8]).to(device)
    with torch.no_grad():
        return model(x).argmax(1).cpu().numpy().tolist()


def evaluate_clean(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def train_once(payload, args, trial_idx):
    seed = args.seed + trial_idx
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
    model = build_model(ckpt['arch']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    if args.reset_fc:
        # Re-init the FC layer so the victim learns the boundary fresh on poisoned data.
        nn.init.kaiming_normal_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)

    train_ds = PoisonedFinetuneSet(
        data_root=args.data_root,
        poisons_uint8=payload['poisons_uint8'],
        poison_label=payload['meta']['base_class'],
        clean_per_class=args.clean_per_class,
        seed=seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    valid_set = datasets.CIFAR10(args.data_root, train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                                 ]))
    valid_loader = DataLoader(valid_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)

    history = []
    target_uint8 = payload['target_uint8']
    target_class = payload['meta']['target_class']
    base_class = payload['meta']['base_class']

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        valid_acc = evaluate_clean(model, valid_loader, device)
        target_pred = evaluate_target(model, target_uint8, device)
        asr = float(np.mean([p == base_class for p in target_pred]))
        history.append({'epoch': epoch, 'valid_acc': valid_acc,
                        'target_pred': target_pred, 'attack_success_rate': asr})
        print(f'trial {trial_idx} epoch {epoch} valid_acc {valid_acc:.4f} '
              f'target_pred {target_pred} asr {asr}')

    return {
        'trial': trial_idx, 'seed': seed, 'history': history,
        'final': {
            'valid_acc': history[-1]['valid_acc'],
            'target_pred': history[-1]['target_pred'],
            'attack_success_rate': history[-1]['attack_success_rate'],
            'target_class': target_class, 'base_class': base_class,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained', required=True)
    p.add_argument('--poisons', required=True, help='Output pkl from feature_collision.py')
    p.add_argument('--data-root', default='./data')
    p.add_argument('--epochs', type=int, default=10, help='Paper used 10 fine-tune epochs')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--clean-per-class', type=int, default=500,
                   help='Number of clean training images per class to fine-tune on')
    p.add_argument('--trials', type=int, default=1)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--reset-fc', action='store_true',
                   help='Re-init final FC layer (fine-tune as transfer-learning rather than warm-start).')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    with open(args.poisons, 'rb') as f:
        payload = pickle.load(f)

    results = [train_once(payload, args, t) for t in range(args.trials)]
    final_asr = [r['final']['attack_success_rate'] for r in results]
    final_valid = [r['final']['valid_acc'] for r in results]
    summary = {
        'n_trials': len(results),
        'asr_mean': float(np.mean(final_asr)),
        'asr_std': float(np.std(final_asr)),
        'valid_acc_mean': float(np.mean(final_valid)),
        'valid_acc_std': float(np.std(final_valid)),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        json.dump({'poisons_path': args.poisons, 'pretrained_path': args.pretrained,
                   'summary': summary, 'trials': results, 'args': vars(args)}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
