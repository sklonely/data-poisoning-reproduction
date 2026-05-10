"""0% baseline: train ConvNetBN victim from scratch on CLEAN CIFAR-10 (no poisons),
measure target prediction for the first 5 bird images. ASR_baseline = % predicted as dog (class 5).
This corresponds to Fig 4 paper's "0.001% budget = effectively zero poisons" point.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from .cifar_models import build_model

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='./data')
    p.add_argument('--arch', default='resnet20', help='resnet20 (paper-aligned PyTorch victim)')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0)
    p.add_argument('--weight-decay', type=float, default=0)
    p.add_argument('--schedule', default='100,150')
    p.add_argument('--trials', type=int, default=6)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--target-class', type=int, default=2, help='bird')
    p.add_argument('--adv-class', type=int, default=5, help='dog')
    p.add_argument('--n-targets', type=int, default=5, help='number of bird target images to track (test set bird IDs 0..n-1)')
    p.add_argument('--output', required=True)
    return p.parse_args()


def get_target_images(data_root, target_class, n):
    """Return first n images of target_class from CIFAR-10 TEST set as a tensor."""
    # raw test set without normalization, applied later in eval
    test = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)
    imgs, labels = test.data, np.array(test.targets)
    idxs = np.where(labels == target_class)[0][:n]
    imgs = imgs[idxs].astype(np.uint8)
    return imgs, idxs.tolist()


def evaluate_targets(model, target_imgs, device):
    """Evaluate model on target images, return predicted classes."""
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    batch = torch.stack([transform(x) for x in target_imgs]).to(device)
    with torch.no_grad():
        logits = model(batch)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
    return preds


def train_clean_once(args, trial_idx, target_imgs):
    seed = args.seed + trial_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.arch).to(device)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type=='cuda')

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(s) for s in args.schedule.split(',') if s], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            preds = evaluate_targets(model, target_imgs, device)
            print(f'trial {trial_idx} | epoch {epoch} | preds={preds} | elapsed={int(time.time()-t0)}s')

    final_preds = evaluate_targets(model, target_imgs, device)
    return {'trial': trial_idx, 'seed': seed, 'final_preds': final_preds}


def main():
    args = parse_args()
    target_imgs, target_idxs = get_target_images(args.data_root, args.target_class, args.n_targets)
    print(f'Tracking {args.n_targets} target images of class {args.target_class}, IDs {target_idxs}')

    results = []
    for t in range(args.trials):
        r = train_clean_once(args, t, target_imgs)
        results.append(r)

    all_preds = [p for r in results for p in r['final_preds']]
    asr = sum(1 for p in all_preds if p == args.adv_class) / len(all_preds)
    summary = {
        'n_trials': args.trials,
        'n_targets': args.n_targets,
        'total_votes': len(all_preds),
        'baseline_asr_pct_predicted_as_adv': asr * 100,
        'pred_distribution': {c: all_preds.count(c) for c in range(10)},
        'target_class': args.target_class,
        'adv_class': args.adv_class,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'summary': summary, 'trials': results, 'args': vars(args)}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
