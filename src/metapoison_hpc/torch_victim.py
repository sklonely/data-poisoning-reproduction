import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .cifar_models import build_model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class ArrayDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        ops = []
        if augment:
            ops.extend([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
        self.transform = transforms.Compose(ops)
        self.images = images.astype(np.uint8)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), int(self.labels[idx])


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return {
        'loss': loss_sum / max(total, 1),
        'acc': correct / max(total, 1),
    }


def evaluate_target(model, xtarget, ytarget, ytargetadv, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    batch = torch.stack([transform(x.astype(np.uint8)) for x in xtarget]).to(device)
    with torch.no_grad():
        logits = model(batch)
        preds = logits.argmax(dim=1).cpu().numpy()
    return {
        'target_true': np.asarray(ytarget).tolist(),
        'target_adv': np.asarray(ytargetadv).tolist(),
        'target_pred': preds.tolist(),
        'attack_success_rate': float(np.mean(preds == np.asarray(ytargetadv))),
    }


def train_once(payload, args, trial_idx):
    seed = args.seed + trial_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_model(args.arch).to(device)

    train_loader = DataLoader(
        ArrayDataset(payload['xtrain'], payload['ytrain'], augment=args.augment),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    valid_loader = DataLoader(
        ArrayDataset(payload['xvalid'], payload['yvalid'], augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(step) for step in args.schedule.split(',') if step],
        gamma=0.1,
    )

    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        valid_metrics = evaluate(model, valid_loader, device)
        target_metrics = evaluate_target(model, payload['xtarget'], payload['ytarget'], payload['ytargetadv'], device)
        record = {
            'epoch': epoch,
            'train_loss': train_loss / max(total, 1),
            'train_acc': train_correct / max(total, 1),
            'valid_loss': valid_metrics['loss'],
            'valid_acc': valid_metrics['acc'],
            'attack_success_rate': target_metrics['attack_success_rate'],
            'target_pred': target_metrics['target_pred'],
        }
        history.append(record)
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(
                f'trial {trial_idx} | epoch {epoch} | '
                f'train_acc {record["train_acc"]:.4f} | '
                f'valid_acc {record["valid_acc"]:.4f} | '
                f'asr {record["attack_success_rate"]:.4f}'
            )

    final_target = evaluate_target(model, payload['xtarget'], payload['ytarget'], payload['ytargetadv'], device)
    return {
        'trial': trial_idx,
        'seed': seed,
        'history': history,
        'final': {
            'valid_acc': history[-1]['valid_acc'],
            'valid_loss': history[-1]['valid_loss'],
            'train_acc': history[-1]['train_acc'],
            'attack_success_rate': final_target['attack_success_rate'],
            'target_pred': final_target['target_pred'],
            'target_true': final_target['target_true'],
            'target_adv': final_target['target_adv'],
        },
    }


def summarize(results):
    final_valid_acc = [trial['final']['valid_acc'] for trial in results]
    final_asr = [trial['final']['attack_success_rate'] for trial in results]
    return {
        'n_trials': len(results),
        'valid_acc_mean': float(np.mean(final_valid_acc)),
        'valid_acc_std': float(np.std(final_valid_acc)),
        'attack_success_rate_mean': float(np.mean(final_asr)),
        'attack_success_rate_std': float(np.std(final_asr)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train a PyTorch victim on a MetaPoison poisoned dataset export.')
    parser.add_argument('--dataset', required=True, help='Path to poisondataset-<craftstep>.pkl')
    parser.add_argument('--output', default='runs/torch_compare/results.json', help='Where to save JSON results')
    parser.add_argument('--arch', default='resnet20', help='Victim model architecture')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument('--schedule', default='100,150')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--device', default=None)
    parser.add_argument('--augment', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    with dataset_path.open('rb') as f:
        payload = pickle.load(f)

    results = [train_once(payload, args, trial_idx) for trial_idx in range(args.trials)]
    summary = summarize(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump({'dataset': str(dataset_path), 'summary': summary, 'trials': results}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
