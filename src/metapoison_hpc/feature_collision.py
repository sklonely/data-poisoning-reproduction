"""Feature Collision baseline (Shafahi et al. 2018, "Poison Frogs!").

Poison-crafting algorithm. Given a frozen feature extractor f(.), a target image x_t
(true class = bird, attacker wants it predicted as dog), and N base images x_b drawn
from the adversarial class (dog), produce N poison images x_p that:
  1) sit close to x_t in feature space:        ||f(x_p) - f(x_t)||^2  small
  2) stay visually close to their base x_b:    ||x_p - x_b||^2        small

Optimization uses forward-backward splitting (paper Algorithm 1):
  forward:  x_hat <- x_p - lr * grad_xp ||f(x_p) - f(x_t)||^2
  backward: x_p   <- (x_hat + beta * lr * x_b) / (1 + beta * lr)

After crafting, the poisons are labeled as the adversarial class (dog) and inserted
into the victim's training set. The victim then fine-tunes a pretrained classifier
on (clean ∪ poison); test-time prediction on x_t flips to dog.

This file ONLY does poison crafting. Victim fine-tuning lives in torch_finetune_victim.py.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from .cifar_models import build_model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def normalize_batch(x_uint8_0_255):
    """Convert a [N,32,32,3] uint8 tensor in 0..255 to normalized float [N,3,32,32]."""
    x = x_uint8_0_255.float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor(CIFAR10_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def craft_feature_collision(
    feature_fn,
    x_target_uint8,      # [1, 32, 32, 3] uint8
    x_base_uint8,        # [N, 32, 32, 3] uint8 — base images (adversarial class)
    *,
    lr=0.01 * 255.0,     # in 0..255 image space
    beta=0.25,
    iters=12000,
    eps=None,            # optional L_inf clip on (x_p - x_b) in 0..255 units; None = no clip
    decay_iter=None,     # iter at which to drop lr by 10x
    log_every=500,
    device='cuda',
):
    """Run forward-backward splitting on each base image to produce a feature-collision poison.

    `feature_fn` must be a callable mapping normalized input batch -> features.
    Caller is responsible for putting the underlying model in eval mode and freezing
    its parameters.

    Returns: (x_poison_uint8 [N,32,32,3], stats dict).
    """
    x_t = x_target_uint8.float().to(device)        # [1,32,32,3] in 0..255
    x_b = x_base_uint8.float().to(device)           # [N,32,32,3] in 0..255
    x_p = x_b.clone().requires_grad_(True)          # poison image, in 0..255

    # Pre-compute target features once.
    f_t = feature_fn(normalize_batch(x_t))          # [1, D]

    history = []
    cur_lr = lr
    for it in range(iters):
        f_p = feature_fn(normalize_batch(x_p))      # [N, D]
        loss_feat = ((f_p - f_t) ** 2).sum(dim=1).mean()

        grad = torch.autograd.grad(loss_feat, x_p)[0]

        with torch.no_grad():
            # forward step on feature loss
            x_hat = x_p - cur_lr * grad
            # backward (proximal) step on beta*||x_p - x_b||^2
            x_p_new = (x_hat + beta * cur_lr * x_b) / (1.0 + beta * cur_lr)
            # optional L_inf perturbation budget (clean-label appearance)
            if eps is not None:
                delta = torch.clamp(x_p_new - x_b, min=-eps, max=eps)
                x_p_new = x_b + delta
            x_p_new.clamp_(0.0, 255.0)
            x_p.copy_(x_p_new)

        if decay_iter is not None and it == decay_iter:
            cur_lr = cur_lr / 10.0

        if it % log_every == 0 or it == iters - 1:
            with torch.no_grad():
                f_p_now = feature_fn(normalize_batch(x_p))
                feat_dist = ((f_p_now - f_t) ** 2).sum(dim=1).sqrt().mean().item()
                pixel_dist = (x_p - x_b).abs().mean().item()
            history.append({'iter': it, 'feat_l2': feat_dist, 'pixel_l1': pixel_dist})
            print(f'iter {it:5d} | feat ||f(p)-f(t)|| = {feat_dist:.4f} | mean |p - b| = {pixel_dist:.3f}/255')

    x_p_uint8 = x_p.detach().round().clamp(0, 255).byte().cpu()
    return x_p_uint8, {'history': history, 'lr': lr, 'beta': beta, 'iters': iters, 'eps': eps}


def load_pretrained(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt['arch'])
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model


def select_target_and_bases(data_root, target_class, base_class, target_idx, n_base, seed):
    """Pull the target image (target_class) and N base images (base_class) from CIFAR-10 test
    set. CIFAR-10 test set has 1000 images per class. Returns uint8 [H,W,C] arrays."""
    test = datasets.CIFAR10(data_root, train=False, download=True, transform=None)
    targets = np.array(test.targets)
    target_indices = np.where(targets == target_class)[0]
    base_indices = np.where(targets == base_class)[0]
    rng = np.random.default_rng(seed)
    base_pick = rng.choice(base_indices, size=n_base, replace=False)
    x_t = np.array(test.data[target_indices[target_idx]])[None]            # [1,32,32,3]
    x_b = np.array(test.data[base_pick])                                   # [N,32,32,3]
    y_t = int(targets[target_indices[target_idx]])
    y_b = [int(targets[i]) for i in base_pick]
    return (
        torch.from_numpy(x_t),
        torch.from_numpy(x_b),
        {'target_idx_in_class': int(target_idx),
         'target_class': target_class, 'base_class': base_class,
         'target_label': y_t, 'base_labels': y_b,
         'target_global_idx': int(target_indices[target_idx]),
         'base_global_idx': base_pick.tolist()},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained', required=True, help='Path to clean CIFAR-10 checkpoint')
    p.add_argument('--data-root', default='./data')
    p.add_argument('--target-class', type=int, default=2, help='True class of target (default 2 = bird)')
    p.add_argument('--base-class', type=int, default=5,
                   help='Class of base images / desired adversarial label (default 5 = dog)')
    p.add_argument('--target-idx', type=int, default=0,
                   help='Index within the target class in the test set (0..999)')
    p.add_argument('--n-poison', type=int, default=50)
    p.add_argument('--lr', type=float, default=2.55, help='lr in 0..255 image space (default 0.01 * 255)')
    p.add_argument('--beta', type=float, default=0.25)
    p.add_argument('--iters', type=int, default=12000)
    p.add_argument('--eps', type=float, default=None,
                   help='Optional L_inf budget on |poison - base| in 0..255 units. '
                        'Set ~16 to mimic Shafahi watermark-free clean-label setting; '
                        'leave None for paper-default unbounded perturbation.')
    p.add_argument('--decay-iter', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', required=True, help='Path to save poisons .pkl')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = load_pretrained(args.pretrained, device)
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad_(False)

    x_t, x_b, meta = select_target_and_bases(
        args.data_root, args.target_class, args.base_class,
        args.target_idx, args.n_poison, args.seed,
    )

    # Use only the penultimate layer for feature collision (per Shafahi et al.)
    feature_fn = feature_extractor.features

    poisons, stats = craft_feature_collision(
        feature_fn, x_t, x_b,
        lr=args.lr, beta=args.beta, iters=args.iters,
        eps=args.eps, decay_iter=args.decay_iter, device=device,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        pickle.dump({
            'poisons_uint8': poisons.numpy(),               # [N,32,32,3]
            'base_uint8': x_b.numpy(),                      # [N,32,32,3]
            'target_uint8': x_t.numpy(),                    # [1,32,32,3]
            'meta': meta,
            'stats': stats,
            'args': vars(args),
        }, f)
    print(json.dumps({'output': str(out), 'n_poison': args.n_poison,
                      'final_feat_l2': stats['history'][-1]['feat_l2']}, indent=2))


if __name__ == '__main__':
    main()
