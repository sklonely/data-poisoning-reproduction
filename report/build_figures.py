"""Build the three figures for Checkpoint Presentation 2.

Inputs (all already on local disk under hpc-results/):
- Round 1 craft trajectory: hpc-results/round1-job20240192/outputs/local/craft1/<key>/metrics.jsonl
- Round 2 craft trajectory: hpc-results/round2-job20241084/outputs-v2/local/craft2/<key>/metrics.jsonl
- Round 2 PyTorch results: hpc-results/round2-job20241084/outputs-v2/torch-results.json
- Round 2 TF victim metrics: hpc-results/round2-job20241084/outputs-v2/local/victim1/<key>/metrics.jsonl
- Phase A victim metrics: hpc-results/phase-a-victims/victim-A0{1..4}/<key>/metrics.jsonl
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / 'hpc-results'
OUT = Path(__file__).parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def parse_craft_metrics(craft_dir):
    """Read metrics.jsonl from a single craft experiment dir, return list of (craftstep, cwT)."""
    points = []
    mp = craft_dir / 'metrics.jsonl'
    if not mp.exists():
        return points
    for line in mp.open():
        r = json.loads(line)
        step = r.get('step')
        cwT = r.get('metrics', {}).get('cwT')
        if step is not None and cwT is not None:
            points.append((step, cwT))
    points.sort()
    return points


def find_first_subdir(p):
    p = Path(p)
    if not p.exists():
        return None
    for d in sorted(p.iterdir()):
        if d.is_dir():
            return d
    return None


def fig1_craft_trajectory():
    """Round 1 (nmeta=2) vs Round 2 (nmeta=16) cwT over craftsteps."""
    r1 = find_first_subdir(RESULTS / 'round1-job20240192' / 'outputs' / 'local' / 'craft1')
    r2 = find_first_subdir(RESULTS / 'round2-job20241084' / 'outputs-v2' / 'local' / 'craft2')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if r1:
        pts = parse_craft_metrics(r1)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, '-o', label='Round 1: nmeta=2 (1 GPU, nreplay=2)', color='C3', markersize=4)
    if r2:
        pts = parse_craft_metrics(r2)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, '-s', label='Round 2: nmeta=16 (4 GPU, nreplay=4)', color='C0', markersize=4)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Attack signal threshold (cwT < 0)')
    ax.set_xlabel('Craftstep')
    ax.set_ylabel(r'Carlini-Wagner adversarial loss on target ($cwT$)')
    ax.set_title('Surrogate-ensemble size determines attack signal\n(ResNet, dog→bird, 5000 poisons)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT / 'fig1_craft_trajectory.png'
    fig.savefig(out, dpi=150)
    print(f'Saved {out}')
    plt.close(fig)


def fig2_round2_tf_vs_pytorch():
    """Round 2: TF victim ASR vs PyTorch victim ASR side-by-side."""
    # PyTorch from torch-results.json
    pt = json.load(open(RESULTS / 'round2-job20241084' / 'outputs-v2' / 'torch-results.json'))
    pt_asr = pt['summary']['attack_success_rate_mean']
    pt_acc = pt['summary']['valid_acc_mean']

    # TF from victim1 dir: read each victim experiment, find final target_pred
    tf_dir = RESULTS / 'round2-job20241084' / 'outputs-v2' / 'local' / 'victim2'
    tf_preds = []
    for d in sorted(tf_dir.iterdir() if tf_dir.exists() else []):
        if not d.is_dir():
            continue
        mp = d / 'metrics.jsonl'
        if not mp.exists():
            continue
        last_target = None
        for line in mp.open():
            r = json.loads(line)
            mm = r.get('metrics', {})
            if any(k.startswith('class-0-') for k in mm):
                classes = {int(k.split('-')[-1]): v for k, v in mm.items() if k.startswith('class-0-')}
                last_target = min(classes, key=classes.get)
        if last_target is not None:
            tf_preds.append(last_target)
    tf_asr = sum(1 for p in tf_preds if p == 5) / len(tf_preds) if tf_preds else 0

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ['TF victim\n(in-domain)', 'PyTorch victim\n(cross-framework)']
    asrs = [tf_asr * 100, pt_asr * 100]
    bars = ax.bar(labels, asrs, color=['C0', 'C2'])
    ax.set_ylabel('Attack Success Rate (%)')
    ax.set_ylim(0, 110)
    ax.set_title(f'Round 2: same poison set, two victim frameworks\n'
                 f'(ResNet, dog→bird, 5000 poisons, n={len(tf_preds)} TF / 3 PyTorch trials)')
    for bar, val in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.0f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.axhline(90, color='red', linestyle=':', alpha=0.6,
               label='Paper Fig 4: ResNet20 dog-bird @10% budget ≈ 90%')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = OUT / 'fig2_round2_tf_vs_pytorch.png'
    fig.savefig(out, dpi=150)
    print(f'Saved {out}, tf_preds={tf_preds}, pt_asr={pt_asr}')
    plt.close(fig)


def fig3_phase_a_asr_vs_budget():
    """Phase A: ConvNetBN dog-bird ASR vs poison budget. Mirrors paper Fig 4 (right)."""
    cells = [
        ('A01', 5, 'dog-bird'),
        ('A02', 50, 'dog-bird'),
        ('A03', 500, 'dog-bird'),
        ('A04', 2500, 'dog-bird'),
    ]
    base = RESULTS / 'phase-a-victims'
    budgets = []
    asrs = []
    n_per = []
    yadv = 5  # dog
    for cell, npoison, _ in cells:
        cdir = base / f'victim-{cell}'
        preds = []
        for d in sorted(cdir.iterdir() if cdir.exists() else []):
            if not d.is_dir():
                continue
            mp = d / 'metrics.jsonl'
            if not mp.exists():
                continue
            last_target = None
            for line in mp.open():
                r = json.loads(line)
                mm = r.get('metrics', {})
                if any(k.startswith('class-0-') for k in mm):
                    classes = {int(k.split('-')[-1]): v for k, v in mm.items() if k.startswith('class-0-')}
                    last_target = min(classes, key=classes.get)
            if last_target is not None:
                preds.append(last_target)
        if not preds:
            continue
        asr = sum(1 for p in preds if p == yadv) / len(preds)
        budgets.append(npoison / 50000 * 100)  # %
        asrs.append(asr * 100)
        n_per.append(len(preds))
        print(f'{cell}: npoison={npoison}, n={len(preds)}, preds={preds}, ASR={asr:.3f}')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(budgets, asrs, '-o', color='C0', markersize=10, linewidth=2,
            label=f'Our reproduction (ConvNetBN, dog-bird, n=6 each)')

    # Reference paper datapoints (read off Fig 4 right ConvNetBN dog-bird curve):
    paper_x = [0.001, 0.01, 0.1, 1.0, 10.0]
    paper_y = [2, 5, 18, 60, 90]  # approx ASR % from paper Fig 4 ConvNetBN dog-bird
    ax.plot(paper_x, paper_y, '--^', color='C3', alpha=0.6, markersize=8,
            label='Paper Fig 4 ConvNetBN dog-bird (visual estimate)')

    ax.set_xscale('log')
    ax.set_xlabel('Poison budget (%)')
    ax.set_ylabel('Attack Success Rate (%)')
    ax.set_title('Phase A reproduction: ASR vs poison budget\n(CIFAR-10 ConvNetBN, dog→bird, 200-epoch from-scratch victims)')
    ax.set_ylim(-5, 110)
    ax.set_xlim(0.005, 20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    for x, y, n in zip(budgets, asrs, n_per):
        ax.annotate(f'{y:.0f}% (n={n})', xy=(x, y), xytext=(5, -15),
                    textcoords='offset points', fontsize=9)
    fig.tight_layout()
    out = OUT / 'fig3_phase_a_asr_vs_budget.png'
    fig.savefig(out, dpi=150)
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    fig1_craft_trajectory()
    fig2_round2_tf_vs_pytorch()
    fig3_phase_a_asr_vs_budget()
