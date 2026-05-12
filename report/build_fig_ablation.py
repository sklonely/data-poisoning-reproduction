"""Augmentation ablation — 2-bar version.

Shows that turning off CIFAR-10 augmentation flips the attack from "barely works"
to "decisive", holding everything else fixed (same Round 2 poison set, same
ResNet20 victim, same momentum/weight-decay, n=3 from-scratch 200-epoch trials).

Bars:
  - "augmentation ON"  = Round 2 PyTorch victim result (the 33% datapoint shown
    in fig2_round2_tf_vs_pytorch.png).      torch-results.json -> 1/3 trials
  - "augmentation OFF" = ablation job 20290350 condition B (same config, aug
    disabled).                              outputs-ablation/B_noaug_mom_wd.json -> 3/3 trials

Condition C from the original 3-bar slide (aug OFF + plain SGD) is dropped on
purpose: it also changes the optimizer, so it isn't a clean one-variable
comparison, and condition B already makes the point at 100%.

Caveat printed in the caption: "augmentation ON" is a borderline regime. An
independent re-run of that exact config (ablation condition A) landed 0/3, so
treat the 33% as a noisy point estimate of "small", not a stable property.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

ROUND2_TORCH = ROOT / 'hpc-results' / 'round2-job20241084' / 'outputs-v2' / 'torch-results.json'
ABLATION_B = ROOT / 'hpc-results' / 'staging-temp' / 'outputs-ablation' / 'B_noaug_mom_wd.json'
ABLATION_A = ROOT / 'hpc-results' / 'staging-temp' / 'outputs-ablation' / 'A_baseline_aug_mom_wd.json'


def asr_and_n(path):
    d = json.load(path.open())
    s = d['summary']
    return s['attack_success_rate_mean'], s['n_trials']


aug_on_asr, aug_on_n = asr_and_n(ROUND2_TORCH)        # 1/3 = 0.333
aug_off_asr, aug_off_n = asr_and_n(ABLATION_B)        # 3/3 = 1.0
rerun_a_asr, rerun_a_n = asr_and_n(ABLATION_A)        # 0/3 = 0.0 (footnote only)

labels = ['Augmentation ON\n(random crop + flip)', 'Augmentation OFF']
asrs = [aug_on_asr * 100, aug_off_asr * 100]
ns = [aug_on_n, aug_off_n]

fig, ax = plt.subplots(figsize=(6.5, 4.6))
bars = ax.bar(labels, asrs, color=['#9e9e9e', '#3a7d3a'], width=0.55)
ax.set_ylabel('Attack Success Rate (%)')
ax.set_ylim(0, 115)
ax.set_title('Augmentation drives the ASR gap — not the framework\n'
             'Same Round 2 poison (ResNet20, dog→bird, 5000 poisons = 10% budget)\n'
             f'PyTorch victim, n={aug_on_n} from-scratch 200-epoch trials, only the augmentation flag changes',
             fontsize=10)
for bar, val, n in zip(bars, asrs, ns):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 2.5, f'{val:.0f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width() / 2, 3, f'n={n}',
            ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

ax.axhline(100, color='#3a7d3a', linestyle=':', alpha=0.4)
ax.grid(axis='y', alpha=0.3)

# Footnote: the aug-ON regime is borderline
fig.text(0.5, -0.02,
         f'“Augmentation ON” is a borderline regime: an independent re-run of the exact same config '
         f'(ablation job 20290350, cond. A) got {int(round(rerun_a_asr*rerun_a_n))}/{rerun_a_n} — '
         f'read the 33% as a noisy estimate of “≈0”, not a stable value.',
         ha='center', va='top', fontsize=7.5, style='italic', wrap=True)

fig.tight_layout()
out = OUT / 'fig_augmentation_ablation.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
print(f'  aug ON  : ASR={aug_on_asr:.3f} (n={aug_on_n})  [Round 2 PyTorch, = fig2 datapoint]')
print(f'  aug OFF : ASR={aug_off_asr:.3f} (n={aug_off_n})  [ablation cond. B]')
print(f'  (footnote) aug ON re-run cond. A: ASR={rerun_a_asr:.3f} (n={rerun_a_n})')
plt.close(fig)
