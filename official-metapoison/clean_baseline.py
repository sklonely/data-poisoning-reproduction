"""TF clean baseline victim — train a clean victim on un-poisoned CIFAR-10
and measure target image classifications. Used for Fig 4's "0 poison" point.

Usage:
  python clean_baseline.py UID -net ConvNetBN -targetclass 2 -ytargetadv 5 \
    -nvictimepoch 200 -ntrial 6 -targetids 0 1 2 3 4 -artifactroot ...
"""
print('loading modules clean_baseline')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tf_compat import tf
from parse import get_parser
from meta import Meta
from data import *
from utils import *
from tracking import LocalExperiment
import json
import numpy as np
from time import time
from local_mpi import MPI

mpi = MPI.COMM_WORLD
nproc, rank = mpi.Get_size(), mpi.Get_rank()
localrank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank))

parser = get_parser()
parser.add_argument('-ntrial', default=6, type=int)
parser.add_argument('-nvictimepoch', default=200, type=int)
parser.add_argument('-output_json', default='clean_baseline.json', type=str)
args = parser.parse_args()
args.gpu = set_available_gpus(args)

print('==> loading data')
xtrain, ytrain, xvalid, yvalid, xbase, ybase, xtarget, ytarget, ytargetadv = load_and_apportion_data(mpi, args)
print(f'==> {len(xtarget)} target images, true class={args.targetclass}, adv class={args.ytargetadv}')

print('==> building graph')
meta = Meta(args, xbase, ybase, xtarget, ytarget, ytargetadv, victim=True)

gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(localrank % len(args.gpu)))
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
pretrain_weights = meta.global_initialize(args, sess)
sess.graph.finalize()

results = []
for trial in range(args.ntrial):
    print(f'\n=== trial {trial} ===')
    meta.global_initialize(args, sess)  # reset weights
    tic = time()
    for epoch in range(args.nvictimepoch):
        lrnrate = lr_schedule(args.lrnrate, epoch, args.warmupperiod, args.schedule)
        # NO poison loaded — meta.poisoninputs is whatever it init'd to (zeros / clean)
        # The trick is to NOT call meta.poisoninputs.load() so poison locations are clean
        for victimfeed in feeddict_generator(xtrain, ytrain, lrnrate, meta, args, victim=True):
            sess.run([meta.trainop], victimfeed)
        if epoch % 50 == 0 or epoch == args.nvictimepoch - 1:
            # Eval on target images
            resVs = []
            for _, validfeed, _ in feeddict_generator(xvalid, yvalid, lrnrate, meta, args, valid=True):
                resV = sess.run(meta.resultV, validfeed)
                resVs.append(resV)
            resV_avg = avg_n_dicts(resVs)
            # Extract target predictions from class-{tid}-{cls} keys (lowest = predicted)
            target_preds = []
            for tid in range(args.ntarget):
                losses = {cls: resV_avg[f'class-{tid}-{cls}'] for cls in range(10) if f'class-{tid}-{cls}' in resV_avg}
                if losses:
                    target_preds.append(min(losses, key=losses.get))
            acc_val = resV_avg.get("acc")
            acc_str = f'{acc_val:.3f}' if isinstance(acc_val, (int, float)) else str(acc_val)
            print(f'  epoch {epoch}: target preds {target_preds}, valid acc {acc_str}, elapsed {round(time()-tic,1)}s')
    results.append({'trial': trial, 'final_target_preds': target_preds})

# Aggregate
all_preds = [p for r in results for p in r['final_target_preds']]
adv_count = sum(1 for p in all_preds if p == args.ytargetadv)
asr = adv_count / len(all_preds) if all_preds else 0
summary = {
    'arch': args.net,
    'target_class': args.targetclass,
    'adv_class': args.ytargetadv,
    'n_trials': args.ntrial,
    'n_targets': args.ntarget,
    'total_votes': len(all_preds),
    'baseline_asr_pct': asr * 100,
    'pred_distribution': {c: all_preds.count(c) for c in range(10)},
}
print('\n=== FINAL ===')
print(json.dumps(summary, indent=2))
with open(args.output_json, 'w') as f:
    json.dump({'summary': summary, 'trials': results, 'args': vars(args)}, f, indent=2, default=str)
print(f'wrote {args.output_json}')
