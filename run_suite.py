"""Run the full experiment suite of the report (resumable, parallel on CPU).

Every run is `train.py` with one configuration; finished runs (metrics.json
present) are skipped, so the script can be re-launched after an interruption.

    python run_suite.py --dataset losloop --groups main improve tuning --jobs 4
    python run_suite.py --dataset sztaxi  --groups main improve --jobs 4

Groups
  main     : phase 1+2 — GRU, GCN, T-GCN as in the paper (3 seeds)
  improve  : phase 3   — T-GCN-RT (residual + time features) vs GRU-RT (3 seeds)
  tuning   : phase 3   — input window and hidden units of T-GCN (1 seed)
  variants : one component at a time, incl. graph removed (A = I) (1 seed)
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tgcn.utility import ROOT

SEEDS = [42, 123, 2026]
DATASET_DEFAULTS = {  # 15..60 min ahead; hidden 64 (paper's choice for Los-loop)
    'losloop': dict(horizon=12, seq_len=12, hidden=64),
    'sztaxi': dict(horizon=4, seq_len=12, hidden=64),
    'demo': dict(horizon=4, seq_len=12, hidden=16),
    'metrla': dict(horizon=12, seq_len=12, hidden=64),
    'pemsbay': dict(horizon=12, seq_len=12, hidden=64),
}
RT = ['--residual', '--time-features']


def configs(dataset, groups):
    """Returns [(run_name, train.py flags)] in priority order."""
    d = DATASET_DEFAULTS[dataset]
    base = lambda L=d['seq_len'], hd=d['hidden']: ['--horizon', str(d['horizon']),
                                                  '--seq-len', str(L), '--hidden-dim', str(hd)]
    runs = []
    if 'main' in groups:          # phase 1 + 2: paper models
        for s in SEEDS:
            for m in ('gru', 'gcn', 'tgcn'):
                runs.append((f'main/{m}/seed_{s}', ['--model', m, '--seed', str(s), *base()]))
    if 'improve' in groups:       # phase 3: improvements (3 seeds for the key pair)
        for s in (SEEDS if dataset == 'losloop' else [42]):   # SZ-taxi: secondary dataset, 1 seed
            runs.append((f'improve/tgcn_res_time/seed_{s}', ['--model', 'tgcn', *RT, '--seed', str(s), *base()]))
            runs.append((f'improve/gru_res_time/seed_{s}', ['--model', 'gru', *RT, '--seed', str(s), *base()]))
        if dataset == 'losloop':  # best graph variant gets 3 seeds too (seed 42 lives in variants/)
            for s in SEEDS[1:]:
                runs.append((f'improve/tgcn_selfsep_res_time/seed_{s}',
                             ['--model', 'tgcn', '--self-sep', *RT, '--seed', str(s), *base()]))
                runs.append((f'improve/gru_res_time_rawloss/seed_{s}',
                             ['--model', 'gru', *RT, '--raw-loss', '--seed', str(s), *base()]))
                runs.append((f'improve/tgcn_selfsep/seed_{s}',
                             ['--model', 'tgcn', '--self-sep', '--seed', str(s), *base()]))
        if dataset == 'sztaxi':
            runs.append(('improve/tgcn_res_time_masked/seed_42',
                         ['--model', 'tgcn', *RT, '--mask-zero', '--seed', '42', *base()]))
            runs.append(('improve/tgcn_masked/seed_42',
                         ['--model', 'tgcn', '--mask-zero', '--seed', '42', *base()]))
    if 'variants' in groups:      # one component at a time (seed 42)
        one = {'tgcn_selfsep_res_time': ['--model', 'tgcn', '--self-sep', *RT],
               'gru_res_time_rawloss': ['--model', 'gru', *RT, '--raw-loss'],
               'tgcn_selfsep': ['--model', 'tgcn', '--self-sep'],
               'tgcn_res_time_rawloss': ['--model', 'tgcn', *RT, '--raw-loss'],
               'tgcn_res': ['--model', 'tgcn', '--residual'],
               'tgcn_time': ['--model', 'tgcn', '--time-features'],
               'tgcn_plus': ['--model', 'tgcn_plus'],
               'tgcn_res_time_huber': ['--model', 'tgcn', *RT, '--loss', 'huber'],
               'tgcn_res_time_nograph': ['--model', 'tgcn', *RT, '--no-graph']}
        for name, flags in one.items():
            runs.append((f'variants/{name}/seed_42', [*flags, '--seed', '42', *base()]))
    if 'tuning' in groups:        # phase 3: input window and hidden units (plain T-GCN)
        for L in (6, 24):
            runs.append((f'tuning/tgcn_seq{L}/seed_42', ['--model', 'tgcn', '--seed', '42', *base(L=L)]))
        for hd in (32, 100):
            runs.append((f'tuning/tgcn_h{hd}/seed_42', ['--model', 'tgcn', '--seed', '42', *base(hd=hd)]))
        for L in (6, 24):
            runs.append((f'tuning/tgcn_res_time_seq{L}/seed_42',
                         ['--model', 'tgcn', *RT, '--seed', '42', *base(L=L)]))
    if 'modern' in groups:        # newer architectures, same protocol (3 seeds for the cheap ones)
        cheap = {'gru_rtp': ['--model', 'gru', *RT, '--profile'],
                 'tgcn_rtsp': ['--model', 'tgcn', '--self-sep', *RT, '--profile'],
                 'stid': ['--model', 'stid'], 'stid_res': ['--model', 'stid', '--residual']}
        for s in SEEDS:
            for name, flags in cheap.items():
                runs.append((f'modern/{name}/seed_{s}', [*flags, '--seed', str(s), *base()]))
        for name, flags in {'dcrnn': ['--model', 'dcrnn'], 'dcrnn_res': ['--model', 'dcrnn', '--residual']}.items():
            runs.append((f'modern/{name}/seed_42', [*flags, '--seed', '42', *base()]))
        runs.append(('modern/gwnet/seed_42', ['--model', 'gwnet', '--seed', '42', *base(), '--epochs', '60',
                                              '--patience', '12']))
        for s in SEEDS:   # the best neural model on Los-loop -> 3 seeds
            runs.append((f'modern/gwnet_res/seed_{s}', ['--model', 'gwnet', '--residual', '--seed', str(s), *base(),
                                                         '--epochs', '60', '--patience', '12']))
    if 'modern_gpu' in groups:    # papers' full sizes; meant for a local GPU (see LOCAL_RUN.md)
        for s in SEEDS:
            for name, flags in {'gwnet_full': ['--model', 'gwnet', '--size', 'full'],
                                'gwnet_full_res': ['--model', 'gwnet', '--size', 'full', '--residual'],
                                'staeformer_full': ['--model', 'staeformer', '--size', 'full'],
                                'staeformer_full_res': ['--model', 'staeformer', '--size', 'full', '--residual'],
                                'staeformer': ['--model', 'staeformer'],
                                'gwnet': ['--model', 'gwnet'], 'gwnet_res': ['--model', 'gwnet', '--residual']}.items():
                runs.append((f'modern_gpu/{name}/seed_{s}', [*flags, '--seed', str(s), *base()]))
    if 'paper' in groups:         # authors' protocol, only to reproduce Table 1
        # authors: lr 1e-3 constant, batch 32, thousands of epochs, keep min-test epoch
        hp = {'losloop': ('3', '64', '300'), 'sztaxi': ('1', '100', '150')}.get(dataset, ('3', '16', '5'))
        for m in ('gru', 'tgcn'):
            runs.append((f'paper_protocol/{m}/seed_42',
                         ['--model', m, '--protocol', 'paper', '--seed', '42', '--horizon', hp[0],
                          '--seq-len', '12', '--hidden-dim', hp[1], '--batch-size', '32',
                          '--lr', '0.001', '--lr-schedule', 'none', '--epochs', hp[2],
                          '--patience', hp[2]]))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(DATASET_DEFAULTS), default='losloop')
    p.add_argument('--spec', nargs='+', default=None,
                   help='dataset:group pairs sharing one job pool, e.g. losloop:paper sztaxi:main')
    p.add_argument('--groups', nargs='+', default=['main'],
                   choices=['main', 'improve', 'variants', 'tuning', 'paper', 'modern', 'modern_gpu'])
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--threads', type=int, default=1, help='torch threads per job')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto'])
    p.add_argument('--only', nargs='+', default=None, help='run only these run names (prefix match)')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    spec = [tuple(x.split(':')) for x in a.spec] if a.spec else [(a.dataset, g) for g in a.groups]
    runs = [(ds, n, f) for ds, g in spec for n, f in configs(ds, [g])]
    if a.only:
        runs = [(ds, n, f) for ds, n, f in runs if any(n.startswith(o) for o in a.only)]
    todo = [(ds, n, f) for ds, n, f in runs
            if not (ROOT / 'results' / ds / n / 'metrics.json').is_file()]
    print(f'{len(runs)} runs, {len(todo)} to do', flush=True)
    env = dict(os.environ, TGCN_THREADS=str(a.threads))

    def job(item):
        ds, name, flags = item
        logdir = ROOT / 'results' / ds / 'logs'; logdir.mkdir(parents=True, exist_ok=True)
        out = ROOT / 'results' / ds / name
        cmd = [sys.executable, str(ROOT / 'train.py'), '--dataset', ds,
               '--epochs', str(a.epochs), '--patience', str(a.patience), '--track-test',
               '--device', a.device, '--result-dir', str(out), *flags]
        if a.dry_run:
            print(' '.join(cmd)); return
        t0 = time.time()
        log = logdir / (name.replace('/', '__') + '.log')
        with open(log, 'w') as fh:
            r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        status = 'OK' if r.returncode == 0 else f'FAILED({r.returncode}) see {log}'
        print(f'[{time.strftime("%H:%M:%S")}] {ds}/{name}: {status} in {time.time()-t0:.0f}s', flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(job, todo))


if __name__ == '__main__':
    main()
