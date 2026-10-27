"""Portable launcher using explicit local/remote paths from a site JSON."""
import argparse
import json
import os
from pathlib import Path


def command(site, dataset, episodes, out, *, max_steps=675, keep_rgb=False):
    root, repo = Path(site['root']), Path(site['repo'])
    names = ('train.json.gz', 'train_gt.json.gz') if dataset == 'r2r' else ('train_guide.json.gz', 'train_guide_gt.json.gz')
    argv = [site['sim_python'], '-u', '-m', 's8n_dagger.collect', '--dataset', dataset,
            '--data-path', str(root / 'assets' / dataset / names[0]),
            '--gt-path', str(root / 'assets' / dataset / names[1]),
            '--scenes-dir', site['scenes'], '--episode-list', str(episodes),
            '--checkpoint', site['checkpoint'], '--model-path', site['base'],
            '--model-python', site['model_python'], '--habitat-config',
            str(repo / 'configs/eval' / ('vln_r2r.yaml' if dataset == 'r2r' else 'vln_rxr15.yaml')),
            '--simulator-source', str(repo / 'docs/evaluation_setup'),
            '--max-steps', str(max_steps), '--out', str(out)]
    if keep_rgb:
        argv.append('--keep-rejected-rgb')
    return argv


def environment(site):
    env = os.environ.copy()
    env.update(PYTHONPATH=site['source'] + '/src:' + site['repo'], OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0',
               MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet')
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', required=True)
    parser.add_argument('--dataset', choices=['r2r', 'rxr15'], required=True)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-steps', type=int, default=675)
    parser.add_argument('--keep-rgb', action='store_true')
    args = parser.parse_args()
    site = json.loads(Path(args.site).read_text())
    argv = command(site, args.dataset, args.episodes, args.out,
                   max_steps=args.max_steps, keep_rgb=args.keep_rgb)
    os.chdir(site['repo'])
    os.execve(argv[0], argv, environment(site))


if __name__ == '__main__':
    main()
