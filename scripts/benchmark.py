#!/usr/bin/env python3
"""Compare a linear classifier and regularized MLP with the same training protocol."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted([*ROOT.glob('src/*.cpp'), *ROOT.glob('include/mlp/*.h'), ROOT / 'CMakeLists.txt']):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('build/release/mlp'))
    parser.add_argument('--data-dir', type=Path, default=Path('data/prepared'))
    parser.add_argument('--output', type=Path, default=Path('runs/benchmark'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    args = parser.parse_args()
    if args.epochs < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('epochs must be positive and seeds must be distinct')
    if any(seed < 0 or seed > 2**32 - 1 for seed in args.seeds):
        parser.error('seeds must fit uint32')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be new or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    binary = str(args.binary.resolve())
    manifest = args.data_dir / 'manifest.json'
    report = {
        'protocol': 'Fixed stratified 90/10 training split; best validation loss; test only after training.',
        'source_sha256': source_fingerprint(),
        'binary_sha256': hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'dataset_manifest': json.loads(manifest.read_text()) if manifest.exists() else None,
        'runs': [],
    }
    for seed in args.seeds:
        for name, hidden, dropout in [('linear', 'none', '0'), ('mlp', '128,64', '0.1')]:
            run = args.output / f'{name}-seed{seed}'
            train_command = [binary, 'train', '--data-dir', str(args.data_dir), '--output', str(run),
                             '--hidden', hidden, '--dropout', dropout, '--epochs', str(args.epochs),
                             '--seed', str(seed), '--batch-size', '64', '--lr', '0.001',
                             '--weight-decay', '0.0001', '--patience', '5', '--val-fraction', '0.1', '--clip-norm', '5']
            start = time.perf_counter()
            subprocess.run(train_command, check=True)
            training_seconds = time.perf_counter() - start
            evaluation_command = [binary, 'evaluate', '--model', str(run / 'best.mlp'),
                                  '--vectors', str(args.data_dir / 'fashion_mnist_test_vectors.csv'),
                                  '--labels', str(args.data_dir / 'fashion_mnist_test_labels.csv'),
                                  '--output', str(run / 'test')]
            subprocess.run(evaluation_command, check=True)
            entry = {
                'model': name, 'seed': seed, 'training_seconds_including_io': training_seconds,
                'train_command': train_command, 'evaluation_command': evaluation_command,
                'config': json.loads((run / 'config.json').read_text()),
                'selection': json.loads((run / 'summary.json').read_text()),
                'validation': json.loads((run / 'validation_metrics.json').read_text()),
                'test': json.loads((run / 'test/metrics.json').read_text()),
                'checkpoint_sha256': hashlib.sha256((run / 'best.mlp').read_bytes()).hexdigest(),
                'split_sha256': {kind: hashlib.sha256((run / f'{kind}_indices.csv').read_bytes()).hexdigest()
                                 for kind in ('train', 'validation')},
            }
            report['runs'].append(entry)
            # Portable, small evidence bundle, suitable for committing without model binaries.
            evidence = args.output / 'evidence'
            evidence.mkdir(exist_ok=True)
            shutil.copyfile(run / 'history.csv', evidence / f'{name}-seed{seed}-history.csv')
            (evidence / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    report['aggregate'] = {
        name: {
            'seeds': len(args.seeds),
            'mean_test_accuracy': statistics.mean(r['test']['accuracy'] for r in report['runs'] if r['model'] == name),
            'stdev_test_accuracy': statistics.stdev(r['test']['accuracy'] for r in report['runs'] if r['model'] == name)
                if len(args.seeds) > 1 else None,
        } for name in ('linear', 'mlp')
    }
    (args.output / 'evidence/report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['aggregate'], indent=2))


if __name__ == '__main__':
    main()
