"""Exercise complete training, repeatability and reloaded inference without downloads."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve())


def run(*args, success=True):
    result = subprocess.run([binary, *map(str, args)], capture_output=True, text=True)
    if (result.returncode == 0) != success:
        raise AssertionError(f'{args}: code={result.returncode}\n{result.stdout}\n{result.stderr}')
    return result


with tempfile.TemporaryDirectory(prefix='mlp-cli-') as temporary:
    root = Path(temporary)
    vectors = root / 'fashion_mnist_train_vectors.csv'
    labels = root / 'fashion_mnist_train_labels.csv'
    with vectors.open('w') as x, labels.open('w') as y:
        for _ in range(10):
            for label in range(10):
                pixels = ['0'] * 784
                pixels[label] = '255'
                x.write(','.join(pixels) + '\n')
                y.write(f'{label}\n')
    for name in ('first', 'second'):
        run('train', '--data-dir', root, '--output', root / name, '--hidden', '16',
            '--dropout', '0.1', '--epochs', '8', '--batch-size', '17', '--lr', '0.03')
    first, second = root / 'first', root / 'second'
    for name in ('best.mlp', 'train_indices.csv', 'validation_indices.csv', 'config.json', 'summary.json', 'validation_metrics.json'):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    history = list(csv.DictReader((first / 'history.csv').open()))
    assert float(history[-1]['train_loss']) < float(history[0]['train_loss'])
    run('evaluate', '--model', first / 'best.mlp', '--vectors', vectors, '--labels', labels, '--output', root / 'evaluation')
    run('predict', '--model', first / 'best.mlp', '--vectors', vectors, '--output', root / 'predictions.csv')
    assert (root / 'predictions.csv').read_bytes() == (root / 'evaluation/predictions.csv').read_bytes()
    metrics = json.loads((root / 'evaluation/metrics.json').read_text())
    assert metrics['samples'] == 100 and metrics['accuracy'] == 1.0
    assert len(metrics['confusion_matrix']) == 10
    run('train', '--data-dir', root, '--output', first, success=False)
    for args in [('bad-command',), ('train', '--epochs', '-1'), ('train', '--epochs', '2junk'),
                 ('train', '--hidden', '16,'), ('train', '--lr', 'nan'),
                 ('train', '--seed', '4294967296'), ('train', '--epohs', '1'),
                 ('train', '--epochs'), ('train', '--epochs', '1', '--epochs', '2'),
                 ('predict',), ('evaluate', '--model', root / 'missing')]:
        run(*args, success=False)
    run('--help')
print('CLI training, deterministic artifacts, evaluation and error handling passed')
