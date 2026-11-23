#!/usr/bin/env python3
"""Independently evaluate integer class predictions using only Python's stdlib."""
import argparse
import json
from pathlib import Path


def read_labels(path, classes=10):
    values = []
    with Path(path).open(encoding='utf-8') as stream:
        for row, line in enumerate(stream, 1):
            try:
                value = int(line.strip())
            except ValueError as error:
                raise ValueError(f'{path}:{row}: expected one integer label') from error
            if not 0 <= value < classes:
                raise ValueError(f'{path}:{row}: label outside [0,{classes - 1}]')
            values.append(value)
    if not values:
        raise ValueError(f'{path}: no labels')
    return values


def evaluate(predictions, truth, classes=10):
    if classes < 2 or not truth or len(predictions) != len(truth):
        raise ValueError('Need at least two classes and nonempty, equally sized prediction/label lists')
    confusion = [[0] * classes for _ in range(classes)]
    for predicted, actual in zip(predictions, truth):
        if not 0 <= predicted < classes or not 0 <= actual < classes:
            raise ValueError('Label outside class range')
        confusion[actual][predicted] += 1
    per_class = []
    for label in range(classes):
        tp = confusion[label][label]
        actual = sum(confusion[label])
        predicted = sum(row[label] for row in confusion)
        per_class.append({
            'class': label,
            'precision': tp / predicted if predicted else 0.0,
            'recall': tp / actual if actual else 0.0,
            'f1': 2 * tp / (actual + predicted) if actual + predicted else 0.0,
            'support': actual,
        })
    return {
        'samples': len(truth),
        'accuracy': sum(confusion[i][i] for i in range(classes)) / len(truth),
        'macro_f1': sum(row['f1'] for row in per_class) / classes,
        'per_class': per_class,
        'confusion_matrix': confusion,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('truth', type=Path)
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--json', type=Path, help='Optional JSON report path')
    args = parser.parse_args()
    try:
        metrics = evaluate(read_labels(args.predictions, args.classes), read_labels(args.truth, args.classes), args.classes)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Accuracy {metrics['accuracy']:.4f} | Macro F1 {metrics['macro_f1']:.4f}")
    if args.json:
        args.json.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
