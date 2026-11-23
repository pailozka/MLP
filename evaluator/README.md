# Independent prediction evaluation

The Python standard-library evaluator independently checks the C++ predictions.
It reports accuracy, macro F1, per-class precision/recall/F1 and the confusion
matrix (rows: actual, columns: predicted). It rejects empty files, invalid labels,
and unequal row counts, including under `python -O`.

```bash
python3 evaluator/evaluate.py runs/evaluation/predictions.csv \
  data/prepared/fashion_mnist_test_labels.csv --json runs/independent-metrics.json
```

Both input files contain one integer class label per line, with no header.
Unlike the C++ evaluator, this utility cannot compute cross-entropy because it
receives predicted classes rather than logits or probabilities.
