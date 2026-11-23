# Recorded reference experiment

This directory contains actual results from:

```bash
python3 scripts/benchmark.py --epochs 20 --output runs/reference
```

The default seed was 42. The binary was built with GCC 15.2.0 and CMake 4.2.3 in
Release mode on Linux x86-64, Intel Core i5-1240P. No accelerator or external
numerical library was used. `report.json` records complete hyperparameters,
selection epochs, validation/test metrics, source and binary fingerprints,
checkpoint hashes, split hashes, and the official dataset's checksums. Command
paths in the report identify the machine where it ran; substitute your checkout
path when reproducing. Histories record every executed epoch.

| Model | Test accuracy | Test macro F1 | Best epoch | Epochs executed |
| --- | ---: | ---: | ---: | ---: |
| Linear | 0.8457 | 0.845955 | 18 | 20 |
| MLP 128/64 | 0.8833 | 0.882824 | 12 | 17 |

Both models use exactly the same split memberships (verify `split_sha256`). The
10,000-sample test set is used only after validation-based checkpoint selection.
The MLP's prediction metrics were independently checked with
`evaluator/evaluate.py`; its confusion matrix and accuracy matched exactly, with
macro F1 agreeing within JSON serialization precision.

This is a one-seed reference, not a significance study or an original-model
comparison. Timings include file I/O and were collected on a shared machine.
The generated model checkpoints, prediction files and full split indices remain
under ignored `runs/reference/`; they can be regenerated using the command above.
