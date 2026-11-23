import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_data', ROOT / 'scripts/prepare_data.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
spec_eval = importlib.util.spec_from_file_location('evaluate', ROOT / 'evaluator/evaluate.py')
evaluator = importlib.util.module_from_spec(spec_eval)
spec_eval.loader.exec_module(evaluator)


class PreparationTests(unittest.TestCase):
    def test_idx_conversion(self):
        images = struct.pack('>IIII', 2051, 2, 28, 28) + bytes([0] * 784 + [255] * 784)
        labels = struct.pack('>II', 2049, 2) + bytes([0, 9])
        with tempfile.TemporaryDirectory() as temporary:
            paths, count = prepare.convert(images, labels, 'test', Path(temporary))
            self.assertEqual(count, 2)
            self.assertEqual(paths[1].read_text(), '0\n9\n')
            self.assertEqual(paths[0].read_text().splitlines()[1], ','.join(['255'] * 784))
            self.assertEqual(len(prepare.sha256(paths[0])), 64)

    def test_rejects_malformed_idx(self):
        for payload in (b'', struct.pack('>IIII', 1, 1, 28, 28), struct.pack('>IIII', 2051, 1, 28, 28)):
            with self.assertRaises(ValueError):
                prepare.decode_images(payload)
        for payload in (b'', struct.pack('>II', 2049, 1) + bytes([10]), struct.pack('>II', 2049, 2) + bytes([1])):
            with self.assertRaises(ValueError):
                prepare.decode_labels(payload)


class EvaluationTests(unittest.TestCase):
    def test_known_confusion(self):
        metrics = evaluator.evaluate([0, 0, 1, 1], [0, 1, 1, 2], classes=3)
        self.assertEqual(metrics['accuracy'], 0.5)
        self.assertEqual(metrics['confusion_matrix'], [[1, 0, 0], [1, 1, 0], [0, 1, 0]])
        self.assertAlmostEqual(metrics['macro_f1'], (2 / 3 + 0.5) / 3)
        self.assertEqual(metrics['per_class'][2]['precision'], 0)

    def test_invalid_input(self):
        for predictions, truth in [([], []), ([0], [0, 1]), ([10], [0]), ([0], [-1])]:
            with self.assertRaises(ValueError):
                evaluator.evaluate(predictions, truth)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'labels.csv'
            for text in ['', 'nan\n', '1.0\n', '10\n', '1,2\n']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    evaluator.read_labels(path)


if __name__ == '__main__':
    unittest.main()
