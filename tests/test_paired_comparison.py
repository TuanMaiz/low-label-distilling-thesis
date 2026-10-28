"""Paired bootstrap and McNemar for plan 261023 Phase 4."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.paired_comparison import aligned, compare, match_f1, mcnemar_exact, paired_bootstrap


def write(path, rows):
    path.write_text(''.join(json.dumps({'pair_id': f'p{i}', 'label': y, 'prediction': p, 'is_valid': True}) + '\n'
                            for i, (y, p) in enumerate(rows)))


class PairedComparisonTests(unittest.TestCase):
    def test_match_f1(self):
        y = np.array([1, 1, 0, 0]); p = np.array([1, 0, 1, 0])
        self.assertAlmostEqual(match_f1(y, p), 0.5)

    def test_identical_predictions_give_zero_interval(self):
        y = np.array([1, 0] * 50); p = np.array([1, 1] * 50)
        r = paired_bootstrap(y, p, p, resamples=200)
        self.assertEqual((r['delta_f1'], r['ci_low'], r['ci_high']), (0.0, 0.0, 0.0))

    def test_bootstrap_is_deterministic_and_brackets_the_estimate(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 500); a = np.where(rng.random(500) < 0.9, y, 1 - y); b = np.where(rng.random(500) < 0.8, y, 1 - y)
        r1, r2 = paired_bootstrap(y, a, b, resamples=300), paired_bootstrap(y, a, b, resamples=300)
        self.assertEqual(r1, r2)
        self.assertLess(r1['ci_low'], r1['delta_f1'])
        self.assertLess(r1['delta_f1'], r1['ci_high'])

    def test_mcnemar_exact_known_values(self):
        y = np.zeros(5, dtype=int)
        self.assertAlmostEqual(mcnemar_exact(y, np.zeros(5, dtype=int), np.ones(5, dtype=int))['p_value'], 0.0625)
        self.assertEqual(mcnemar_exact(y, y, y)['p_value'], 1.0)

    def test_misaligned_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            a, b, c = Path(t, 'a'), Path(t, 'b'), Path(t, 'c')
            write(a, [(1, 1), (0, 0)]); write(b, [(1, 1)]); write(c, [(0, 1), (0, 0)])
            with self.assertRaises(ValueError):
                compare(a, b)
            with self.assertRaises(ValueError):
                compare(a, c)


if __name__ == '__main__':
    unittest.main()
