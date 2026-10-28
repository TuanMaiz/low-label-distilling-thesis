"""Gold audit of the category rule (plan 261023 Phase 4)."""
import unittest

from analysis.category_rule_gold_audit import audit


class GoldAuditTests(unittest.TestCase):
    def test_counts_per_action_and_detection(self):
        rows = [
            {'pair_id': 'a', 'detected': 'True', 'step': '2', 'action': 'flip', 'original_label': 'match', 'final_label': 'non_match'},
            {'pair_id': 'b', 'detected': 'True', 'step': '3', 'action': 'remove', 'original_label': 'match', 'final_label': ''},
            {'pair_id': 'c', 'detected': 'False', 'step': '', 'action': 'keep', 'original_label': 'non_match', 'final_label': 'non_match'},
        ]
        gold = {'a': 'non_match', 'b': 'match', 'c': 'match'}
        r = audit(rows, gold)
        self.assertEqual(r['by_step_and_action']['step2:flip'], {'pairs': 1, 'teacher_correct': 0, 'final_correct': 1, 'gold_match': 0})
        self.assertEqual(r['by_step_and_action']['step3:remove'], {'pairs': 1, 'teacher_correct': 1, 'final_correct': 0, 'gold_match': 1})
        self.assertEqual(r['teacher_error_rate']['not_detected']['rate'], 1.0)
        self.assertEqual(r['teacher_error_rate']['detected']['rate'], 0.5)

    def test_mismatched_pairs_rejected(self):
        with self.assertRaises(ValueError):
            audit([{'pair_id': 'a', 'detected': 'False', 'step': '', 'action': 'keep', 'original_label': 'match', 'final_label': 'match'}], {})


if __name__ == '__main__':
    unittest.main()
