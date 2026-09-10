"""Synthetic regression checks without loading experimental recordings."""

import unittest
from unittest.mock import patch

import numpy as np
from ml.random_forest import extract_features, train_random_forest


class RandomForestTests(unittest.TestCase):
    def test_fits_one_complete_forest(self):
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])
        with patch("ml.random_forest.RandomForestClassifier") as forest:
            result = train_random_forest(x, y)
            forest.assert_called_once_with(n_estimators=100, random_state=42, n_jobs=-1)
            forest.return_value.fit.assert_called_once_with(x, y)
            self.assertIs(result, forest.return_value)

    def test_feature_flattening_preserves_trial_labels(self):
        class Epochs:
            events = np.array([[0, 0, 1], [10, 0, 2]])

            def get_data(self):
                return np.arange(12).reshape(2, 2, 3)

        x, y = extract_features(Epochs())
        np.testing.assert_array_equal(x, np.arange(12).reshape(2, 6))
        np.testing.assert_array_equal(y, [1, 2])

    def test_small_forest_can_predict(self):
        x = np.arange(40).reshape(20, 2)
        y = np.repeat([0, 1], 10)
        model = train_random_forest(x, y)
        self.assertEqual(len(model.estimators_), 100)
        self.assertEqual(model.predict(x[:2]).shape, (2,))


if __name__ == "__main__":
    unittest.main()
