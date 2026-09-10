import unittest

import mne
import numpy as np

from research.ransac_comparison import compare, signal_metrics, validate_epochs


class RansacComparisonTests(unittest.TestCase):
    def epochs(self):
        names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']
        info = mne.create_info(names, 200, 'eeg')
        info.set_montage(mne.channels.make_standard_montage('standard_1020'))
        rng = np.random.default_rng(4)
        signal = np.sin(np.linspace(0, 4*np.pi, 80))[None, None, :] * 2e-5
        data = np.broadcast_to(signal, (8, len(names), 80)).copy() + rng.normal(0, 1e-7, (8, len(names), 80))
        return mne.EpochsArray(data, info, tmin=-.1, baseline=None, verbose=False)

    def test_requires_reviewed_positions(self):
        epochs = self.epochs()
        epochs.set_montage(None)
        with self.assertRaisesRegex(ValueError, 'montage'):
            validate_epochs(epochs)

    def test_metrics_are_reported_in_microvolt_units(self):
        epochs = self.epochs()
        metrics = signal_metrics(epochs)
        self.assertAlmostEqual(metrics['sd_uV'] ** 2, metrics['variance_uV2'])

    def test_comparison_preserves_input_geometry_and_samples(self):
        epochs = self.epochs()
        epochs._data[:, 0] = np.random.default_rng(5).normal(0, 1e-4, (len(epochs), len(epochs.times)))
        before = epochs.get_data().copy()
        corrected, report = compare(epochs, 'epidural-paper')
        np.testing.assert_array_equal(epochs.get_data(), before)
        np.testing.assert_array_equal(corrected.times, epochs.times)
        self.assertEqual(corrected.ch_names, epochs.ch_names)
        self.assertEqual(report['parameters']['n_resample'], 7)
        self.assertIn('Fp1', report['detected_bad_channels'])
        self.assertFalse(np.array_equal(corrected.get_data()[:, 0], before[:, 0]))
        self.assertTrue(np.isfinite(report['ransac']['reconstruction_correlation']))

if __name__ == '__main__':
    unittest.main()
