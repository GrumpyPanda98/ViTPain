"""Exercise TDT conversion using a synthetic block, never animal data."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from general.loader import tdt_to_mne


class LoaderTests(unittest.TestCase):
    def block(self):
        return SimpleNamespace(
            streams={
                "Wav1": SimpleNamespace(
                    data=np.zeros((2, 100)), fs=1000, channel=[1, 2]
                )
            },
            epocs={"PC0_": SimpleNamespace(onset=np.array([0.01, 0.02]))},
        )

    def test_conversion_without_montage(self):
        with patch("tdt.read_block", return_value=self.block()):
            raw, events = tdt_to_mne("synthetic", "Wav1", "PC0_")
        self.assertEqual(raw.get_data().shape, (2, 100))
        np.testing.assert_array_equal(events, [[10, 0, 1], [20, 0, 1]])
        self.assertIsNone(raw.get_montage())

    def test_missing_stream_id_fails_before_read(self):
        with patch("tdt.read_block") as read:
            with self.assertRaisesRegex(ValueError, "stream_id"):
                tdt_to_mne("synthetic")
            read.assert_not_called()

    def test_wrong_channel_count_fails_clearly(self):
        with patch("tdt.read_block", return_value=self.block()):
            with self.assertRaisesRegex(ValueError, "32 channels"):
                tdt_to_mne("synthetic", "Wav1", "PC0_", electrode_type="ecog")


if __name__ == "__main__":
    unittest.main()
