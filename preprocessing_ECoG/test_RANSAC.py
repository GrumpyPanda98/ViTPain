# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:28:24 2024

@author: GZ57NM
"""

import mne
from autoreject import Ransac
import numpy as np

# Configuration
mne.set_log_level(False)
BASE_PATH = r'D:\RANSAC comparison\intracortical\Data\Preprocessed\Experiment 09-epo.fif' 

epochs = mne.read_epochs(BASE_PATH, preload=True)

epochs = epochs['Motor']
channel_types_dict = {ch_name: 'eeg' for ch_name in epochs.ch_names}
epochs.set_channel_types(channel_types_dict)

# Initialize Ransac outside the loop for efficiency
rsc = Ransac(n_jobs=6, n_resample=50, min_channels=0.125, min_corr=0.85, unbroken_time=0.4)

# Apply Ransac autoreject method
epochs_RANSAC = rsc.fit_transform(epochs)



raw_fit = rsc.fit(epochs)
raw_corr = np.mean(raw_fit.corr_)
raw_var = np.var(epochs)
raw_std = np.std(epochs_RANSAC)

RANSAC_fit = rsc.fit(epochs_RANSAC)
rsc_corr = np.mean(RANSAC_fit.corr_)
rsc_var = np.var(epochs_RANSAC)
rsc_std = np.std(epochs_RANSAC)


