# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:14:58 2024

@author: GZ57NM
"""

import os
import mne
from autoreject import Ransac
from tqdm import tqdm

# Configuration
mne.set_log_level(False)
BASE_PATH = r'D:\RANSAC comparison\Epidural\Data\Preprocessed'
RANSAC_PATH = r'D:\RANSAC comparison\Epidural\Data\Preprocessed + RANSAC'

# Initialize Ransac outside the loop for efficiency
rsc = Ransac(n_jobs=6, n_resample=50, min_channels=0.125, min_corr=0.85, unbroken_time=0.4)

# Iterate through all subdirectories (folders) in the base directory
for file in tqdm(os.listdir(BASE_PATH), position=0, desc='Experiment Block', leave=True):
    folder_path = os.path.join(BASE_PATH, file)
    epochs = mne.read_epochs(folder_path, preload=True)

    # Apply Ransac autoreject method
    epochs_RANSAC = rsc.fit_transform(epochs)

    # Save the preprocessed data
    output_path = os.path.join(RANSAC_PATH, 'RANSAC_' + os.path.basename(folder_path))
    epochs_RANSAC.save(output_path, fmt='double')
