# -*- coding: utf-8 -*-
"""
Enhanced Script for Processing EEG Data
Created on Fri Jan 12 14:55:45 2024
@author: GZ57NM
"""

import os
import mne
import numpy as np
from tqdm import tqdm
from general.loader import tdt_to_mne

# Configuration
MNE_LOG_LEVEL = False
DESIRED_SFREQ = 2000
BASE_PATH = r'D:\RANSAC comparison\Epidural\Data\Raw'
PREPROCESSED_PATH = r'D:\RANSAC comparison\Epidural\Data\Preprocessed'

# Function to process each folder
def process_folder(folder_path):
    try:
        print(f"Processing folder: {folder_path}")
        raw, events = tdt_to_mne(folder_path, stream_id='Wav1', onset_id='PC0_', ch_types='eeg', electrode_type='ecog')
        
        # Filters and Epochs
        raw.notch_filter([50, 100, 150, 200], filter_length='auto', notch_widths=2, method='fir', n_jobs=-1)
        raw.filter(1, 200, l_trans_bandwidth=1, h_trans_bandwidth=10, method='fir', n_jobs=-1)
        epochs = mne.Epochs(raw, np.int64(np.rint(events)), tmin=-0.2, tmax=0.5, detrend=1, baseline=(-0.2, 0), preload=True)
        
        # Decimation
        decim = round(epochs.info['sfreq'] / DESIRED_SFREQ)
        epochs.decimate(decim=decim, verbose=True)
        print(f'Decimated to {raw.info["sfreq"]/decim} Hz with a factor of {decim}.')
        
        # Save Processed Data
        processed_file_path = os.path.join(PREPROCESSED_PATH, f'{os.path.basename(folder_path)}-epo.fif')
        epochs.save(processed_file_path, fmt='double')
        print(f"Saved processed data to {processed_file_path}")
    except Exception as e:
        print(f"Error processing {folder_path}: {e}")

# Main Script
if __name__ == "__main__":
    mne.set_log_level(MNE_LOG_LEVEL)

    for main_dir in tqdm(os.listdir(BASE_PATH), position=0, desc='Experiment', leave=True):
        for folder in tqdm(os.listdir(os.path.join(BASE_PATH, main_dir)), position=1, desc='Block', leave=True):
            folder_path = os.path.join(BASE_PATH, main_dir, folder)
            if os.path.isdir(folder_path):
                process_folder(folder_path)
