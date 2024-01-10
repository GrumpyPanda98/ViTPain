# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 20:11:27 2024

@author: GZ57NM
"""

import os
from general.loader import tdt_to_mne
import mne
import numpy as np

# Specify the base directory
base_path = r'D:\RANSAC comparison\Epidural'

# Iterate through all subdirectories (folders) in the base directory
for main in os.listdir(base_path):
    for folder in os.listdir(os.path.join(base_path, main)):
        folder_path = os.path.join(base_path, main, folder)
        
        # Check if the item in the directory is a subdirectory
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")
    
            raw, events = tdt_to_mne(folder_path, stream_id='Wav1', onset_id='PC0_', ch_types='eeg')
            
            # Mark unwanted channels as bad
            unwanted_channels = ['17', '18', '19', '20', '21', '23', '26']
    
            # Apply notch filter
            raw.notch_filter([50, 100, 150, 200], filter_length='auto', notch_widths=2, method='fir', n_jobs=-1)
    
            # Apply bandpass filter
            raw.filter(1, 200, l_trans_bandwidth=1, h_trans_bandwidth=10, method='fir', n_jobs=-1)
    
            epochs = mne.Epochs(raw, np.int64(np.rint(events)), tmin=-0.2, tmax=0.5, detrend=None, preload=True)
            
            epochs.save(f'{folder_path}-epo.fif', fmt='double')
            
            #%% Plot before and after interpolation 
            import matplotlib.pyplot as plt
            
            
            evoked_raw = epochs.average()
            
            exp = os.path.basename(folder_path)[-6:]
            
            # Plot evoked responses for each event side by side with custom titles
            fig, axes = plt.subplots(1, 2, figsize=(15, 10))
            fig.suptitle(f'{exp} - MEA ({evoked_raw.info["nchan"]} channels) \nBads: {", ".join(unwanted_channels)}', x=0.53, fontsize=16)
            
            # Plotting evoked responses for raw data
            evoked_raw.plot(axes=axes[0], show=False)
            axes[0].set_title('Filtered')
            
            # Exclude unwanted channels before plotting
            epochs.drop_channels(unwanted_channels)
            
            evoked_raw = epochs.average()
            
            # Plotting evoked responses for RANSAC data
            evoked_raw.plot(axes=axes[1], show=False)  # Plotting the same data since RANSAC is skipped
            axes[1].set_title('Filtered (Channels Removed)')
            
            fig.tight_layout()
            plt.show()
