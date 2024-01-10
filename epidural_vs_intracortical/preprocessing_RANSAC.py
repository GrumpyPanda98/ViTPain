# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:14:58 2024

@author: GZ57NM
"""

import os
from general.loader import tdt_to_mne
import mne
import numpy as np
from autoreject import Ransac

desired_sfreq = 2000  # Hz

# Specify the base directory
base_path = r'D:\RANSAC comparison\Epidural\Data\Raw\Suzan_Chronic_PZA10-230413'

# Iterate through all subdirectories (folders) in the base directory
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    
    # Check if the item in the directory is a subdirectory
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_path}")

        raw, events = tdt_to_mne(folder_path,stream_id='Wav1', onset_id='PC0_', ch_types='eeg', electrode_type='ecog')
        
        # Mark unwanted channels as bad
        # unwanted_channels = ['17', '18', '19', '20', '21', '23', '26']
        # raw.info['bads'] = unwanted_channels

        
        raw.notch_filter([50, 100, 150, 200], filter_length='auto', notch_widths=2, method='fir', n_jobs=-1)
        raw.filter(1, 200, l_trans_bandwidth=1, h_trans_bandwidth=10, method='fir', n_jobs=-1)
        
        epochs = mne.Epochs(raw, np.int64(np.rint(events)), tmin = -0.2, tmax = 0.5, detrend = 1, baseline=(-0.2, -0.5), preload=True)
        
        current_sfreq = epochs.info["sfreq"]
        decim = np.round(current_sfreq / desired_sfreq).astype(int)
        obtained_sfreq = current_sfreq / decim
        
        print(
        f'desired sampling frequency was {desired_sfreq} Hz; decim factor of {decim} yielded an '
        f'actual sampling frequency of {epochs.info["sfreq"]/decim} Hz.'
        )
        
        epochs.decimate(decim=decim, verbose=True)
        
        # Apply Ransac autoreject method
        rsc = Ransac(n_jobs=-1)
        
        epochs_RANSAC = rsc.fit_transform(epochs) #https://www.frontiersin.org/articles/10.3389/fninf.2015.00016/full
        
        #%% Plot before and after interpolation 
        import matplotlib.pyplot as plt
        
        
        evoked_raw = epochs.average()
        evoked_RANSAC= epochs_RANSAC.average()
        
        exp = os.path.basename(folder_path)[-6:]
        
        # Plot evoked responses for each event side by side with custom titles
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        fig.suptitle(f'{exp} - MEA ({evoked_raw.info["nchan"]} channels) \nBads: {", ".join(rsc.bad_chs_)}', x=0.53, fontsize=16)
        
        # Plotting evoked responses for raw data
        evoked_raw.plot(axes=axes[0], show=False)
        axes[0].set_title('Filtered')
        
        # Plotting evoked responses for RANSAC data
        evoked_RANSAC.plot(axes=axes[1], show=False)
        axes[1].set_title('Filtered + RANSAC')
        
        fig.tight_layout()
        plt.show()

        #%% Plot heatmap
        
        ch_names = [epochs.ch_names[ii] for ii in rsc.picks]
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        fig.suptitle(f'{exp} - MEA ({evoked_raw.info["nchan"]} channels) \nBads: {", ".join(rsc.bad_chs_)}', x=0.53, fontsize=16)
        
        ax.imshow(rsc.bad_log, cmap='Reds',
                  interpolation='nearest')
        ax.grid(False)
        ax.set_xlabel('Sensors')
        ax.set_ylabel('Epochs')
        plt.setp(ax, xticks=range(epochs_RANSAC.info['nchan']),
                 xticklabels=ch_names)
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.tick_params(axis=u'both', which=u'both', length=0)
        fig.tight_layout(rect=[None, None, None, 1.1])
        
        fig.gca().set_aspect('auto')  # Adjust the aspect ratio as needed
        fig.tight_layout()
        plt.show