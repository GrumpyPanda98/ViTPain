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
import matplotlib.pyplot as plt
from tqdm import tqdm

mne.set_log_level(False)
desired_sfreq = 2000

# Specify the base directory
base_path = r'D:\RANSAC comparison\Epidural\Data\Raw'

# Iterate through all subdirectories (folders) in the base directory
for main in tqdm(os.listdir(base_path), position=0, desc = 'Experiment', leave=True):
    for folder in tqdm(os.listdir(os.path.join(base_path, main)), position=1, desc = 'Block', leave=True):
        folder_path = os.path.join(base_path, main, folder)
        
        # Check if the item in the directory is a subdirectory
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")
        
            raw, events = tdt_to_mne(folder_path,stream_id='Wav1', onset_id='PC0_', ch_types='eeg', electrode_type='ecog')
                 
            raw.notch_filter([50, 100, 150, 200], filter_length='auto', notch_widths=2, method='fir', n_jobs=-1)
            raw.filter(1, 200, l_trans_bandwidth=1, h_trans_bandwidth=10, method='fir', n_jobs=-1)
            
            epochs = mne.Epochs(raw, np.int64(np.rint(events)), tmin = -0.2, tmax = 0.5, detrend = 1, baseline=(-0.2, 0), preload=True)
            
            # Decimate to reduce size
            current_sfreq = epochs.info['sfreq']
            decim = np.round(current_sfreq / desired_sfreq).astype(int)
            obtained_sfreq = current_sfreq / decim
            
            print(
            f'desired sampling frequency was {desired_sfreq} Hz; decim factor of {decim} yielded an '
            f'actual sampling frequency of {epochs.info["sfreq"]/decim} Hz.'
            )
            
            epochs.decimate(decim=decim, verbose=True)
            
            
            
            # Apply Ransac autoreject method
            rsc = Ransac(n_jobs=-1, n_resample = 50)
            
            epochs_RANSAC = rsc.fit_transform(epochs) #https://www.frontiersin.org/articles/10.3389/fninf.2015.00016/full
            
            evoked_raw = epochs.average()
            evoked_RANSAC= epochs_RANSAC.average()
            
            
            #%% Plot figures
            exp = os.path.basename(folder_path)
            # Create a figure with 2 rows and 2 columns, but only use 3 plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 9))
            
            # Set the overall title for the figure
            fig.suptitle(f'{exp} - μECoG ({evoked_raw.info["nchan"]} channels) \nBads: {", ".join(rsc.bad_chs_)}', fontsize=16)
            
            # Plotting evoked responses for raw data in the first subplot (top-left)
            evoked_raw.plot(axes=axes[0, 0], show=False)
            axes[0, 0].set_title('Filtered')
            
            # Plotting evoked responses for RANSAC data in the second subplot (top-right)
            evoked_RANSAC.plot(axes=axes[0, 1], show=False)
            axes[0, 1].set_title('Filtered + RANSAC')
            axes[0, 1].set_ylim(axes[0, 0].get_ylim())
            
            # Plotting the heatmap in the third subplot (entire bottom row)
            axes[1, 0].remove()  # Remove the unused subplot
            axes[1, 1].remove()  # Remove the unused subplot
            heatmap_ax = fig.add_subplot(2, 1, 2)  # Add a new subplot for the heatmap
            
            # Plot the heatmap
            ch_names = [epochs.ch_names[i] for i in rsc.picks]
            im = heatmap_ax.imshow(rsc.bad_log, cmap='Reds', interpolation='nearest')
            heatmap_ax.grid(False)
            heatmap_ax.set_xlabel('Sensors')
            heatmap_ax.set_ylabel('Epochs')
            plt.setp(heatmap_ax, xticks=range(epochs_RANSAC.info['nchan']), xticklabels=ch_names)
            plt.setp(heatmap_ax.get_xticklabels(), rotation=90)
            heatmap_ax.tick_params(axis='both', which='both', length=0)
            heatmap_ax.set_aspect('auto')  # Adjust the aspect ratio for the heatmap
            heatmap_ax.set_title('Bad Epochs and Channels')
            
            # Adjust the layout
            fig.tight_layout()
            # plt.savefig(f'D:\RANSAC comparison\Epidural\Figures\RANSAC_detrend\{exp}.svg')
            plt.show()
            plt.close()
            
            # evoked_raw.plot_topomap()
            # evoked_RANSAC.plot_topomap()
    
