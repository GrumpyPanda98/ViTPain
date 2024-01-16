# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:14:58 2024

@author: GZ57NM
"""

import os
import mne
from autoreject import Ransac
import matplotlib.pyplot as plt
from tqdm import tqdm

mne.set_log_level(False)

# Specify the base directory
base_path = r'D:\RANSAC comparison\Epidural\Data\Preprocessed'

# Iterate through all subdirectories (folders) in the base directory
for file in tqdm(os.listdir(base_path), position=0, desc = 'Experiment Block', leave=True):
    folder_path = os.path.join(base_path, file)
    
    epochs = mne.read_epochs(folder_path)
    
    for n_resample in [50]:

        # Apply Ransac autoreject method
        rsc = Ransac(n_jobs=6, n_resample=n_resample , min_channels=0.125, min_corr=0.85, unbroken_time=0.4)
        
        epochs_RANSAC = rsc.fit_transform(epochs) #https://www.frontiersin.org/articles/10.3389/fninf.2015.00016/full
        
        evoked_raw = epochs.average()
        evoked_RANSAC= epochs_RANSAC.average()
        
        
        #%% Plot figures
        exp = os.path.basename(folder_path)
        # Create a figure with 2 rows and 2 columns, but only use 3 plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        
        # Set the overall title for the figure
        fig.suptitle(f'{exp} - μECoG ({evoked_raw.info["nchan"]} channels) \nBads: {", ".join(rsc.bad_chs_)} Resamples: {rsc.n_resample}', fontsize=16)
        
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

    
