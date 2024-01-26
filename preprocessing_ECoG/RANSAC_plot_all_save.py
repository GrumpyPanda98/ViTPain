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
import numpy as np

# CONFIGURATION
mne.set_log_level(False)

ch_type = 'Intracortical'
base_path = rf'D:\RANSAC comparison\{ch_type}\Data\Preprocessed' # Specify the base directory
fig_output_path = rf'D:\RANSAC comparison\{ch_type}\Figures\temp' # Specify figure output

elec_dict = {'Epidural':'ECoG', 'Intracortical':'MEA'}


# Iterate through all subdirectories (folders) in the base directory
for file in tqdm(os.listdir(base_path), position=0, desc = 'Experiment', leave=True):
    folder_path = os.path.join(base_path, file)
    
    epochs = mne.read_epochs(folder_path)
    
    
    if ch_type == 'Intracortical':
        epochs = epochs['Motor']
        channel_types_dict = {ch_name: 'eeg' for ch_name in epochs.ch_names}
        epochs.set_channel_types(channel_types_dict)

    # Apply Ransac autoreject method
    rsc = Ransac(n_jobs=6, n_resample=50 , min_channels=0.25, min_corr=0.75, unbroken_time=0.4, verbose=False)
    rsc_fit = Ransac(n_jobs=6, n_resample=50 , min_channels=0.25, min_corr=0.75, unbroken_time=0.4, verbose=False)
    
    
    
    epochs_RANSAC = rsc.fit_transform(epochs) #https://www.frontiersin.org/articles/10.3389/fninf.2015.00016/full
    
    evoked_raw = epochs.average()
    evoked_RANSAC= epochs_RANSAC.average()
    
    # Metrics
    raw_mtrs = {
    "corr": np.mean(rsc.corr_),
    "var": np.var(epochs),
    "std": np.std(epochs)
    }

    rsc_mtrs = {
    "corr": np.mean(rsc_fit.fit(epochs_RANSAC).corr_),
    "var": np.var(epochs_RANSAC),
    "std": np.std(epochs_RANSAC)
    }
    
    #%% Plot figures
    exp = os.path.basename(folder_path).replace("-epo.fif", "")
    # Create a figure with 2 rows and 2 columns, but only use 3 plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    
    # Set the overall title for the figure
    fig.suptitle(f'{exp} - {elec_dict[ch_type]} ({evoked_raw.info["nchan"]} channels) \nBads: {", ".join(rsc.bad_chs_)}', fontsize=16)
    
    # Plotting evoked responses for raw data in the first subplot (top-left)
    evoked_raw.plot(axes=axes[0, 0], show=False)
    axes[0, 0].set_title(f"Filtered \n(std: {raw_mtrs['std']:.2e}, var: {raw_mtrs['var']:.2e}, corr: {raw_mtrs['corr']:.2f})")
    
    
    # Plotting evoked responses for RANSAC data in the second subplot (top-right)
    evoked_RANSAC.plot(axes=axes[0, 1], show=False)
    axes[0, 1].set_title(f"Filtered + RANSAC \n(std: {rsc_mtrs['std']:.2e}, var: {rsc_mtrs['var']:.2e}, corr: {rsc_mtrs['corr']:.2f})")
    axes[0, 1].set_ylim(axes[0, 0].get_ylim())
    
    # Plotting the heatmap in the third subplot (entire bottom row)
    axes[1, 0].remove()  # Remove the unused subplot
    axes[1, 1].remove()  # Remove the unused subplot
    
    heatmap_ax = fig.add_subplot(2, 1, 2)  # Add a new subplot for the heatmap
    
    # Plot the heatmap
    ch_names = [epochs.ch_names[i] for i in rsc.picks]
    im = heatmap_ax.imshow(rsc.bad_log, cmap='Reds', interpolation='nearest')
    heatmap_ax.grid(False)
    heatmap_ax.set_xlabel('Channel')
    heatmap_ax.set_ylabel('Epochs')
    plt.setp(heatmap_ax, xticks=range(epochs_RANSAC.info['nchan']), xticklabels=ch_names)
    plt.setp(heatmap_ax.get_xticklabels(), rotation=90)
    heatmap_ax.tick_params(axis='both', which='both', length=0)
    heatmap_ax.set_aspect('auto')  # Adjust the aspect ratio for the heatmap
    heatmap_ax.set_title('Bad Epochs and Channels')
    
    # Adjust the layout
    fig.tight_layout()
    plt.savefig(os.path.join(fig_output_path, f'{exp}.pdf'))
    plt.show()
    plt.close()
    
    # evoked_raw.plot_topomap()
    # evoked_RANSAC.plot_topomap()


