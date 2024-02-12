# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:59:11 2024

@author: GZ57NM
"""

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
scale=False

ch_type = 'Epidural'
base_path = rf'D:\RANSAC comparison\{ch_type}\Data\Preprocessed' # Specify the base directory
fig_output_path = rf'D:\RANSAC comparison\{ch_type}\Figures\temp' # Specify figure output
elec_dict = {'Epidural':'ECoG', 'Intracortical':'MEA'}

# Iterate through all subdirectories (folders) in the base directory
for file in tqdm(os.listdir(base_path), position=0, desc = 'Experiment', leave=True):
    folder_path = os.path.join(base_path, file)
    
    epochs = mne.read_epochs(folder_path)
    
    if scale:
        positions = epochs.get_montage().get_positions()
        
        ch_pos = positions['ch_pos']
    
        # Define your transformation parameters
        xa, ya = 0.001, 0.001
        xb, yb = -0.015, 0
        
        # Apply transformation to each channel's position
        transformed_ch_pos = {ch: (x * xa + xb, y * ya + yb, z) for ch, (x, y, z) in ch_pos.items()}
        
        # Create a new montage with the transformed positions
        new_montage = mne.channels.make_dig_montage(ch_pos=transformed_ch_pos, nasion=positions['nasion'], lpa=positions['lpa'], rpa=positions['rpa'])
        
        # Set the new montage to the epochs
        epochs.set_montage(new_montage)
    
    
    
    if ch_type == 'Intracortical':
        epochs = epochs['Motor']
        channel_types_dict = {ch_name: 'eeg' for ch_name in epochs.ch_names}
        epochs.set_channel_types(channel_types_dict)

    # Scaling the data within the Epochs object. Patch solution... -->  Have to go back and rescale the raw dataset
    if ch_type == 'Epidural':
        # Access the data, scale it, and then set it back
        data = epochs.get_data()  # This extracts the data as a numpy array
        scaled_data = data * 1e-6  # Scale the data
        epochs._data = scaled_data  # Set the scaled data back to the epochs object

    # Apply Ransac autoreject method
    rsc = Ransac(n_jobs=6, n_resample=50 , min_channels=0.125, min_corr=0.85, unbroken_time=0.4, verbose=False)
    rsc_fit = Ransac(n_jobs=6, n_resample=50 , min_channels=0.125, min_corr=0.85, unbroken_time=0.4, verbose=False)
    
    
    
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

    # Create the first plot for evoked_raw
    fig_raw = evoked_raw.plot_joint(show=False, times='peaks', topomap_args={'extrapolate': 'local', 'outlines':None}, ts_args ={'titles': f"{exp} - Filtered \n(std: {raw_mtrs['std']:.2e}, var: {raw_mtrs['var']:.2e}, corr: {raw_mtrs['corr']:.2f})\n {elec_dict[ch_type]} -"})
    
    # Extract y-limits from the correct axes
    y_lim = fig_raw.axes[0].get_ylim()  # Adjust the index if necessary
    
    # Save the first figure
    plt.savefig(os.path.join(fig_output_path, f'{exp}-topo_filt.pdf'))
    plt.show()
    # plt.close(fig_raw)
    
    # Create the second plot for evoked_RANSAC
    fig_rsc = evoked_RANSAC.plot_joint(show=False, times='peaks', topomap_args={'extrapolate': 'local', 'outlines':None}, ts_args ={'titles': f"{exp} - Filtered + RANSAC \n(std: {rsc_mtrs['std']:.2e}, var: {rsc_mtrs['var']:.2e}, corr: {rsc_mtrs['corr']:.2f})\n {elec_dict[ch_type]} -"})
    
    # Manually set y-limits for the time series axes of the second plot
    # fig_rsc.axes[0].set_ylim(y_lim)
    
    # Save the second figure
    plt.savefig(os.path.join(fig_output_path, f'{exp}-topo_rsc.pdf'))
    plt.show()
    # plt.close(fig_rsc)

