# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:47:20 2023

@author: GZ57NM
"""
import numpy as np
import mne
import plyer as pl
from autoreject import Ransac
import os


#%% Load MNE epochs object
desired_sfreq = 2000  # Hz

paths=pl.filechooser.open_file(filters=[('MNE fif (*.fif)', '*.fif')], multiple = True)

for path in paths:
    exp = os.path.basename(path)[:-8]
   
    # load epoch fif here 
    epochs = mne.read_epochs(path)
    
    
    #%% RANSAC
    channel_types_dict = {ch_name: 'eeg' for ch_name in epochs.ch_names}
    epochs.set_channel_types(channel_types_dict)
    
    rsc = Ransac(n_jobs=-1)
    
    epochs_RANSAC = rsc.fit_transform(epochs) #https://www.frontiersin.org/articles/10.3389/fninf.2015.00016/full
    
    if  exp == 'Experiment 09' or exp == 'Experiment 19':
        epochs_RANSAC = epochs_RANSAC['Motor']
        
    #%% Decimate
    
    current_sfreq = epochs.info["sfreq"]
    decim = np.round(current_sfreq / desired_sfreq).astype(int)
    obtained_sfreq = current_sfreq / decim
    
    print(
    f'desired sampling frequency was {desired_sfreq} Hz; decim factor of {decim} yielded an '
    f'actual sampling frequency of {epochs.info["sfreq"]/decim} Hz.'
    )
    
    epochs_RANSAC.decimate(decim=decim, verbose=True)
    
    epochs_RANSAC.save(f'D:\MEA DCBUN vs UN\Data and Notes\Data\Bad Channel Decimate\{exp}-epo.fif', fmt='double')
    

import matplotlib.pyplot as plt


evoked_raw = epochs.average(by_event_type=True)
evoked_RANSAC= epochs_RANSAC.average(by_event_type=True)

# %matplotlib inline
exp = os.path.basename(path)[:-8]

# Plot evoked responses for each event side by side with custom titles
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'{exp} - MEA ({evoked_raw[0].info["nchan"]} channels) \nBads: {", ".join(rsc.bad_chs_)}', x=0.53, fontsize=16)

# Plotting evoked responses for raw data
for i, event in enumerate(evoked_raw):
    event.plot(axes=axes[i, 0], show=False)
    axes[i, 0].set_title(f'Filtered - {event.comment}')

# Plotting evoked responses for RANSAC data
for i, event in enumerate(evoked_RANSAC):
    event.plot(axes=axes[i, 1], show=False)
    axes[i, 1].set_title(f'Filtered + RANSAC - {event.comment}')

fig.tight_layout()
plt.show()