# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:01:27 2023

@author: GZ57NM
"""
import mne
import matplotlib.pyplot as plt
import plyer as pl
from autoreject import AutoReject, get_rejection_threshold, Ransac
import os


#%% Create MNE epochs object
paths=pl.filechooser.open_file(filters=[("MNE fif (*.fif)", "*.fif")], multiple = True)

for path in paths:
    # load epoch fif here 
    epochs = mne.read_epochs(path)
    
    #%% AutoReject
    
    ar = AutoReject(n_jobs=-1)

    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)  
    reject = get_rejection_threshold(epochs)  

    
    #%% Plot evoked response side by side
    evoked_raw = epochs.average(by_event_type=True)
    evoked_clean= epochs_clean.average(by_event_type=True)
    
    # %matplotlib inline
    exp = os.path.basename(path)[:-8]
    
    # Plot evoked responses for each event side by side with custom titles
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{exp} - MEA ({evoked_raw[0].info["nchan"]} channels) \nBads: {", ".join(reject_log.ch_names)}', x=0.53, fontsize=16)
    
    # Plotting evoked responses for raw data
    for i, event in enumerate(evoked_raw):
        event.plot(axes=axes[i, 0], show=False)
        axes[i, 0].set_title(f'Filtered - {event.comment}')
    
    # Plotting evoked responses for RANSAC data
    for i, event in enumerate(evoked_clean):
        event.plot(axes=axes[i, 1], show=False)
        axes[i, 1].set_title(f'Filtered + AutoReject - {event.comment}')
    
    fig.tight_layout()
    plt.savefig(f'D:\MEA DCBUN vs UN\Data and Notes\Data\Bad Channel Plots\AutoReject\{exp}_AutoReject.svg')
    plt.close()
    

    
    # Create the plot without showing it
    reject_log.plot(show=False)
    # Customize the aspect ratio of the entire figure
    fig = plt.gcf()
    fig.suptitle(f'{exp} - MEA ({evoked_raw[0].info["nchan"]} channels) \nBads: {", ".join(reject_log.ch_names)}', x=0.53, fontsize=16)
    fig.set_size_inches(10, 8)  # Adjust dimensions as needed
    fig.gca().set_aspect('auto')  # Adjust the aspect ratio as needed
    fig.tight_layout()
    # Show the modified plot
    plt.savefig(f'D:\MEA DCBUN vs UN\Data and Notes\Data\Bad Channel Plots\AutoReject\{exp}_AutoReject_heat.svg')


