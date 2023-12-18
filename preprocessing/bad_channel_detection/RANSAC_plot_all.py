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
    
    #%% RANSAC
    channel_types_dict = {ch_name: 'eeg' for ch_name in epochs.ch_names}
    epochs.set_channel_types(channel_types_dict)
    
    rsc = Ransac(verbose=True, n_jobs=-1)
    
    epochs_RANSAC = rsc.fit_transform(epochs) #https://www.frontiersin.org/articles/10.3389/fninf.2015.00016/full
    
    #%% Plot evoked response side by side
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
    plt.savefig(f'D:\MEA DCBUN vs UN\Data and Notes\Data\Bad Channel Plots\RANSAC\{exp}_RANSAC.svg')
    plt.close()

    #%% Plot heatmap
    
    ch_names = [epochs.ch_names[ii] for ii in rsc.picks]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    fig.suptitle(f'{exp} - MEA ({evoked_raw[0].info["nchan"]} channels) \nBads: {", ".join(rsc.bad_chs_)}', x=0.53, fontsize=16)
    
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
    plt.savefig(f'D:\MEA DCBUN vs UN\Data and Notes\Data\Bad Channel Plots\RANSAC\{exp}_RANSAC_heat.svg')
    plt.close()

