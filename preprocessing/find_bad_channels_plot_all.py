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
path=pl.filechooser.open_file(filters=[("MNE fif (*.fif)", "*.fif")])

# load epoch fif here 
epochs = mne.read_epochs(path[0])

epochs.plot(events=epochs.events, use_opengl = True)

#%% Autoreject - Have ot find out whether this only detects events
# ar = AutoReject(n_jobs=-1)

# epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)  
# reject = get_rejection_threshold(epochs)  

# epochs_clean.plot(events=epochs_clean.events, use_opengl = True)

# #https://autoreject.github.io/stable/index.html

#%% RANSAC
channel_types_dict = {ch_name: 'eeg' for ch_name in epochs.ch_names}
epochs.set_channel_types(channel_types_dict)

rsc = Ransac(verbose=True, n_jobs=-1)

epochs_RANSAC = rsc.fit_transform(epochs) #https://www.frontiersin.org/articles/10.3389/fninf.2015.00016/full

print('\n'.join(rsc.bad_chs_))

epochs_RANSAC.plot(events=epochs_RANSAC.events, use_opengl = True)


#%% Plot evoked response 
evoked_raw = epochs.average(by_event_type=True)
evoked_RANSAC= epochs_RANSAC.average(by_event_type=True)

# %matplotlib inline
folder = os.path.basename(path[0])

# Plot evoked responses for each event with custom titles

for evoked in [evoked_raw, evoked_RANSAC]:
    fig, axes = plt.subplots(len(evoked), 1, figsize=(10, 6))
    fig.suptitle(f'{folder[:-8]} - MEA ({evoked[0].info["nchan"]} channels)', x=0.53, fontsize=14)
    
    for event, ax in zip(evoked, axes):    
        # Plotting evoked response
        fig = event.plot(axes=ax, show=False)
        ax.set_title(f'{event.comment}')
    
    fig.tight_layout()
    plt.show()


#%% Heatmap for bad channels
ch_names = [epochs.ch_names[ii] for ii in rsc.picks]
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(rsc.bad_log, cmap='Reds', interpolation='nearest')

ax.grid(False)
ax.set_xlabel('Sensors')
ax.set_ylabel('Trials')

# Adjust xticks based on the number of sensors
plt.setp(ax, xticks=range(0, len(rsc.picks), 1), xticklabels=ch_names)

plt.setp(ax.get_yticklabels(), rotation=0)
plt.setp(ax.get_xticklabels(), rotation=90)
ax.tick_params(axis=u'both', which=u'both', length=0)

fig.colorbar(im, ax=ax)  # Add colorbar

fig.tight_layout(rect=[None, None, None, 1.1])
plt.show()