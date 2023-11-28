# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:07:57 2023

@author: GZ57NM
"""

import numpy as np
import mne

from loader import load_mat


#%% Load and structure data

data=load_mat()

# Resample freq
# frs=2000

# Extract necessary values from the dictionary
data_continous = data['dataContinous'] # EEG data
fs = float(data['fs']) # Sampling frequency"
stim_times = data['stimTimes'] # Events cutan and motor stimulation times
ch_names=['chan{}'.format(i) for i in range(data_continous.shape[0])] # Channel list for MNE

# Create info structure
info = mne.create_info(ch_names=ch_names,
                       sfreq=fs, ch_types='ecog')

# Make simple montage for MEA
montage_ysize = 4 if len(ch_names) == 16 else 8 # Determine montage size based on the length of ch_names
montage_positions = [(x, y, 0) for x in range(4) for y in range(montage_ysize)]
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, montage_positions), ))
info.set_montage(montage) # Apply the montage to the info structure

# Create events for the two types of stimulations
events_cutaneous = np.column_stack((stim_times[0]*fs, np.zeros_like(stim_times[0]), np.ones_like(stim_times[0])))
events_motor = np.column_stack((stim_times[1]*fs, np.zeros_like(stim_times[1]), 2 * np.ones_like(stim_times[1])))

events = np.concatenate((events_cutaneous, events_motor), axis=0)
events = events[events[:, 0].argsort()] # Sort events
event_id = {'Cutaneous': 1, 'Motor': 2}

if 
stimValues = np.concatenate([
    np.arange(0, 301, 50), np.arange(350, 1001, 50), np.arange(1100, 1501, 100), np.arange(1750, 5001, 250), np.arange(5500, 7001, 500), np.arange(8000, 16001, 1000),
    np.arange(0, 301, 50), np.arange(350, 1001, 50), np.arange(1100, 1501, 100), np.arange(1750, 5001, 250), np.arange(5500, 7001, 500), np.arange(8000, 12001, 1000),
    np.arange(0, 301, 50), np.arange(350, 1001, 50), np.arange(1100, 1501, 100), np.arange(1750, 5001, 250), np.arange(5500, 7001, 500), np.arange(8000, 10001, 1000),
    np.arange(0, 301, 50), np.arange(350, 1001, 50), np.arange(1100, 1501, 100), np.arange(1750, 5001, 250), np.arange(5500, 7001, 500), np.arange(8000, 15001, 1000)
])

stimValuesrep = np.tile(stimValues,4)

#%% Create MNE Raw object
raw = mne.io.RawArray(data_continous, info)

# Trigger corrections
# For pulse durations of 100 us > 0.07778 ms
# For pulse durations of 500 us > 4.422 ms
# For pulse durations of 1000 us > 9.922 ms
# For pulse durations of 5000 us > 53.91 ms


# Resample to 4000 Hz sampling frequency --> 10 x 200 Hz limit. ONLY FOR VISUALIZATION SPEED UP. RESAMPLE AFTER EPOCHING TO AVOID JITTERS
raw = raw.resample(sfreq=2000, npad='auto', n_jobs=-1)

# Compute PSD and Filter
before=raw.compute_psd(method='welch', fmax=500, n_fft=20000, n_jobs=-1)
before.plot()
before.plot(average=True)

# Apply filters
raw.notch_filter([50, 100, 150, 200],filter_length='auto', notch_widths=2, method='fir', n_jobs=-1)
raw.filter(1, 200, l_trans_bandwidth=1, h_trans_bandwidth=10, method='fir', n_jobs = -1)

after=raw.compute_psd(method='welch', fmax=500, n_fft=20000, n_jobs=-1)
after.plot()
after.plot(average=True)

#%% Create MNE epochs object

epochs = mne.Epochs(raw, np.int64(np.rint(events)), event_id, baseline = (None,0), tmin = -0.2, tmax = 0.5, detrend = None, preload=True)


avg=epochs.average(by_event_type=True)

_ = [i.plot(gfp=True) for i in avg]

epochs.plot(events=events)


