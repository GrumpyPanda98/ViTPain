# -*- coding: utf-8 -*-
"""
Created on Mon Nov 6 14:35:41 2023

@author: GZ57NM
"""
import tdt
import mne
import numpy as np


# Path to your TDT data
path = 'D:\RANSAC comparison\Suzan_Chronic_PZA10-230228\ChronicPig-230228-105608'

# Load TDT data
data = tdt.read_block(path)

# Extract streams and epocs
streams = data.streams
epocs = data.epocs

# Create MNE Raw object
raw_data = streams['Wav1'].data  # Replace 'YourStreamName' with the actual stream name
sfreq = streams['Wav1'].fs  # Sampling frequency of the stream

events = epocs['PC0_'].onset * sfreq

events_mne = np.column_stack((events, np.zeros_like(events), np.ones_like(events)))

# Creating MNE Raw object with correct channel names
ch_names = [str(chan) for chan in streams['Wav1'].channel]  # Convert channel numbers to strings

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='ecog')

raw = mne.io.RawArray(raw_data, info) 

epochs = mne.Epochs(raw, np.int64(np.rint(events_mne)), tmin = -0.2, tmax = 0.5, detrend = None, preload=True)

evoked=epochs.average()
evoked.plot()
