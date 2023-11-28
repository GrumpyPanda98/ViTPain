# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:24:16 2023

@author: GZ57NM
"""

from loadTDT import loadTDT
from mne.io import RawArray
import numpy as np
from mne import create_info
# Replace 'your_tdt_file.tdt' with the actual path to your TDT file
path = 'D:\MEA DCBUN vs UN\Data and Notes\Data\Experiment 6\Subject1-200211-113253'

# Load the TDT file
tdt_data = loadTDT(path)

# # Extract the necessary data from tdt_data and create MNE Raw object
# data = tdt_data.streams['RSn1'].data  
# fs = tdt_data.streams['RSn1'].fs  

# # Create MNE Info object
# ch_names = [f'CH{i}' for i in range(len(data))]
# info = create_info(ch_names, sfreq=fs, ch_types='eeg')


# # Create MNE RawArray object
# raw = RawArray(data=np.array(data).T, info=info, verbose=False)
