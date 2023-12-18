# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:53:34 2023

@author: GZ57NM
"""
import mne
import numpy as np
from scipy.signal import find_peaks

subject_ids = ['Experiment 13', 'Experiment 15', 'Experiment 16']

# Function to load the MNE raw data
def load_mne_data(subject_id):
    epochs = mne.read_epochs(f'D:\\MEA DCBUN vs UN\\Data and Notes\\Data\\Bad Channel Decimate\\{subject_id}-epo.fif', verbose=False)
    return epochs


n1_cut, p1_cut, n1_mot, p1_mot = 0, 0, 0, 0

latency_dict = {
    'Cutaneous': {'n1': 0, 'p1': 0},
    'Motor': {'n1': 0, 'p1': 0}
}

# Loop through subjects
for subject_id in subject_ids:
    # Load MNE data for the current subject
    epochs = load_mne_data(subject_id)
    

    for stim_type in latency_dict:
        for epoch in epochs[stim_type]:
            for ch in epoch:
                latency_dict[stim_type]['n1'] += np.where(ch==np.argmin(ch))
                latency_dict[stim_type]['p1'] += np.where(ch==np.argmax(ch))
    
    
    # Calculate averages
    latency_dict[stim_type]['n1'] /= len(epochs[stim_type])
    latency_dict[stim_type]['p1'] /= len(epochs[stim_type])
    

    print(latency_dict)

