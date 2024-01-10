# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:44:39 2023

@author: GZ57NM
"""

import mne
mne.set_log_level(verbose=False)

subject_ids = ['Experiment 13', 'Experiment 15', 'Experiment 16']

# Function to load the MNE raw data
def load_mne_data(subject_id):
    epochs = mne.read_epochs(f'D:\\MEA DCBUN vs UN\\Data and Notes\\Data\\Bad Channel Decimate\\{subject_id}-epo.fif', verbose=False)
    return epochs

subject_dict = {}

# Loop through subjects
for subject_id in subject_ids:
    amplitude_dict = {
        'Cutaneous': {},
        'Motor': {}
        }
    
    # Load MNE data for the current subject
    epochs = load_mne_data(subject_id)
    
    for stim in amplitude_dict:
        evoked = epochs[stim].average()
    
        amplitude_dict[stim]=evoked.crop(tmin=0.01, tmax=0.1).data.mean()*1e6
        
    subject_dict[subject_id] = amplitude_dict
    
#%% Plotting

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Convert the data into a format suitable for seaborn lineplot
amplitude_data = {'Subject': [], 'Condition': [], 'Amplitude': []}

# Populate the data dictionary
for subject_id, amplitude_dict in subject_dict.items():
    for stim, value in amplitude_dict.items():
        amplitude_data['Subject'].append(subject_id)
        amplitude_data['Condition'].append(stim)
        amplitude_data['Amplitude'].append(value)

# Create Pandas DataFrame
amplitude_df = pd.DataFrame(amplitude_data)

# Set up the matplotlib figure
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

# Lineplot for Amplitude
sns.lineplot(x='Condition', y='Amplitude', hue='Subject', data=amplitude_df, marker='o', linestyle='-', markersize=8, ax=axes)
axes.set_title('Amplitude for Cutaneous and Motor Conditions')
axes.set_ylabel('Amplitude (μV)')

# Adjust layout
plt.tight_layout()
plt.show()