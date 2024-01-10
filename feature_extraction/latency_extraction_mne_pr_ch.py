# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:04:01 2023

@author: GZ57NM
"""
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

subject_ids = ['Experiment 13', 'Experiment 15', 'Experiment 16']

# Function to load the MNE raw data
def load_mne_data(subject_id):
    epochs = mne.read_epochs(f'D:\\MEA DCBUN vs UN\\Data and Notes\\Data\\Bad Channel Decimate\\{subject_id}-epo.fif', verbose=False)
    return epochs

subject_dict = {}

# Loop through subjects
for subject_id in subject_ids:
    latency_dict = {
        'Cutaneous': {'n1': [], 'p1': [], 'n1_amp': [], 'p1_amp': []},
        'Motor': {'n1': [], 'p1': [], 'n1_amp': [], 'p1_amp': []}
    }
    
    # Load MNE data for the current subject
    epochs = load_mne_data(subject_id)
    
    for stim in latency_dict:
        for ch in epochs.info['ch_names']:
            evoked = epochs[stim].average().pick(ch)
            
            # Store latency information for each subject, condition, and channel
            _, p1_lat, p1_amp = evoked.get_peak(mode='pos', tmin=0, tmax=0.2, return_amplitude=True)
            _, n1_lat, n1_amp = evoked.get_peak(mode='neg', tmin=0, tmax=0.2, return_amplitude=True)
            latency_dict[stim]['p1'].append(p1_lat)
            latency_dict[stim]['n1'].append(n1_lat)
            latency_dict[stim]['p1_amp'].append(p1_amp) 
            latency_dict[stim]['n1_amp'].append(n1_amp)

    subject_dict[subject_id] = latency_dict

# Convert the data into a format suitable for seaborn boxplot
latency_data = {'Subject': [], 'Condition': [], 'N1 Latency': [], 'P1 Latency': []}
amplitude_data = {'Subject': [], 'Condition': [], 'N1 Amplitude': [], 'P1 Amplitude': []}

# Populate the data dictionaries
for subject_id, latency_dict in subject_dict.items():
    for stim, values in latency_dict.items():
        latency_data['Subject'].extend([subject_id] * len(values['n1']))
        latency_data['Condition'].extend([stim] * len(values['n1']))
        latency_data['N1 Latency'].extend(values['n1'])
        latency_data['P1 Latency'].extend(values['p1'])
        
        amplitude_data['Subject'].extend([subject_id] * len(values['n1_amp']))
        amplitude_data['Condition'].extend([stim] * len(values['n1_amp']))
        amplitude_data['N1 Amplitude'].extend(values['n1_amp'])
        amplitude_data['P1 Amplitude'].extend(values['p1_amp'])

# Create Pandas DataFrames
latency_df = pd.DataFrame(latency_data)
amplitude_df = pd.DataFrame(amplitude_data)

# Set up the matplotlib figure
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Boxplot for Latency
sns.boxplot(x='Condition', y='N1 Latency', data=latency_df, ax=axes[0, 0])
axes[0, 0].set_title('Latency')
axes[0, 0].set_ylabel('N1 Latency (s)')

# Boxplot for Amplitude
sns.boxplot(x='Condition', y='N1 Amplitude', data=amplitude_df, ax=axes[0, 1])
axes[0, 1].set_title('Amplitude')
axes[0, 1].set_ylabel('N1 Amplitude (uV)')

# Boxplot for Latency
sns.boxplot(x='Condition', y='P1 Latency', data=latency_df, ax=axes[1, 0])
axes[1, 0].set_title('P1 Latency')
axes[1, 0].set_ylabel('Latency (s)')

# Boxplot for Amplitude
sns.boxplot(x='Condition', y='P1 Amplitude', data=amplitude_df, ax=axes[1, 1])
axes[1, 1].set_title('P1 Amplitude')
axes[1, 1].set_ylabel('Amplitude (uV)')

# Adjust layout
plt.tight_layout()
plt.show()
