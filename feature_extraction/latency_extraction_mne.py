# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:15:15 2023

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
    latency_dict = {
        'Cutaneous': {'n1': 0, 'p1': 0, 'n1_amp': 0, 'p1_amp': 0},
        'Motor': {'n1': 0, 'p1': 0, 'n1_amp': 0, 'p1_amp': 0}
        }
    
    # Load MNE data for the current subject
    epochs = load_mne_data(subject_id)
    
    for stim in latency_dict:
        for ch in epochs.info['ch_names']:
            evoked = epochs[stim].average().pick(ch)
            
            # Store latency information for each subject, condition, and channel
            _, p1_lat, p1_amp = evoked.get_peak(mode='pos', tmin=0.01, tmax=0.1, return_amplitude=True)
            _, n1_lat, n1_amp = evoked.get_peak(mode='neg', tmin=0.01, tmax=0.1, return_amplitude=True)
            latency_dict[stim]['p1'] += p1_lat
            latency_dict[stim]['n1'] += n1_lat
            latency_dict[stim]['p1_amp'] += p1_amp 
            latency_dict[stim]['n1_amp'] += n1_amp

        # Store subject-specific dictionary in the main latency_dict
        latency_dict[stim]['p1'] /= epochs.info['nchan']
        latency_dict[stim]['n1'] /= epochs.info['nchan']
        latency_dict[stim]['p1_amp'] /= epochs.info['nchan']
        latency_dict[stim]['n1_amp'] /= epochs.info['nchan']
        latency_dict[stim]['p1_amp'] *= 1e+6 #V to microV
        latency_dict[stim]['n1_amp'] *= 1e+6 #V to microV
        subject_dict[subject_id] = latency_dict

#%% Global average
import numpy as np

# Initialize global averages
global_avg_latencies = {'Cutaneous': {'n1': 0, 'p1': 0}, 'Motor': {'n1': 0, 'p1': 0}}
global_avg_amplitudes = {'Cutaneous': {'n1_amp': 0, 'p1_amp': 0}, 'Motor': {'n1_amp': 0, 'p1_amp': 0}}
num_subjects = len(subject_ids)

# Loop through subjects
for subject_id in subject_ids:
    # Load MNE data for the current subject
    epochs = load_mne_data(subject_id)
    
    for stim in latency_dict:
        # Update global averages
        global_avg_latencies[stim]['p1'] += subject_dict[subject_id][stim]['p1']
        global_avg_latencies[stim]['n1'] += subject_dict[subject_id][stim]['n1']
        global_avg_amplitudes[stim]['p1_amp'] += subject_dict[subject_id][stim]['p1_amp']
        global_avg_amplitudes[stim]['n1_amp'] += subject_dict[subject_id][stim]['n1_amp']

# Compute global averages
for stim in latency_dict:
    global_avg_latencies[stim]['p1'] /= num_subjects
    global_avg_latencies[stim]['n1'] /= num_subjects
    global_avg_amplitudes[stim]['p1_amp'] /= num_subjects
    global_avg_amplitudes[stim]['n1_amp'] /= num_subjects

# Print or use global averages as needed
print("Global Average Latencies:")
print(global_avg_latencies)
print("\nGlobal Average Amplitudes:")
print(global_avg_amplitudes)


#%% Seaborn plot

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Convert the data into a format suitable for seaborn boxplot
latency_data = {'Subject': [], 'Condition': [], 'N1 Latency': [], 'P1 Latency': []}
amplitude_data = {'Subject': [], 'Condition': [], 'N1 Amplitude': [], 'P1 Amplitude': []}

# Populate the data dictionaries
for subject_id, latency_dict in subject_dict.items():
    for stim, values in latency_dict.items():
        latency_data['Subject'].append(subject_id)
        latency_data['Condition'].append(stim)
        latency_data['N1 Latency'].append(values['n1'])
        latency_data['P1 Latency'].append(values['p1'])
        
        amplitude_data['Subject'].append(subject_id)
        amplitude_data['Condition'].append(stim)
        amplitude_data['N1 Amplitude'].append(values['n1_amp'])
        amplitude_data['P1 Amplitude'].append(values['p1_amp'])

# Create Pandas DataFrames
latency_df = pd.DataFrame(latency_data)
amplitude_df = pd.DataFrame(amplitude_data)

min_lat, max_lat, min_amp, max_amp = None, None, None, None

# Set up the matplotlib figure
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Boxplot for Latency
sns.boxplot(x='Condition', y='N1 Latency', data=latency_df, ax=axes[0,0])
axes[0,0].set_title('N1 Latency')
axes[0,0].set_ylabel('Latency (s)')
axes[0,0].set_ylim(bottom=min_lat, top=max_lat)  # Replace min_lat and max_lat with your desired limits

# Boxplot for Amplitude
sns.boxplot(x='Condition', y='N1 Amplitude', data=amplitude_df, ax=axes[1,0])
axes[1,0].set_title('N1 Amplitude')
axes[1,0].set_ylabel('Amplitude (μV)')
axes[1,0].set_ylim(bottom=min_amp, top=max_amp)  # Replace min_amp and max_amp with your desired limits

# Boxplot for Latency
sns.boxplot(x='Condition', y='P1 Latency', data=latency_df, ax=axes[0,1])
axes[0,1].set_title('P1 Latency')
axes[0,1].set_ylabel('Latency (s)')
axes[0,1].set_ylim(bottom=min_lat, top=max_lat)  # Replace min_lat and max_lat with your desired limits

# Boxplot for Amplitude
sns.boxplot(x='Condition', y='P1 Amplitude', data=amplitude_df, ax=axes[1,1])
axes[1,1].set_title('P1 Amplitude')
axes[1,1].set_ylabel('Amplitude (μV)')
axes[1,1].set_ylim(bottom=min_amp, top=max_amp)  # Replace min_amp and max_amp with your desired limits

# Adjust layout
plt.tight_layout()
plt.show()

#%% Lineplot
# Set up the matplotlib figure
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))


sns.lineplot(x='Condition', y='N1 Latency', hue='Subject',data=latency_df, ax=axes[0,0])
axes[0,0].set_title('N1 Latency')
axes[0,0].set_ylabel('Latency (s)')
axes[0,0].set_ylim(bottom=min_lat, top=max_lat)  # Replace min_lat and max_lat with your desired limits


sns.lineplot(x='Condition', y='N1 Amplitude', hue='Subject',data=amplitude_df, ax=axes[1,0])
axes[1,0].set_title('N1 Amplitude')
axes[1,0].set_ylabel('Amplitude (μV)')
axes[1,0].set_ylim(bottom=min_amp, top=max_amp)  # Replace min_amp and max_amp with your desired limits


sns.lineplot(x='Condition', y='P1 Latency', hue='Subject',data=latency_df, ax=axes[0,1])
axes[0,1].set_title('P1 Latency')
axes[0,1].set_ylabel('Latency (s)')
axes[0,1].set_ylim(bottom=min_lat, top=max_lat)  # Replace min_lat and max_lat with your desired limits


sns.lineplot(x='Condition', y='P1 Amplitude', hue='Subject',data=amplitude_df, ax=axes[1,1])
axes[1,1].set_title('P1 Amplitude')
axes[1,1].set_ylabel('Amplitude (μV)')
axes[1,1].set_ylim(bottom=min_amp, top=max_amp)  # Replace min_amp and max_amp with your desired limits

# Adjust layout
plt.tight_layout()
plt.show()

