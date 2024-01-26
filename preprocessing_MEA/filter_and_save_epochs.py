# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:01:27 2023

@author: GZ57NM
"""

import os
import numpy as np
import mne
import mat73
from tqdm import tqdm
import gc # Garbage collection module

from utils import create_raw_object, montage_remap

# CONFIGURATION
DESIRED_SFREQ = 2000
mne.set_log_level('WARNING')
main_folder = r'D:\MEA DCBUN vs UN\Data and Notes\Data\sorted MEA data'
output_path = r'D:\RANSAC comparison\Intracortical\Data\Preprocessed'


# Main loop to iterate over experiment folders
experiment_folders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

for experiment_folder in tqdm(experiment_folders, position=0, desc = 'Experiment', leave=True):
    # Load data for each experiment
    data = mat73.loadmat(experiment_folder+'\set1_cut&mot.mat')
    folder = os.path.basename(os.path.dirname(experiment_folder+'\set1_cut&mot.mat'))

    # Extract necessary values from the dictionary
    data_continous = data['dataContinous']
    fs = float(data['fs'])
    stim_times = data['stimTimes']
    del data
    gc.collect()  # Manual garbage collection
    
    ch_names = [f'{i}' for i in range(data_continous.shape[0])]


    if folder == 'Experiment 09':
        baseAmp = np.concatenate([np.arange(20, 301, 20),
                                  np.arange(350, 1001, 50), 
                                  np.arange(1100, 1501, 100), 
                                  np.arange(1750, 5001, 250), 
                                  np.arange(5500, 7001, 500)]) 
        
        sweep1, sweep3 = [np.concatenate([baseAmp, np.arange(8000, limit, 1000)]) for limit in [16001, 10001]]
        sweeps = [sweep1, sweep3]
        stimValues = np.concatenate(sweeps)
        
    else:
        baseAmp = np.concatenate([np.arange(0, 301, 50), 
                                  np.arange(350, 1001, 50), 
                                  np.arange(1100, 1501, 100), 
                                  np.arange(1750, 5001, 250), 
                                  np.arange(5500, 7001, 500)])
        
        sweep1, sweep2, sweep3, sweep4 = [np.concatenate([baseAmp, np.arange(8000, limit, 1000)]) for limit in [16001, 12001, 10001, 15001]]
        sweeps = [sweep1, sweep2, sweep3, sweep4]
        stimValues = np.concatenate(sweeps)

    # Define the delays in seconds
    delays = np.array([0.07778, 4.422, 9.922, 53.91])/1000    

    delayVector = np.concatenate([np.repeat(delay, len(sweep)) for delay, sweep in zip(delays, sweeps)])
    delayVectorrep = np.tile(delayVector, 4)
    stimValuesrep = np.tile(stimValues, 4)
    mask = stimValuesrep >= 1000

    maskdel_cut = stim_times[0][mask] * fs - delayVectorrep[mask]
    maskdel_mot = stim_times[1][mask] * fs - delayVectorrep[mask]

    events_cutaneous = np.column_stack((maskdel_cut, np.zeros_like(maskdel_cut), np.ones_like(maskdel_cut)))
    events_motor = np.column_stack((maskdel_mot, np.zeros_like(maskdel_mot), 2 * np.ones_like(maskdel_mot)))

    events = np.concatenate((events_cutaneous, events_motor), axis=0)
    events = events[events[:, 0].argsort()]
    event_id = {'Cutaneous': 1, 'Motor': 2}
    
    # Setup montage positions using remapper
    montage_positions = montage_remap(folder)

    raw = create_raw_object(data_continous, ch_names, fs, montage_positions)
    
    raw.notch_filter([50, 100, 150, 200], filter_length='auto', notch_widths=2, method='fir', n_jobs=-1)
    raw.filter(1, 200, l_trans_bandwidth=1, h_trans_bandwidth=10, method='fir', n_jobs=-1)

    epochs = mne.Epochs(raw, np.int64(np.rint(events)), event_id, tmin=-0.2, tmax=0.5, detrend=1, baseline=(-0.2, 0), preload=False) # Preload to false to reduce risk of memory errors
    
    # Decimation
    decim = round(epochs.info['sfreq'] / DESIRED_SFREQ)
    epochs.decimate(decim=decim, verbose=True)
    print(f'Decimated to {raw.info["sfreq"]/decim} Hz with a factor of {decim}.')
    
    epochs.save(os.path.join(output_path, f'{folder}-epo.fif'), fmt='double', overwrite = True)
     
    del raw, data_continous
    gc.collect()  # Manual garbage collection