# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:25:33 2024

@author: GZ57NM
"""
import mne

epochs=mne.read_epochs(r'D:\RANSAC comparison\Intracortical\Data\Preprocessed\Experiment 09-epo.fif', proj=True, preload=True, verbose=None)

xa = 1
ya = 1
xb = 0
yb = 0

montage_positions = [(0,3,0), (0,2,0), (0,1,0), (0,0,0), (1,3,0), (1,2,0), (1,1,0), (1,0,0), 
                     (2,3,0), (2,2,0), (2,1,0), (2,0,0), (3,3,0), (3,2,0), (3,1,0), (3,0,0),]

# Modify each tuple in the list
montage_positions = [(x * xa+xb, y * ya+yb, z) for x, y, z in montage_positions]

montage = mne.channels.make_dig_montage(ch_pos=dict(zip(epochs.ch_names, montage_positions)), coord_frame='unknown')

montage.plot()

