# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:34:45 2024

@author: GZ57NM
"""

import numpy as np
import mne
import tdt

path = 'D:\RANSAC comparison\Epidural\Data\Raw\Suzan_Chronic_PZA10-230228\ChronicPig-230228-105608'

# Load TDT data
data = tdt.read_block(path)

# Extract streams and epocs
streams = data.streams
epocs = data.epocs
sfreq = streams['Wav1'].fs  

#%% Frame
# Create MNE Raw object with correct channel names
ch_names = [str(chan) for chan in streams['Wav1'].channel]  # Convert channel numbers to strings

montage_positions = [(7,0,0), (7,1,0), (7,2,0), (7,3,0), (6,0,0), (6,1,0), (6,2,0), (6,3,0), (5,0,0), (5,1,0), (5,2,0), (5,3,0), (4,0,0), (4,1,0), (4,2,0), (4,3,0), 
                     (3,3,0), (3,2,0), (3,1,0), (3,0,0), (2,3,0), (2,2,0), (2,1,0), (2,0,0), (1,3,0), (1,2,0), (1,1,0), (1,0,0), (0,3,0), (0,2,0), (0,1,0), (0,0,0)]

# 0,0,0	 1,0,0	2,0,0	3,0,0	4,0,0	5,0,0	6,0,0	7,0,0
# 0,1,0	 1,1,0	2,1,0	3,1,0	4,1,0	5,1,0	6,1,0	7,1,0
# 0,2,0	 1,2,0	2,2,0	3,2,0	4,2,0	5,2,0	6,2,0	7,2,0
# 0,3,0	 1,3,0	2,3,0	3,3,0	4,3,0	5,3,0	6,3,0	7,3,0

# 29	25	21	17	16	12	8	4
# 30	26	22	18	15	11	7	3
# 31	27	23	19	14	10	6	2
# 32	28	24	20	13	9	5	1


# Create MNE info object
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, montage_positions)), coord_frame='unknown')
info.set_montage(montage)

montage.plot()

