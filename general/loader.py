# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:35:41 2023

@author: GZ57NM
"""



def load_mat():
    """
    Parameters
    ----------
    None

    Returns
    -------
    data : dict
        Contains data from the MATLAB file
    folder_name : str
        Name of the folder containing the selected MATLAB file
    """
    
    import mat73
    import ctypes
    import tkinter as tk
    from tkinter import filedialog
    import os
    
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(filetypes=[("Binary MATLAB file (*.mat)", "*.mat")])

    if path:
        data_dict = mat73.loadmat(path)
        folder_name = os.path.basename(os.path.dirname(path))
        return data_dict, folder_name
    else:
        ctypes.windll.user32.MessageBoxW(0, "Please select the correct file path", "Path not found")
        

def tdt_to_mne(path, stream_id=str, onset_id=str, ch_types='eeg', electrode_type = None):
    """
    Convert TDT data to MNE Raw object.

    Parameters
    ----------
    stream_id : str, optional
        The ID of the TDT stream. The default is str.
    onset_id : str, optional
        The ID of the TDT onset. The default is str.
    ch_types : str, optional
        The type of channels, e.g., 'eeg'. The default is 'eeg'.

    Returns
    -------
    raw : mne.io.RawArray
        The MNE RawArray object containing the raw data.
    events_stack : numpy.ndarray
        A 2D array representing the events in MNE format [onset, preceeding signal value, event_type].

    """

    import tdt
    import mne
    import numpy as np

    # Load TDT data
    data = tdt.read_block(path)

    # Extract streams and epocs
    streams = data.streams
    epocs = data.epocs

    # Extract raw data from the specified stream
    raw_data = streams[stream_id].data 

    # Sampling frequency of the specified stream
    sfreq = streams[stream_id].fs  

    # Extract onset events from the specified epoc
    events = epocs[onset_id].onset * sfreq

    # Stack events for MNE format (column format: [onset, duration, event_type])
    events_stack = np.int64(np.rint(np.column_stack((events, np.zeros_like(events), np.ones_like(events)))))

    # Create MNE Raw object with correct channel names
    ch_names = [str(chan) for chan in streams[stream_id].channel]  # Convert channel numbers to strings
    
    if electrode_type == 'ecog':
        # Create MNE Raw object with correct channel names
        ch_names = [str(chan) for chan in streams['Wav1'].channel]  # Convert channel numbers to strings

        montage_positions = [(7,0,0), (7,1,0), (7,2,0), (7,3,0), (6,0,0), (6,1,0), (6,2,0), (6,3,0), (5,0,0), (5,1,0), (5,2,0), (5,3,0), (4,0,0), (4,1,0), (4,2,0), (4,3,0), 
                             (3,3,0), (3,2,0), (3,1,0), (3,0,0), (2,3,0), (2,2,0), (2,1,0), (2,0,0), (1,3,0), (1,2,0), (1,1,0), (1,0,0), (0,3,0), (0,2,0), (0,1,0), (0,0,0)]

        # Coords for MNE montage
        # 0,3,0	1,3,0	2,3,0	3,3,0	4,3,0	5,3,0	6,3,0	7,3,0
        # 0,2,0	1,2,0	2,2,0	3,2,0	4,2,0	5,2,0	6,2,0	7,2,0
        # 0,1,0	1,1,0	2,1,0	3,1,0	4,1,0	5,1,0	6,1,0	7,1,0
        # 0,0,0	1,0,0	2,0,0	3,0,0	4,0,0	5,0,0	6,0,0	7,0,0

        # Ch position
        # 29	25	21	17	16	12	8	4
        # 30	26	22	18	15	11	7	3
        # 31	27	23	19	14	10	6	2
        # 32	28	24	20	13	9	5	1
    
    if electrode_type == 'mea':
        montage_ysize = 4 if len(ch_names) == 16 else 8
        montage_positions = [(x, y, 0) for x in range(4) for y in range(montage_ysize)]

    # Create MNE info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, montage_positions)))
    
    info.set_montage(montage)
    
    # Create MNE Raw object
    raw = mne.io.RawArray(raw_data, info)

    return raw, events_stack
