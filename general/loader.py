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
        

def tdt_to_mne(path, stream_id=None, onset_id=None, ch_types='eeg', electrode_type = None, scale=False):
    """
    Convert TDT data to MNE Raw object.

    Parameters
    ----------
    stream_id : str, optional
        The ID of the TDT stream; must be supplied explicitly.
    onset_id : str, optional
        The ID of the TDT onset epoc; must be supplied explicitly.
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

    if not isinstance(stream_id, str) or not stream_id:
        raise ValueError("stream_id must name a TDT stream")
    if not isinstance(onset_id, str) or not onset_id:
        raise ValueError("onset_id must name a TDT onset epoc")
    if electrode_type not in (None, 'ecog'):
        raise ValueError("electrode_type must be None or 'ecog'")
    if scale and electrode_type != 'ecog':
        raise ValueError("Montage scaling requires electrode_type='ecog'")

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
    
    montage_positions = None
    if electrode_type == 'ecog':
        # Create MNE Raw object with correct channel names
        ch_names = [str(chan) for chan in streams[stream_id].channel]  # Convert channel numbers to strings


        



        montage_positions = [(7,0,0), (7,1,0), (7,2,0), (7,3,0), (6,0,0), (6,1,0), (6,2,0), (6,3,0), (5,0,0), (5,1,0), (5,2,0), (5,3,0), (4,0,0), (4,1,0), (4,2,0), (4,3,0), 
                             (3,3,0), (3,2,0), (3,1,0), (3,0,0), (2,3,0), (2,2,0), (2,1,0), (2,0,0), (1,3,0), (1,2,0), (1,1,0), (1,0,0), (0,3,0), (0,2,0), (0,1,0), (0,0,0)]
    if scale:
        xa = 0.001
        ya = 0.001
        xb = -0.015
        yb = 0
        
        # Modify each tuple in the list
        montage_positions = [(x * xa+xb, y * ya+yb, z) for x, y, z in montage_positions]

    # Create MNE info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    if montage_positions is not None:
        if len(ch_names) != len(montage_positions):
            raise ValueError("The built-in ECoG montage requires exactly 32 channels")
        montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, montage_positions)))
        info.set_montage(montage)
    
    # Create MNE Raw object
    raw = mne.io.RawArray(raw_data, info)

    return raw, events_stack
