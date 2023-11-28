# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:35:41 2023

@author: GZ57NM
"""

import mat73
import ctypes
import tkinter as tk
from tkinter import filedialog
import os

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
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(filetypes=[("Binary MATLAB file (*.mat)", "*.mat")])

    if path:
        data_dict = mat73.loadmat(path)
        folder_name = os.path.basename(os.path.dirname(path))
        return data_dict, folder_name
    else:
        ctypes.windll.user32.MessageBoxW(0, "Please select the correct file path", "Path not found")