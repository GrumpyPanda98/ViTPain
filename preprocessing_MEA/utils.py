# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:57:55 2023

@author: GZ57NM
"""
import mne
import numpy as np
import matplotlib.pyplot as plt

# Function to create MNE Raw object
def create_raw_object(data, ch_names, fs, montage_positions):
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='ecog')
    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, montage_positions)))
    info.set_montage(montage)
    raw = mne.io.RawArray(data, info)
    return raw

# Function to apply filters to MNE Raw object
def apply_filters(raw):
    raw.notch_filter([50, 100, 150, 200], filter_length='auto', notch_widths=2, method='fir', n_jobs=-1)
    raw.filter(1, 200, l_trans_bandwidth=1, h_trans_bandwidth=10, method='fir', n_jobs=-1)
    return raw

# Function to compute PSD and plot before and after applying filters
def plot_psd(raw, folder):
    before = raw.compute_psd(method='welch', fmax=300, n_fft=20000, n_jobs=-1)
    raw = apply_filters(raw)
    after = raw.compute_psd(method='welch', fmax=300, n_fft=20000, n_jobs=-1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(f'{folder} - PSD ({raw.info["nchan"]} channels)', fontsize=14)

    before.plot(average=False, axes=axes[0], show=False)
    axes[0].set_title('Before Filters')

    after.plot(average=False, axes=axes[1], show=False)
    axes[1].set_title('After Filters')

    for ax in axes:
        ax.set_xlabel('Frequency (Hz)')

    fig.tight_layout()
    plt.savefig(f'{folder}_PSD.svg')
    plt.close()

# Function to create MNE epochs object and plot evoked responses
def plot_evoked_responses(raw, events, event_id, folder):
    epochs = mne.Epochs(raw, np.int64(np.rint(events)), event_id, tmin=-0.2, tmax=0.5, detrend=None, preload=True)
    evoked = epochs.average(by_event_type=True)

    fig, axes = plt.subplots(len(evoked), 1, figsize=(10, 6))
    fig.suptitle(f'{folder} - MEA ({evoked[0].info["nchan"]} channels)', x=0.53, fontsize=14)

    for event, ax in zip(evoked, axes):
        fig = event.plot(axes=ax, show=False)
        ax.set_title(f'{event.comment}')

    fig.tight_layout()
    plt.savefig(f'{folder}_EvokedResponses.svg')
    plt.close()