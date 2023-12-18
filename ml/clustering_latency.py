# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:24:51 2023

@author: GZ57NM
"""

import mne
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

subject_ids = ['Experiment 09', 'Experiment 13', 'Experiment 15', 'Experiment 16', 'Experiment 19']


# Function to load the MNE raw data
def load_mne_data(subject_id):
    epochs = mne.read_epochs(f'D:\\MEA DCBUN vs UN\\Data and Notes\\Data\\Bad Channel Decimate\\{subject_id}-epo.fif', verbose=False)
    return epochs

# Function to extract features from MNE data, including N1-P1 latency
def extract_features(epochs):
    X = epochs.get_data()
    y = epochs.events[:, 2]

    # Initialize an array to store N1-P1 latencies
    latencies = []

    # Iterate over each epoch
    for epoch_data in tqdm(X, desc='Extracting Features'):
        # Assuming that your data has channels along axis 0 and time along axis 2
        channel_data = epoch_data.mean(axis=0)  # Average across channels

        # Find the index of the minimum (N1) and maximum (P1) values in the averaged data
        n1_index = np.argmin(channel_data)
        p1_index = np.argmax(channel_data)

        # Convert the indices to time (assuming your data is in seconds)
        n1_latency = epochs.times[n1_index]
        p1_latency = epochs.times[p1_index]

        # Compute N1-P1 latency
        latency = p1_latency - n1_latency
        latencies.append(latency)

    # Convert the list to a NumPy array
    latencies = np.array(latencies)

    # Now 'latencies' contains the N1-P1 latencies for each epoch
    return latencies, y

# Split subjects into train and test sets
train_subjects, test_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)

# Load and extract features for train data
train_latencies = []
train_labels = []

for subject_id in train_subjects:
    epochs = load_mne_data(subject_id)
    latencies, labels = extract_features(epochs)
    train_latencies.append(latencies)
    train_labels.append(labels)

train_latencies = np.concatenate(train_latencies)
train_labels = np.concatenate(train_labels)

# Load and extract features for test data
test_latencies = []
test_labels = []

for subject_id in test_subjects:
    epochs = load_mne_data(subject_id)
    latencies, labels = extract_features(epochs)
    test_latencies.append(latencies)
    test_labels.append(labels)

test_latencies = np.concatenate(test_latencies)
test_labels = np.concatenate(test_labels)

# Use KMeans as the clustering method on train data
n_clusters = len(np.unique(train_labels))  # Number of clusters = number of unique classes
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
train_clusters = kmeans.fit_predict(train_latencies.reshape(-1, 1))

# Use the same clustering model to predict clusters for test data
test_clusters = kmeans.predict(test_latencies.reshape(-1, 1))

# Use a classifier (Random Forest) to classify based on clusters
clf = RandomForestClassifier(random_state=42)
clf.fit(train_clusters.reshape(-1, 1), train_labels)
predicted_labels = clf.predict(test_clusters.reshape(-1, 1))

# Evaluate the Classification Accuracy
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")
