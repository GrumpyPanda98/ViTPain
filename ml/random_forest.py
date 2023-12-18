# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:30:39 2023

@author: GZ57NM
"""

import mne
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plyer as pl
import numpy as np
from tqdm import tqdm

# Function to load the MNE raw data
def load_mne_data(subject_id):
    epochs = mne.read_epochs(rf'D:\MEA DCBUN vs UN\Data and Notes\Data\Bad Channel Decimate\{subject_id}-epo.fif', verbose=False)
    return epochs

# Function to extract features from MNE data
def extract_features(epochs):
    X = epochs.get_data()
    X = X.reshape(X.shape[0], -1)  # Flatten the data
    y = epochs.events[:, 2]
    return X, y

# Function to train a random forest classifier
def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Use tqdm for the training loop
    with tqdm(desc="Training Random Forest", total=clf.n_estimators, unit="tree") as pbar:
        for _ in range(clf.n_estimators):
            clf.fit(X_train, y_train)
            pbar.update(1)
    
    return clf

# Function to evaluate the classifier
def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy*100} %')




all_X, all_y = [], []

subjects = ['Experiment 09', 'Experiment 13', 'Experiment 15', 'Experiment 16', 'Experiment 19']

# Use tqdm to create a loading bar for subjects
for subject_id in tqdm(subjects, desc="Processing subjects", unit="subject"):
    
    # Load MNE data
    epochs = load_mne_data(subject_id)
    
    # Extract features
    X, y = extract_features(epochs)

    all_X.append(X)
    all_y.append(y)

# Split data for each subject into training and testing sets
X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
for X_subject, y_subject in zip(all_X, all_y):
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
        X_subject, y_subject, test_size=0.2, random_state=42, stratify=y_subject
    )
    X_train_list.append(X_train_sub)
    X_test_list.append(X_test_sub)
    y_train_list.append(y_train_sub)
    y_test_list.append(y_test_sub)

# Combine data from all subjects
X_train = np.concatenate(X_train_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

# Train random forest classifier on combined training data
clf = train_random_forest(X_train, y_train)

# Evaluate the classifier on combined test data
evaluate_classifier(clf, X_test, y_test)

