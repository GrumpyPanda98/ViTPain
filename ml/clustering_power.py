import mne
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to load the MNE raw data
def load_mne_data(subject_id):
    epochs = mne.read_epochs(f'D:\\MEA DCBUN vs UN\\Data and Notes\\Data\\Bad Channel Decimate\\{subject_id}-epo.fif', verbose=False)
    return epochs

# Function to extract features from MNE data, including power in different frequency bands
def extract_features(epochs):
    X = epochs.get_data()
    y = epochs.events[:, 2]

    # Initialize arrays to store power features
    delta_power = []
    theta_power = []
    alpha_power = []
    beta_power = []
    gamma_power = []

    # Iterate over each epoch
    for epoch_data in tqdm(X, desc='Extracting Features'):
        # Compute power spectral density
        psd, freqs = mne.time_frequency.psd_array_welch(epoch_data, sfreq=epochs.info['sfreq'], verbose=False)

        # Define frequency bands
        delta_band = (0.5, 4)  # Delta (0.5 - 4 Hz)
        theta_band = (4, 8)    # Theta (4 - 8 Hz)
        alpha_band = (8, 13)   # Alpha (8 - 13 Hz)
        beta_band = (13, 30)   # Beta (13 - 30 Hz)
        gamma_band = (30, 40)  # Gamma (30 - 40 Hz)

        # Extract power in each frequency band
        delta_power.append(np.sum(psd[:, (freqs >= delta_band[0]) & (freqs <= delta_band[1])], axis=-1))
        theta_power.append(np.sum(psd[:, (freqs >= theta_band[0]) & (freqs <= theta_band[1])], axis=-1))
        alpha_power.append(np.sum(psd[:, (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])], axis=-1))
        beta_power.append(np.sum(psd[:, (freqs >= beta_band[0]) & (freqs <= beta_band[1])], axis=-1))
        gamma_power.append(np.sum(psd[:, (freqs >= gamma_band[0]) & (freqs <= gamma_band[1])], axis=-1))

    # Convert lists to NumPy arrays
    delta_power = np.array(delta_power)
    theta_power = np.array(theta_power)
    alpha_power = np.array(alpha_power)
    beta_power = np.array(beta_power)
    gamma_power = np.array(gamma_power)

    # Stack the power features along the last axis
    power_features = np.stack([delta_power, theta_power, alpha_power, beta_power, gamma_power], axis=-1)

    return power_features, y

# List of subject IDs
subject_ids = ['Experiment 09', 'Experiment 13', 'Experiment 15', 'Experiment 16', 'Experiment 19']

# Split subjects into train and test sets
train_subjects, test_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)

# Initialize arrays to store power features and labels
train_power_features = []
train_labels = []
test_power_features = []
test_labels = []

# Iterate over subjects
for subject_id in subject_ids:
    epochs = load_mne_data(subject_id)
    power_features, labels = extract_features(epochs)

    if subject_id in train_subjects:
        train_power_features.append(power_features)
        train_labels.append(labels)
    else:
        test_power_features.append(power_features)
        test_labels.append(labels)

# Concatenate the arrays and reshape for compatibility with RandomForestClassifier
train_power_features = np.concatenate(train_power_features)
train_labels = np.concatenate(train_labels)
test_power_features = np.concatenate(test_power_features)
test_labels = np.concatenate(test_labels)

# Reshape the power features to flatten the last dimension
train_power_features = train_power_features.reshape(train_power_features.shape[0], -1)
test_power_features = test_power_features.reshape(test_power_features.shape[0], -1)

# Use Random Forest for classification
clf = RandomForestClassifier(random_state=42)
clf.fit(train_power_features, train_labels)
predicted_labels = clf.predict(test_power_features)

# Evaluate the Classification Accuracy
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")
