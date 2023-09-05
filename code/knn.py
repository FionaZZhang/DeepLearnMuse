import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GroupShuffleSplit



def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

        # standardized raw data
        audio = (audio - np.mean(audio)) / np.std(audio)

        # compute stft
        audio = librosa.stft(audio)
        # convert the complex-valued STFT to magnitude and phase
        magnitude, phase = librosa.magphase(audio)
        # average over time
        audio = np.mean(magnitude, axis=1)

        # # compute mfcc
        # mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
        # # computes the mean across time and make it 1d
        # audio = np.mean(mfccs.T, axis=0)

        # # compute Mel spectrogram
        # audio = librosa.feature.melspectrogram(y=audio, n_mels = 128)
        # # convert to dB
        # audio = librosa.power_to_db(audio, ref=np.max)
        # # average over time
        # audio = np.mean(audio, axis=1)

        #print(audio.shape)

    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}, error: {str(e)}")
        return None

    return audio


# path to your folder
folder_path = 'augmented_data'

# Lists to hold features and labels
features = []
labels = []
origin = []
# Iterate through each sound file and extract the features
for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):
        genre, song_id, png = filename.split(".")
        genre = filename.split('.')[0]
        data = extract_features(os.path.join(folder_path, filename))
        features.append(data)
        labels.append(genre)
        origin.append(genre + str(int(song_id) // 5))

# Convert into a pandas dataframe
features_df = pd.DataFrame(features, columns=['feature_' + str(i) for i in range(features[0].shape[0])])
labels_df = pd.DataFrame(labels, columns=['class_label'])
fulldf = pd.concat([features_df, labels_df], axis=1)

# Split the dataset
X = np.array(fulldf.iloc[:, :-1])
y = np.array(fulldf.iloc[:, -1])

# Convert the origin list into a numpy array
origin_array = np.array(origin)
# Initialize GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# Get the train-test split indices
train_idx, test_idx = next(gss.split(X, y, groups=origin_array))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


# Apply KNN classifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(classification_report(y_test, predictions))
