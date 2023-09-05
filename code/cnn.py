import librosa
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import torch.nn as nn

device = torch.device('mps')

class FcnCnn(nn.Module):
    def __init__(self, num_classes):
        super(FcnCnn, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        dummy_input = torch.randn(1, 1, 128)
        dummy_output = self.conv_layers(dummy_input)
        self.output_size = int(np.prod(dummy_output.shape))

        self.fc_layers = nn.Sequential(
            nn.Linear(self.output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        print(x.shape)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

        # standardized raw data
        # audio = (audio - np.mean(audio)) / np.std(audio)

        # # compute stft
        # audio = librosa.stft(audio)
        # # convert the complex-valued STFT to magnitude and phase
        # magnitude, phase = librosa.magphase(audio)
        # # average over time
        # audio = np.mean(magnitude, axis=1)

        # # compute mfcc
        # mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
        # # computes the mean across time and make it 1d
        # audio = np.mean(mfccs.T, axis=0)

        # compute Mel spectrogram
        audio = librosa.feature.melspectrogram(y=audio, n_mels = 128)
        # convert to dB
        audio = librosa.power_to_db(audio, ref=np.max)
        # average over time
        audio = np.mean(audio, axis=1)

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

# Iterate through each sound file and extract the features
for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):
        class_label = filename.split('.')[0]
        data = extract_features(os.path.join(folder_path, filename))
        features.append(data)
        labels.append(class_label)

# Convert into a pandas dataframe
features_df = pd.DataFrame(features, columns=['feature_' + str(i) for i in range(features[0].shape[0])])
labels_df = pd.DataFrame(labels, columns=['class_label'])
fulldf = pd.concat([features_df, labels_df], axis=1)


# Split the dataset
X = np.array(fulldf.iloc[:, :-1])
y = np.array(fulldf.iloc[:, -1])
print(X.shape)
# Encoded genres
genres = { 'rock': 0, 'pop': 1, 'classical': 2, 'reggae': 3, 'disco': 4, 'jazz': 5, 'metal': 6, 'country': 7, 'blues': 8, 'hiphop': 9}
# Convert the genre labels to a tensor
y = [genres[item] for item in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create Tensor datasets
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

# dataloaders
batch_size = 64  # you can change this
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# initialize the model
model = FcnCnn(num_classes=10)
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1
actual_epochs = 0
train_acc_list = []
test_acc_list = []
train_loss_list = []
test_loss_list = []
y_true_list = []
y_pred_list = []

for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0
    actual_epochs += 1

    for i, (input, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(input.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input.size(0)  # scale by batch size
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels.to(device)).sum().item()

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / train_total

    # Evaluate the model on the test set
    test_loss = 0.0
    test_correct = 0.0
    test_total = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for input, labels in test_loader:
            outputs = model(input.to(device))
            loss = criterion(outputs, labels.to(device))
            test_loss += loss.item() * input.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels.to(device)).sum().item()
            # collect predictions and true labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    y_true_list.extend(y_true)
    y_pred_list.extend(y_pred)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# Calculate evaluation metrics
print("\nClassification Report:")
print(classification_report(y_true_list, y_pred_list))