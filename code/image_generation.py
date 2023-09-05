import os
import glob
import seaborn as sns
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image

input_folder = "augmented_data2"
mel_folder = "mel_spectrograms2"
stft_folder = "stft_spectrograms2"
waveform_folder = "waveforms2"


folders = [mel_folder, stft_folder, waveform_folder]
# 
# for folder in folders:
#     files = glob.glob(f"{folder}/*")
#     for f in files:
#         os.remove(f)

input_files = []
waveform_files = []
mel_files = []
stft_files = []
not_processed = []

for file in os.listdir(input_folder):
    input_files.append(file[:-4])

for file in os.listdir(waveform_folder):
    waveform_files.append(file[:-4])

for file in os.listdir(stft_folder):
    stft_files.append(file[:-4])

for file in os.listdir(mel_folder):
    mel_files.append(file[:-4])

for file in input_files:
    if (file not in waveform_files) or (file not in stft_files) or (file not in mel_files):
        not_processed.append(file)


for file in os.listdir(input_folder):
    if file.endswith('.wav'):
        file_path = os.path.join(input_folder, file)
        output_file = file[:-4]
        if output_file not in not_processed:
            continue
        print(output_file)

        # Load the audio file
        y, sr = librosa.load(file_path)
        # Normalize the audio data
        y = librosa.util.normalize(y)

        # waveform
        plt.figure(figsize=(1, 1), dpi=224)
        librosa.display.waveshow(y, sr=sr)
        plt.axis('off')
        plt.savefig(waveform_folder + "/" + output_file + '.png', bbox_inches='tight', pad_inches=0, dpi=224)
        plt.close()

        # stft spectrogram
        stft_spec = librosa.stft(y=y)
        # Convert the complex-valued STFT to magnitude for visualization
        magnitude, _ = librosa.magphase(stft_spec)
        magnitude_db = librosa.power_to_db(magnitude, ref=np.max)
        plt.figure(figsize=(1, 1), dpi=224)
        librosa.display.specshow(magnitude_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
        plt.axis('off')
        plt.savefig(stft_folder + "/" + output_file + '.png', bbox_inches='tight', pad_inches=0, dpi=224)
        plt.close()


        # mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(1, 1), dpi=224)
        librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
        plt.axis('off')
        plt.savefig(mel_folder + "/" + output_file + '.png', bbox_inches='tight', pad_inches=0, dpi=224)
        plt.close()
