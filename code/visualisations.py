import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# blues----------------------------------------------------
# Load the audio file
file_name = 'songs/blues.00001.wav'
audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')


# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title('Waveform (blues)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig("Visuals/raw_blues.png")
plt.close()


# Compute the STFT of the audio
stft = librosa.stft(audio)
# Convert the complex-valued STFT to magnitude for visualization
magnitude, _ = librosa.magphase(stft)
magnitude_db = librosa.power_to_db(magnitude, ref=np.max)
# Plot the spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(magnitude_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.title('STFT Spectrogram (blues)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')
plt.savefig("Visuals/stft_blues.png")
plt.close()


# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
# Convert to dB scale
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
# Visualize the Mel spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (blues)')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.savefig("Visuals/mel_blues.png")

# classical ----------------------------------------------------
# Load the audio file
file_name = 'songs/classical.00021.wav' #20 & 21
audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')


# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title('Waveform (classical)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig("Visuals/raw_classical2.png")
plt.close()


# Compute the STFT of the audio
stft = librosa.stft(audio)
# Convert the complex-valued STFT to magnitude for visualization
magnitude, _ = librosa.magphase(stft)
magnitude_db = librosa.power_to_db(magnitude, ref=np.max)
# Plot the spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(magnitude_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.title('STFT Spectrogram (classical)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')
plt.savefig("Visuals/stft_classical2.png")
plt.close()


# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
# Convert to dB scale
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
# Visualize the Mel spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (classical)')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.savefig("Visuals/mel_classical2.png")

# country----------------------------------------------------
# Load the audio file
file_name = 'songs/country.00001.wav'
audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')


# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title('Waveform (country)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig("Visuals/raw_country.png")
plt.close()


# Compute the STFT of the audio
stft = librosa.stft(audio)
# Convert the complex-valued STFT to magnitude for visualization
magnitude, _ = librosa.magphase(stft)
magnitude_db = librosa.power_to_db(magnitude, ref=np.max)
# Plot the spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(magnitude_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.title('STFT Spectrogram (country)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')
plt.savefig("Visuals/stft_country.png")
plt.close()


# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
# Convert to dB scale
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
# Visualize the Mel spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (country)')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.savefig("Visuals/mel_country.png")

# disco----------------------------------------------------
# Load the audio file
file_name = 'songs/disco.00001.wav'
audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')


# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title('Waveform (disco)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig("Visuals/raw_disco.png")
plt.close()


# Compute the STFT of the audio
stft = librosa.stft(audio)
# Convert the complex-valued STFT to magnitude for visualization
magnitude, _ = librosa.magphase(stft)
magnitude_db = librosa.power_to_db(magnitude, ref=np.max)
# Plot the spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(magnitude_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.title('STFT Spectrogram (disco)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')
plt.savefig("Visuals/stft_disco.png")
plt.close()


# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
# Convert to dB scale
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
# Visualize the Mel spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (disco)')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.savefig("Visuals/mel_disco.png")

# hiphop----------------------------------------------------
# Load the audio file
file_name = 'songs/hiphop.00002.wav' #1 & 2
audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')


# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title('Waveform (hiphop)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig("Visuals/raw_hiphop2.png")
plt.close()


# Compute the STFT of the audio
stft = librosa.stft(audio)
# Convert the complex-valued STFT to magnitude for visualization
magnitude, _ = librosa.magphase(stft)
magnitude_db = librosa.power_to_db(magnitude, ref=np.max)
# Plot the spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(magnitude_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.title('STFT Spectrogram (hiphop)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')
plt.savefig("Visuals/stft_hiphop2.png")
plt.close()


# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
# Convert to dB scale
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
# Visualize the Mel spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (hiphop)')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.savefig("Visuals/mel_hiphop2.png")

# jazz----------------------------------------------------
# Load the audio file
file_name = 'songs/jazz.00006.wav' #4 & 6
audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')


# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title('Waveform (jazz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig("Visuals/raw_jazz2.png")
plt.close()


# Compute the STFT of the audio
stft = librosa.stft(audio)
# Convert the complex-valued STFT to magnitude for visualization
magnitude, _ = librosa.magphase(stft)
magnitude_db = librosa.power_to_db(magnitude, ref=np.max)
# Plot the spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(magnitude_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.title('STFT Spectrogram (jazz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')
plt.savefig("Visuals/stft_jazz2.png")
plt.close()


# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
# Convert to dB scale
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
# Visualize the Mel spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (jazz)')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.savefig("Visuals/mel_jazz2.png")

# metal----------------------------------------------------
# Load the audio file
file_name = 'songs/metal.00001.wav'
audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')


# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title('Waveform (metal)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig("Visuals/raw_metal.png")
plt.close()


# Compute the STFT of the audio
stft = librosa.stft(audio)
# Convert the complex-valued STFT to magnitude for visualization
magnitude, _ = librosa.magphase(stft)
magnitude_db = librosa.power_to_db(magnitude, ref=np.max)
# Plot the spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(magnitude_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.title('STFT Spectrogram (metal)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')
plt.savefig("Visuals/stft_metal.png")
plt.close()


# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
# Convert to dB scale
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
# Visualize the Mel spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (metal)')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.savefig("Visuals/mel_metal.png")

# pop----------------------------------------------------
# Load the audio file
file_name = 'songs/pop.00001.wav'
audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')


# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title('Waveform (pop)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig("Visuals/raw_pop.png")
plt.close()


# Compute the STFT of the audio
stft = librosa.stft(audio)
# Convert the complex-valued STFT to magnitude for visualization
magnitude, _ = librosa.magphase(stft)
magnitude_db = librosa.power_to_db(magnitude, ref=np.max)
# Plot the spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(magnitude_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.title('STFT Spectrogram (pop)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')
plt.savefig("Visuals/stft_pop.png")
plt.close()


# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
# Convert to dB scale
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
# Visualize the Mel spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (pop)')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.savefig("Visuals/mel_pop.png")

# reggae----------------------------------------------------
# Load the audio file
file_name = 'songs/reggae.00001.wav'
audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')


# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title('Waveform (reggae)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig("Visuals/raw_reggae.png")
plt.close()


# Compute the STFT of the audio
stft = librosa.stft(audio)
# Convert the complex-valued STFT to magnitude for visualization
magnitude, _ = librosa.magphase(stft)
magnitude_db = librosa.power_to_db(magnitude, ref=np.max)
# Plot the spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(magnitude_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.title('STFT Spectrogram (reggae)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')
plt.savefig("Visuals/stft_reggae.png")
plt.close()


# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
# Convert to dB scale
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
# Visualize the Mel spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (reggae)')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.savefig("Visuals/mel_reggae.png")

# rock----------------------------------------------------
# Load the audio file
file_name = 'songs/rock.00001.wav'
audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')


# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title('Waveform (rock)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig("Visuals/raw_rock.png")
plt.close()


# Compute the STFT of the audio
stft = librosa.stft(audio)
# Convert the complex-valued STFT to magnitude for visualization
magnitude, _ = librosa.magphase(stft)
magnitude_db = librosa.power_to_db(magnitude, ref=np.max)
# Plot the spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(magnitude_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.title('STFT Spectrogram (rock)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')
plt.savefig("Visuals/stft_rock.png")
plt.close()


# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
# Convert to dB scale
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
# Visualize the Mel spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (rock)')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.savefig("Visuals/mel_rock.png")


