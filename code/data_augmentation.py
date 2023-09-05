import os
import random
import numpy as np
import soundfile as sf

# path to your folder
folder_path = 'songs'

# output folder
output_folder = 'augmented_data2'
os.makedirs(output_folder, exist_ok=True)

# sampling rate, modify as per your audio files (standard is 22050 for .wav)
sampling_rate = 22050

# duration of each clip in seconds
clip_duration = 3

# number of clips to create from each file
num_clips = 5

# duration of each clip in samples
clip_length = clip_duration * sampling_rate

# dictionary to keep track of the count of clips per genre
genre_counter = {}

# iterate over files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):
        # split the filename to get the genre and index
        genre, num, wav = filename.split('.')

        # if the genre is not in the dictionary, add it
        if genre not in genre_counter:
            genre_counter[genre] = 0

        # load audio file
        data, sr = sf.read(os.path.join(folder_path, filename))

        # generate num_clips random starting points
        start_points = random.sample(range(0, len(data) - clip_length), num_clips)

        # create a new clip for each starting point
        for start in start_points:
            # extract clip
            clip = data[start: start + clip_length]

            # # create output filename
            # output_filename = f"{genre}.{str(genre_counter[genre]).zfill(5)}.wav"
            # create output filename
            output_filename = f"{genre}.{num}.{str(genre_counter[genre]).zfill(5)}.wav"

            # write clip to new file
            sf.write(os.path.join(output_folder, output_filename), clip, sr)

            # increment the counter for this genre
            genre_counter[genre] += 1
