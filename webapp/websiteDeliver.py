# from google.cloud import logging
# logging_client = logging.Client()
# logging_client.setup_logging()
# import logging as python_logging
# python_logging.warning("Log")
import pandas as pd
import torchvision.models as models
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory
import io
import soundfile as sf
from PIL import Image
import librosa
import numpy as np
from torchvision.transforms import ToTensor
from torch import nn
import torch
from pydub import AudioSegment
import os
import base64
from scipy.spatial.distance import euclidean


class resnet18(nn.Module):
    def __init__(self, num_classes):
        super(resnet18, self).__init__()

        # Use a pre-trained ResNet model
        self.base_model = models.resnet18(weights=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) #remove last layer
        self._classifier = nn.Linear(num_ftrs, num_classes)
        self._init_weights()

    def forward(self, x1):
        x1 = self.base_model(x1)
        x1 = x1.view(x1.size(0), -1)
        score = self._classifier(x1)
        return score

    def _init_weights(self) -> None:
        for layer in self.base_model.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

def convert_audio_to_wav(input_blob):
    audio = AudioSegment.from_file(io.BytesIO(input_blob), format="webm")
    # Convert to wav
    audio = audio.export("audio.wav", format="wav")
    return audio

def convert_to_spectrogram(input_blob):
    data, samplerate = sf.read(input_blob)
    data = librosa.util.normalize(data)
    mel_spec = librosa.feature.melspectrogram(y=data, sr=samplerate, n_mels=128)
    plt.figure(figsize=(3.2, 3.2), dpi=40)
    sns.heatmap(librosa.power_to_db(mel_spec, ref=np.max), cmap='viridis', yticklabels=False,
                xticklabels=False, cbar=False)

    img_io = io.BytesIO()
    plt.savefig(img_io, bbox_inches='tight', dpi=300, format='png')
    plt.close()
    img_io.seek(0)
    img = Image.open(img_io).convert('RGB')
    img_resized = img.resize((224, 224))
    return img_resized

def image_to_base64(image):
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return img_base64

def classify_music(model, input_img):
    genres = {0: 'rock', 1: 'pop', 2: 'classical', 3: 'reggae', 4: 'disco', 5: 'jazz', 6: 'metal', 7: 'country',
              8: 'blues', 9: 'hiphop'}

    img = ToTensor()(input_img).unsqueeze(0)
    output = model(img)
    probabilities = nn.functional.softmax(output, dim=1)
    _, predicted_class = torch.max(probabilities, 1)
    # Convert MFCC image to base64-encoded string
    mfcc_img_base64 = image_to_base64(input_img)
    return probabilities.detach().numpy(), genres[predicted_class.item()], mfcc_img_base64

def recommend_songs(prob_vector, n_recommendations=5):
    # Load the probabilities from the CSV file into a pandas DataFrame
    df = pd.read_csv('genre_probabilities.csv', index_col=0)
    # Compute the Euclidean distance between the genre probabilities of the
    # given song and all other songs, and sort the DataFrame by this distance
    df['distance'] = df.apply(lambda row: euclidean(row[:-1], prob_vector), axis=1)
    df.sort_values('distance', inplace=True)
    # Return the names of the n closest songs
    return df.index[:n_recommendations].tolist()

app = Flask(__name__)
@app.route('/uploader', methods = ['POST'])
def upload_audio():
    audio_file = request.files['file']
    blob = audio_file.read()
    blob = convert_audio_to_wav(blob)
    img = convert_to_spectrogram(blob)

    # define the model
    model = resnet18(num_classes=10).eval()
    # Get the absolute path to the directory containing your script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Construct the absolute path to the model file
    model_file = os.path.join(script_dir, 'resnet18.pt')
    # Load the model parameters from a .pt file
    model.load_state_dict(torch.load(model_file))

    probabilities, genre, mfcc_img_base64 = classify_music(model, img)
    recommended_songs = recommend_songs(probabilities[0])

    return jsonify({
        'probabilities': probabilities.tolist(),
        'genre': genre,
        'mfcc_img': mfcc_img_base64,
        'recommended_songs': recommended_songs
    })

@app.route('/', methods=['GET'])
def home():
    return render_template('record.html')

@app.route('/songs/<path:filename>')
def download_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'songs'), filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    #app.run(host='0.0.0.0', port=port, debug=True)
    app.run(debug=True)



