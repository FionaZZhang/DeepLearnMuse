# DeepLearnMuse
Author: Junfei Zhang

**Abstract:** Music recommendation systems have emerged as a vital component to enhance user experience and satisfaction for the music streaming services, which dominates music consumption. The key challenge in improving these recommender systems lies in comprehending the complexity of music data, specifically for the underpinning music genre classification. The limitations of manual genre classification have highlighted the need for a more advanced system, namely the Automatic Music Genre Classification (AMGC) system. While traditional machine learning techniques have shown potential in genre classification, they heavily rely on manually engineered features and feature selection, failing to capture the full complexity of music data. On the other hand, deep learning classification architectures like the traditional Convolutional Neural Networks (CNN) are effective in capturing the spatial hierarchies but struggle to capture the temporal dynamics inherent in music data. To address these challenges, this study proposes a novel approach using visual spectrograms as input, and propose a hybrid model that combines the strength of the Residual neural Network (ResNet) and the Gated Recurrent Unit (GRU). This model is designed to provide a more comprehensive analysis of music data, offering the potential to improve the music recommender systems through achieving a more comprehensive analysis of music data and hence potentially more accurate genre classification.

File structure:

`code`
1. `pytorch.ipynb`: the scripts to the training and testing of the proposed model.
2. `svm.py`, `knn.py`, `cnn.py`: testing for the baselines.
3. `data_augmentation.py`: data augmentation of the songs.
4. `image_generation.py`: getting the mfcc, stft, and waveform from .wav files.
5. `visualisations.py`: creating visualisation for the presentation and the paper.

`songs`: the original dataset. \
`webapp`: the Flask application of a music recommender that was developed and deployed, contains the necessary materials for deployment. \
`presentation`: contains the paper and the slides for knowledge sharing.

Other links: \
[Preprint](https://arxiv.org/abs/2307.10773v1) \
[Website](https://deeplearnmuse-3t5mgrwzwa-km.a.run.app/)
