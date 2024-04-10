
# Music Genre Classification Using CNN / DNN / RNN

This project aims to classify music genres using Convolutional Neural Networks (CNNs). It extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio files as features for training the CNN model. The dataset consists of music files from various genres, divided into segments to increase the data volume and improve the model's accuracy.

## Getting Started

### Prerequisites

- Python 3.x
- Libraries: `librosa`, `matplotlib`, `numpy`, `keras`, `sklearn`

Install the required libraries using pip:

```bash
pip install librosa matplotlib numpy keras sklearn
```

### Dataset

The dataset should be organized in folders corresponding to each genre. Each folder contains audio files for that particular genre.

```
Data
└── genres_original
    ├── blues
    ├── classical
    ├── jazz
    ...
```

### Usage

1. **Feature Extraction**

   Run the script to extract MFCC features from the audio files and save them into a JSON file (`data.json`). This file will be used for training the model.

   ```bash
   python feature_extraction.py
   ```

2. **Training the Model**

   Train the model using the extracted features. The script splits the data into training, validation, and test sets, then constructs and trains the CNN model.

   ```bash
   python train_model.py
   ```

3. **Evaluating the Model**

   Evaluate the trained model's performance on the test set. The script will output the accuracy of the model on the test data.

   ```bash
   python evaluate_model.py
   ```

## Model Architecture

The model consists of several convolutional layers followed by max-pooling layers, dropout layers for regularization, and a fully connected layer as the output. Regularization techniques like L2 regularization and dropout are used to prevent overfitting.

## Results

After training CNN model for 80 epochs, it achieved a test accuracy of 83%. 
ANN we get highest with 94% test accuracy.

## Future Work

- Experiment with different architectures and hyperparameters to improve accuracy.
- Increase the dataset size by adding more genres and audio samples.
- Explore the use of other features in addition to MFCCs for genre classification.

## Authors

- [Siwei Tan](siwtan@ucdavis.edu)
- [Fangyu Zhu](fazhu@ucsd.edu)
- [Haitao Peng](201900800133@mail.sdu.edu.cn)
- [Zuge Li](zgli@ucdavis.edu)


## Publication

This project is part of our research presented in the paper titled ["Music Classification with Convolutional and Artificial Neural Network"](https://iopscience.iop.org/article/10.1088/1742-6596/2580/1/012059) by Fangyu Zhu, Haitao Peng, Siwei Tan, and Zuge Li. The paper details the methodology, experiments, and results of using CNNs and ANNs for music genre classification. 

Abstract: Music genre classification focuses on efficiently finding expected music with a similar genre through numerous melodies, which could better satisfy the tastes and expectations of the users when listening to music. This paper proposes a new method to classify different kinds of music with Artificial Neural networks (ANN) and Convolutional Neural Networks (CNNs). 

For further details, please refer to our publication:

Li, Zuge & Peng, Haitao & Tan, Siwei & Zhu, Fangyu. (2023). Music classification with convolutional and artificial neural network. Journal of Physics: Conference Series. 2580. 012059. 10.1088/1742-6596/2580/1/012059.
