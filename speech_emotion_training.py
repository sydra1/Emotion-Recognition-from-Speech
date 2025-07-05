# speech_emotion_training.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from speech_emotion_models import extract_mfcc, build_cnn_model, build_rnn_model, build_lstm_model

# --- Configuration ---
DATA_DIR = 'path_to_your_dataset_folder'  # Update this to your dataset path
EMOTIONS = {'angry': 0, 'happy': 1, 'sad': 2, 'neutral': 3}  # Example emotion classes
MAX_PAD_LEN = 173
NUM_CLASSES = len(EMOTIONS)

# --- Load and Preprocess Data ---
def load_data():
    features = []
    labels = []

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(DATA_DIR, emotion)
        for file in os.listdir(emotion_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(emotion_dir, file)
                mfcc = extract_mfcc(file_path, MAX_PAD_LEN)
                features.append(mfcc)
                labels.append(EMOTIONS[emotion])

    X = np.array(features)
    X = X[..., np.newaxis]  # Add channel dimension
    y = to_categorical(labels, num_classes=NUM_CLASSES)

    return train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model Function ---
def train_model(model, X_train, X_test, y_train, y_test, model_name='model'):
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
    model.save(f'{model_name}.h5')

# --- Main Execution ---
if __name__ == '__main__':
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # --- Train CNN Model ---
    cnn_model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=NUM_CLASSES)
    train_model(cnn_model, X_train, X_test, y_train, y_test, model_name='cnn_emotion')

    # --- Train RNN Model ---
    rnn_model = build_rnn_model(input_shape=X_train.shape[1:], num_classes=NUM_CLASSES)
    train_model(rnn_model, X_train, X_test, y_train, y_test, model_name='rnn_emotion')

    # --- Train LSTM Model ---
    lstm_model = build_lstm_model(input_shape=X_train.shape[1:], num_classes=NUM_CLASSES)
    train_model(lstm_model, X_train, X_test, y_train, y_test, model_name='lstm_emotion')
