# speech_emotion_testing.py

import numpy as np
import librosa
import tensorflow as tf
from speech_emotion_models import extract_mfcc

# --- Configuration ---
EMOTIONS = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'neutral'}  # Must match training labels
MAX_PAD_LEN = 173

# --- Load Pre-trained Models ---
cnn_model = tf.keras.models.load_model('cnn_emotion.h5')
rnn_model = tf.keras.models.load_model('rnn_emotion.h5')
lstm_model = tf.keras.models.load_model('lstm_emotion.h5')

# --- Predict Emotion Function ---
def predict_emotion(model, file_path):
    mfcc = extract_mfcc(file_path, MAX_PAD_LEN)
    X = mfcc[np.newaxis, ..., np.newaxis]  # Add batch and channel dims
    prediction = model.predict(X)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return EMOTIONS[predicted_class]

# --- Example Testing ---
if __name__ == '__main__':
    test_audio = 'path_to_test_audio.wav'  # Update with your test file path

    print("CNN Prediction:", predict_emotion(cnn_model, test_audio))
    print("RNN Prediction:", predict_emotion(rnn_model, test_audio))
    print("LSTM Prediction:", predict_emotion(lstm_model, test_audio))
