import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import os
import librosa

# --- Settings ---
DATA_PATH = "." 
SEQUENCE_LENGTH = 10 # Number of chunks in a sequence
FEATURES_PER_CHUNK = 42 # 40 MFCCs + 1 pitch + 1 zcr

# --- Feature Extraction (same as before) ---
def extract_features(audio_chunk, sr=22050):
    mfccs = np.mean(librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=40).T, axis=0)
    pitches, magnitudes = librosa.piptrack(y=audio_chunk, sr=sr)
    pitch = np.mean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes) else 0
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_chunk))
    return np.hstack((mfccs, pitch, zcr))

# --- Data Preparation for Sequences ---
print("Preparing sequence data...")
X, y = [], []
labels = {'real': 0, 'fake': 1}

for label, numeric_label in labels.items():
    folder_path = os.path.join(DATA_PATH, label)
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            audio, sr = librosa.load(file_path, sr=22050)
            
            # Chop audio into chunks and create sequences
            chunk_size = sr // 2 # 0.5 second chunks
            num_chunks = len(audio) // chunk_size
            
            for i in range(0, num_chunks - SEQUENCE_LENGTH, SEQUENCE_LENGTH // 2): # Overlapping sequences
                sequence = []
                for j in range(SEQUENCE_LENGTH):
                    chunk = audio[(i+j)*chunk_size : (i+j+1)*chunk_size]
                    features = extract_features(chunk, sr)
                    sequence.append(features)
                X.append(sequence)
                y.append(numeric_label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data prepared. Shape of training data: {X_train.shape}")

# --- Build the LSTM Model ---
print("Building LSTM model...")
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES_PER_CHUNK)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train the Model ---
print("Training model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

model.save("voice_detector_lstm.keras")
print("✅ LSTM Model saved to voice_detector_lstm.keras")