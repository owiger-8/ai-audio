import numpy as np
import librosa
import tensorflow as tf

## 1. Feature Extraction Function (Must be IDENTICAL to the one used for training)
def extract_features(file_path):
    """Extracts a fixed-size feature vector from an audio file."""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
        pitch = np.mean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes > 0) else 0

        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        
        features = np.hstack((mfccs, pitch, zcr))
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

## 2. Load the Trained Model
# Make sure the filename matches the one you saved (e.g., .h5 or .keras)
MODEL_PATH = "voice_detector_model.h5" 
print(f"Loading trained model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

## 3. Set the Path to Your New Audio File 🎤
# IMPORTANT: Replace this with the path to a .wav file you want to test
# This file should NOT have been used in your training or test data
AUDIO_FILE_PATH = "./1.wav" 

## 4. Make a Prediction
print(f"\nProcessing audio file: {AUDIO_FILE_PATH}")
# Extract features from the new audio file
new_features = extract_features(AUDIO_FILE_PATH)

if new_features is not None:
    # Reshape features to match the model's input format (1 sample, N features)
    new_features = np.reshape(new_features, (1, -1))

    # Get the model's prediction
    prediction = model.predict(new_features)
    
    # The model outputs a probability (a number between 0 and 1)
    # 0 is 'real', 1 is 'fake'
    confidence = prediction[0][0]

    print("\n--- Prediction Result ---")
    if confidence > 0.5:
        print(f"✅ This is likely an AI voice (fake).")
        print(f"Confidence: {confidence * 100:.2f}%")
    else:
        print(f"✅ This is likely a Real voice.")
        print(f"Confidence: {(1 - confidence) * 100:.2f}%")
else:
    print("Could not extract features from the audio file.")