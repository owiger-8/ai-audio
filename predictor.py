import numpy as np
import librosa
import tensorflow as tf
import os

def extract_features(file_path):
 
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', sr=22050)
        
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
        pitch = np.mean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes) else 0
        
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        
        return np.hstack((mfccs, pitch, zcr))
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

MODEL_PATH = "voice_detector_model.h5"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    print("Please run the training script first to create the model.")
    exit()

print(f"Loading trained model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

AUDIO_FILE_PATH = "./2.wav" 

if not os.path.exists(AUDIO_FILE_PATH):
    print(f"Error: Audio file not found at '{AUDIO_FILE_PATH}'")
    exit()

# --- 3. Make a Prediction ---
print(f"\nProcessing audio file: {os.path.basename(AUDIO_FILE_PATH)}")
features = extract_features(AUDIO_FILE_PATH)

if features is not None:
    # Reshape features to match the model's input format (1 sample, N features)
    features = np.reshape(features, (1, -1))

    # Get the model's prediction
    prediction = model.predict(features)
    confidence = prediction[0][0] # The output is a probability between 0 and 1

    print("\n--- Prediction Result ---")
    # Your model was trained with 'real' as 0 and 'fake' as 1
    if confidence > 0.5:
        print(f"🚨  Prediction: AI Voice (Fake)")
        print(f"   Confidence: {confidence * 100:.2f}%")
    else:
        print(f"✅  Prediction: Real Voice")
        print(f"   Confidence: {(1 - confidence) * 100:.2f}%")
else:
    print("Could not extract features from the audio file.")