import sounddevice as sd
import numpy as np
import tensorflow as tf
import os
import sys
import librosa
from collections import deque
import time

## 1. Global Settings
# --- Use your device index or name ---
DEVICE_NAME = 2 

# --- Load your ORIGINAL MLP model ---
MODEL_PATH = "voice_detector_model.h5" 

SAMPLE_RATE = 22050
UPDATE_INTERVAL = 0.5  # Seconds to wait between predictions
CHUNK_DURATION = 0.5   # Duration of each audio chunk to process
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# --- History Settings ---
HISTORY_SECONDS = 42  # Analyze the last 4 seconds of audio
MAX_HISTORY_LENGTH = int(HISTORY_SECONDS / CHUNK_DURATION)

# A deque to hold the recent history of feature vectors
feature_history = deque(maxlen=MAX_HISTORY_LENGTH)

## 2. Load Model & Feature Extraction Function
print(f"Loading MLP model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

def extract_features(audio_chunk, sr=22050):
    """Extracts a fixed-size feature vector from an audio chunk."""
    try:
        mfccs = np.mean(librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=40).T, axis=0)
        pitches, magnitudes = librosa.piptrack(y=audio_chunk, sr=sr)
        pitch = np.mean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes) else 0
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_chunk))
        return np.hstack((mfccs, pitch, zcr))
    except Exception as e:
        print(f"Feature extraction error: {e}", file=sys.stderr)
        return None

## 3. Main Processing Loop
def process_stream():
    """Main function to handle audio stream and prediction."""
    def audio_callback(indata, frames, time, status):
        """This function is called for each new audio chunk from the stream."""
        if status: print(status)
        features = extract_features(indata.flatten())
        if features is not None:
            feature_history.append(features)

    # Start the audio stream in the background
    stream = sd.InputStream(
        device=DEVICE_NAME, samplerate=SAMPLE_RATE, channels=1,
        dtype='float32', blocksize=CHUNK_SAMPLES, callback=audio_callback
    )
    with stream:
        print("🎙️  Listening... Press Ctrl+C to stop.")
        while True:
            # Only predict if we have some features in our history
            if len(feature_history) > 0:
                
                # --- THE KEY STEP: Average the feature history ---
                averaged_features = np.mean(np.array(feature_history), axis=0)
                
                # Reshape the single averaged vector for the model
                reshaped_features = np.reshape(averaged_features, (1, -1))
                
                # Make a prediction using the MLP model
                prediction = model.predict(reshaped_features, verbose=0)
                confidence = prediction[0][0]
                
                # Display the output
                os.system('cls' if os.name == 'nt' else 'clear')
                print("🎙️  Listening... Press Ctrl+C to stop.")
                print(f"Analyzing last ~{len(feature_history) * CHUNK_DURATION:.1f} seconds of audio...")
                print("\n" + "="*35)
                if confidence > 0.5:
                    print(f"🚨  Prediction: AI Voice (Fake)")
                    print(f"   Confidence: {confidence * 100:.2f}%")
                else:
                    print(f"✅  Prediction: Real Voice")
                    print(f"   Confidence: {(1 - confidence) * 100:.2f}%")
                print("="*35)
            
            time.sleep(UPDATE_INTERVAL) # Wait before the next prediction

if __name__ == "__main__":
    try:
        process_stream()
    except KeyboardInterrupt:
        print("\nStopping detection.")
    except Exception as e:
        print(f"An error occurred: {e}")