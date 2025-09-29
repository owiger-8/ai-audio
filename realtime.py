import sounddevice as sd
import numpy as np
import tensorflow as tf
import queue
import threading
import os
import sys
import librosa

## 1. Global Settings
# --- IMPORTANT: CHANGE THIS TO YOUR DEVICE NAME ---
# Find this name by running a separate script with 'print(sd.query_devices())'
# Example for Windows: "Stereo Mix (Realtek High Definition Audio)"
DEVICE_NAME = 2 

MODEL_PATH = "voice_detector_model.h5"
SAMPLE_RATE = 22050  # Sample rate used during training
CHUNK_DURATION = 2   # Duration of audio chunks to analyze in seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# A queue to safely pass audio data between threads
audio_queue = queue.Queue()

## 2. Load Model & Feature Extraction Function
print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

def extract_features(audio_chunk):
    """Extracts a fixed-size feature vector from an audio chunk."""
    try:
        mfccs = np.mean(librosa.feature.mfcc(y=audio_chunk, sr=SAMPLE_RATE, n_mfcc=40).T, axis=0)
        pitches, magnitudes = librosa.piptrack(y=audio_chunk, sr=SAMPLE_RATE)
        pitch = np.mean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes > 0) else 0
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_chunk))
        features = np.hstack((mfccs, pitch, zcr))
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}", file=sys.stderr)
        return None

## 3. Audio Streaming Threads
def audio_callback(indata, frames, time, status):
    """This function is called by the audio stream for each new chunk."""
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def start_recording():
    """Starts the audio recording stream in a background thread."""
    try:
        stream = sd.InputStream(
            device=DEVICE_NAME,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback
        )
        stream.start()
        print(f"🎧 Listening to device '{DEVICE_NAME}'... Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        print("Please check if the DEVICE_NAME is correct and the device is enabled.")
        os._exit(1) # Use os._exit to force exit from a thread

## 4. Main Processing and Prediction Loop
def process_audio():
    """Continuously pulls audio from the queue and makes predictions."""
    while True:
        audio_chunk = audio_queue.get().flatten()
        features = extract_features(audio_chunk)
        
        if features is not None:
            features = np.reshape(features, (1, -1))
            prediction = model.predict(features, verbose=0)
            confidence = prediction[0][0]

            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"🎧 Listening to device '{DEVICE_NAME}'... Press Ctrl+C to stop.")
            print("\n" + "="*35)
            if confidence > 0.5:
                print(f"🚨  Prediction: AI Voice (Fake)")
                print(f"   Confidence: {confidence * 100:.2f}%")
            else:
                print(f"✅  Prediction: Real Voice")
                print(f"   Confidence: {(1 - confidence) * 100:.2f}%")
            print("="*35)

if __name__ == "__main__":
    record_thread = threading.Thread(target=start_recording, daemon=True)
    record_thread.start()
    
    try:
        process_audio()
    except KeyboardInterrupt:
        print("\nStopping detection.")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        sys.exit(1)