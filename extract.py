import os
import json
import numpy as np
import librosa

def extract_features(file_path):
    """Extracts a fixed-size feature vector from an audio file."""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
        pitch = np.mean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes > 0) else 0
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        features = np.hstack((mfccs, pitch, zcr))
        return features.tolist()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Set the path to the current directory where the script is located ---
DATA_PATH = "." 
real_path = os.path.join(DATA_PATH, "real") # Assumes a 'real' folder exists here
ai_path = os.path.join(DATA_PATH, "fake")   # Assumes a 'fake' folder exists here
output_filename = "features.jsonl"

print(f"Starting incremental feature extraction. Output will be saved to {output_filename}")

# Open the output file in write mode ('w') to start a fresh file
with open(output_filename, 'w') as f:
    # Process real voices (label = "real")
    print("Processing real voices...")
    for filename in os.listdir(real_path):
        if filename.endswith(".wav"):
            path = os.path.join(real_path, filename)
            features = extract_features(path)
            if features is not None:
                record = {"features": features, "label": "real"}
                f.write(json.dumps(record) + '\n')

    # Process AI voices (label = "fake")
    print("Processing AI voices...")
    for filename in os.listdir(ai_path):
        if filename.endswith(".wav"):
            path = os.path.join(ai_path, filename)
            features = extract_features(path)
            if features is not None:
                record = {"features": features, "label": "fake"}
                f.write(json.dumps(record) + '\n')

print(f"✅ Feature extraction complete. Data saved in {output_filename}")