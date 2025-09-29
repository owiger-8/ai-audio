import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

features_list = []
labels_list = [] # This will initially hold string labels "real" and "fake"

# --- 1. Load Data from JSON Lines file ---
print("Loading features from .jsonl file...")
with open("features.jsonl", 'r') as f:
    for line in f:
        data = json.loads(line)
        features_list.append(data["features"])
        labels_list.append(data["label"])

# Convert features to a NumPy array
X = np.array(features_list)

# --- Convert string labels to numbers (Label Encoding) ---
# We map "real" to 0 and "fake" to 1.
y = np.array([0 if label == 'real' else 1 for label in labels_list])

print(f"Data loaded and labels encoded: {len(X)} samples.")

# --- 2. Split Data for Training and Testing ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

# --- 3. Build the Deep Learning Model ---
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(32, activation='relu'),
    
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. Train the Model ---
print("\nStarting model training...")
history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=1)

print("Model training complete.")

# --- 5. Evaluate the Model on Unseen Data ---
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nModel Accuracy on Test Data: {accuracy*100:.2f}%")

# --- 6. Save the Trained Model ---
model.save("voice_detector_model.h5")
print(f"\n✅ Model saved to voice_detector_model.h5")