import numpy as np
import librosa
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Function to preprocess audio files and convert them into spectrograms
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    # Generate Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

# Load audio dataset
def load_dataset(audio_folder, target_shape=(128, 128)):
    spectrograms = []
    labels = []

    # Iterate through audio files in the specified folder
    for label, class_name in enumerate(os.listdir(audio_folder)):
        class_folder = os.path.join(audio_folder, class_name)
        if os.path.isdir(class_folder):
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                if file_path.endswith('.wav'):  # Adjust if using different file formats
                    spectrogram = preprocess_audio(file_path)

                    # Pad or crop the spectrogram to the target shape
                    if spectrogram.shape[1] < target_shape[1]:
                        padded = np.pad(spectrogram, ((0, 0), (0, target_shape[1] - spectrogram.shape[1])), mode='constant')
                    else:
                        padded = spectrogram[:, :target_shape[1]]

                    spectrograms.append(padded)
                    labels.append(label)

    # Convert lists to numpy arrays
    return np.array(spectrograms), np.array(labels)

# Specify the directory containing your audio files
audio_folder = 'mini_speech_commands'
spectrograms, labels = load_dataset(audio_folder)

# Reshape the spectrograms to include a channel dimension
spectrograms = spectrograms[..., np.newaxis]  # Add channel dimension

# Verify unique labels
unique_labels = np.unique(labels)
num_classes = len(unique_labels)
print(f"Number of classes: {num_classes}")  # Print number of classes to verify

# One-hot encode the labels
# Ensure that labels are mapped to the correct range [0, num_classes-1]
labels = np.array([np.where(unique_labels == label)[0][0] for label in labels])
y_train = to_categorical(labels, num_classes)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(spectrograms, y_train, test_size=0.2, random_state=42)

# Build the CNN model (Functional)
input_shape = X_train.shape[1:]  # Shape of your spectrograms
inputs = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.4)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(patience = 5, verbose = 2)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks = [early_stopping], epochs=100, batch_size=32)

# Save the model
model.save('audio_classification_model.h5')

print("Model training complete and saved as 'audio_classification_model.h5'")