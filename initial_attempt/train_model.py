import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

# Define paths for training and testing data
train_path = 'train/'
test_path = 'test/'

# Data generators for preprocessing and train/validation split
datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input, validation_split=0.2)
train_batches = datagen.flow_from_directory(
    directory=train_path, target_size=(64, 64), class_mode='sparse', batch_size=10, shuffle=True, subset='training')
val_batches = datagen.flow_from_directory(
    directory=train_path, target_size=(64, 64), class_mode='sparse', batch_size=10, shuffle=False, subset='validation')

# Determine number of classes dynamically
num_classes = train_batches.num_classes

# Create a mapping from class index to class name for labeling
inverse_label_map = {v: k for k, v in train_batches.class_indices.items()}

# Define the CNN model
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model with sparse categorical loss to match integer labels
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

# Train the model with validation split
history = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop], validation_data=val_batches)

# Save the model
model.save('asl_cnn_best_tf.h5')

# Evaluate on the validation set
imgs, labels = next(val_batches)
scores = model.evaluate(imgs, labels, verbose=0)
print(f"Loss: {scores[0]}, Accuracy: {scores[1] * 100}%")

# Make predictions
predictions = model.predict(imgs, verbose=0)
print("Predictions on a small set of test data:")
for ind, i in enumerate(predictions):
    pred_idx = np.argmax(i)
    print(inverse_label_map[pred_idx], end='   ')

print("\nActual labels:")
for lbl in labels:
    print(inverse_label_map[lbl], end='   ')