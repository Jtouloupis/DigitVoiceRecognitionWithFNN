import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def step_decay_schedule(initial_lr, decay_factor, step_size):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)



# Load and preprocess data
data = []
labels = []

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "data")
data_dir_spec = os.path.join(data_dir, "spectrograms")

for filename in os.listdir(data_dir_spec):
    if filename.endswith(".png"):
        label = int(filename.split("_")[0])
        img = tf.keras.preprocessing.image.load_img(os.path.join(data_dir_spec, filename), target_size=(128, 128))

        img_array = tf.keras.preprocessing.image.img_to_array(img)

        # Normalize pixel values to the range [0, 1]
        img_array /= 255.0

        data.append(img_array)
        labels.append(label)

# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Perform one-hot encoding on the labels
labels = tf.keras.utils.to_categorical(labels, num_classes=10)

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
datagen.fit(X_train)


# Create FNN model
model = Sequential([
    Flatten(input_shape=(128, 128, 3)),

    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),  # Add Dropout layer

    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),  # Add Dropout layer

    Dense(32),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),  # Add Dropout layer

    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Usage
lr_scheduler = step_decay_schedule(initial_lr=1e-3, decay_factor=0.5, step_size=10)

# Train the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=200,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, model_checkpoint, lr_scheduler],
                    verbose=1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Plot Training History
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
