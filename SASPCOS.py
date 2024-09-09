import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from PIL import Image, UnidentifiedImageError
import os

# Set up directories
train_dir = r'C:\Users\khatr\OneDrive - Prime College\Desktop\ML\data\train'
test_dir = r'C:\Users\khatr\OneDrive - Prime College\Desktop\ML\data\test'

# Image preprocessing and augmentation
target_size = (128, 128)
batch_size = 8
epochs = 100  # Reduced to avoid long training times

# Ensure TensorFlow uses the GPU if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and will be used for training.")
else:
    print("GPU not available, training will proceed on the CPU.")

# Custom function to safely load images and skip corrupt ones
def safe_load_img(path, color_mode='rgb', target_size=None, interpolation='nearest'):
    try:
        img = tf.keras.utils.load_img(path, color_mode=color_mode, target_size=target_size, interpolation=interpolation)
        return img
    except (UnidentifiedImageError, OSError, ValueError) as e:
        print(f"Skipping invalid image {path}: {e}")
        return None

# Custom data generator that skips corrupt images
class SafeImageDataGenerator(ImageDataGenerator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []
        for i in index_array:
            img_path = self.filepaths[i]
            img = safe_load_img(img_path, color_mode=self.color_mode, target_size=self.target_size)
            if img is not None:
                img_array = tf.keras.utils.img_to_array(img)
                batch_x.append(img_array)
                batch_y.append(self.labels[i])
        return np.array(batch_x), np.array(batch_y)

train_datagen = SafeImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = SafeImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Define the CNN model
def build_cnn_model(input_shape):
    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output
    model.add(Flatten())

    # Fully Connected Layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Build the model
input_shape = (128, 128, 3)  # Keep input consistent
cnn_model = build_cnn_model(input_shape)

# Define callbacks
checkpoint = ModelCheckpoint(
    'best_model.keras',  # Save the best model
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Stop training if no improvement for 10 epochs
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,  # Reduce learning rate if no improvement for 5 epochs
    min_lr=0.00001,
    verbose=1
)

# Train the model
history = cnn_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Load the best saved model for inference
best_model = tf.keras.models.load_model('best_model.keras')
