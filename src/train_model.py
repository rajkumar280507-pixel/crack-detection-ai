import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

def build_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def create_dummy_dataset(base_dir):
    """
    Creates a small synthetic dataset for demonstration purposes if none exists.
    """
    pos_dir = os.path.join(base_dir, 'Positive')
    neg_dir = os.path.join(base_dir, 'Negative')
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    
    # Generate 10 dummy "crack" images
    for i in range(10):
        img = np.full((224, 224, 3), 200, dtype=np.uint8)
        # Draw a "crack" - a random jagged line
        cv2.line(img, (10, 10), (210, 210), (50, 50, 50), 2)
        cv2.imwrite(os.path.join(pos_dir, f'crack_{i}.jpg'), img)
        
    # Generate 10 dummy "no-crack" images
    for i in range(10):
        img = np.full((224, 224, 3), 200, dtype=np.uint8)
        cv2.imwrite(os.path.join(neg_dir, f'no_crack_{i}.jpg'), img)

def train(dataset_path, model_save_path):
    if not os.path.exists(os.path.join(dataset_path, 'Positive')):
        print("Dataset not found. Creating synthetic dummy dataset for demonstration...")
        create_dummy_dataset(dataset_path)

    batch_size = 32
    img_size = (224, 224)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    model = build_model()
    
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=5, # Short for demo
        validation_data=val_generator
    )

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    train(os.path.join(base_dir, 'dataset'), os.path.join(base_dir, 'models', 'model.h5'))
