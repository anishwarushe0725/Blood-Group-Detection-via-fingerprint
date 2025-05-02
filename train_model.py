import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import json

# Enable memory growth for GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"GPU memory growth enabled on {len(physical_devices)} devices")

# 1. Data Preparation with proper class count determination
def preprocess_dataset(dataset_path, img_size=(128, 128)):
    """
    Preprocess fingerprint images and organize them for training
    """
    # First, determine the number of classes by counting directories
    blood_groups = os.listdir(dataset_path)
    num_classes = len(blood_groups)
    print(f"Found {num_classes} blood groups: {blood_groups}")
    
    # Create blood group dictionary
    blood_group_dict = {idx: group for idx, group in enumerate(blood_groups)}
    print(f"Blood group mapping: {blood_group_dict}")
    
    images = []
    labels = []
    
    # Process images in batches to save memory
    for idx, blood_group in enumerate(blood_groups):
        blood_group_path = os.path.join(dataset_path, blood_group)
        print(f"Processing blood group {blood_group} (class {idx})")
        
        image_paths = [os.path.join(blood_group_path, img_name) 
                      for img_name in os.listdir(blood_group_path)
                      if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"Found {len(image_paths)} images for blood group {blood_group}")
        
        # Process images in smaller batches to save memory
        batch_size = 100
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_labels = []
            
            for img_path in batch_paths:
                try:
                    # Read and preprocess the image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Could not read {img_path}")
                        continue
                        
                    # Resize image to smaller dimensions
                    img = cv2.resize(img, img_size)
                    
                    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img = clahe.apply(img) / 255.0
                    
                    batch_images.append(img)
                    batch_labels.append(idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            if batch_images:
                images.extend(batch_images)
                labels.extend(batch_labels)
                print(f"Processed batch of {len(batch_images)} images")
    
    # Convert to numpy arrays
    images_array = np.array(images)
    labels_array = np.array(labels)
    
    print(f"Final dataset shape: {images_array.shape}, labels shape: {labels_array.shape}")
    print(f"Label distribution: {np.unique(labels_array, return_counts=True)}")
    
    return images_array, labels_array, blood_group_dict, num_classes

# 2. Build CNN Model with correct number of outputs
def build_fingerprint_model(input_shape, num_classes):
    """
    Create a CNN model for fingerprint classification with correct output layer
    """
    print(f"Building model with input shape {input_shape} and {num_classes} output classes")
    
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Second convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Third convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        # Ensure correct number of output neurons
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary to verify output shape
    model.summary()
    
    return model

# 3. Create efficient data loading with tf.data
def create_dataset(image_data, label_data, batch_size=8, is_training=True):
    """Create an efficient tf.data.Dataset for training or validation"""
    
    # Add channel dimension
    image_data = np.expand_dims(image_data, -1)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_data, label_data))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
        
        # Data augmentation on the fly
        def augment(image, label):
            # Random rotation (limited)
            angle = tf.random.uniform([], -10, 10) * (np.pi / 180)
            image = tf.image.rot90(image, k=tf.cast(angle/(np.pi/2), tf.int32))
            
            # Random brightness adjustments
            image = tf.image.random_brightness(image, 0.1)
            
            # Random contrast adjustments
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            
            return image, label
        
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Optimize dataset performance
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# 4. Training the model
def train_model(dataset_path, epochs=20, batch_size=8, img_size=(128, 128)):
    """
    Train the fingerprint blood group classification model
    """
    print(f"Starting training with dataset: {dataset_path}")
    print(f"Parameters: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
    
    # Process the dataset
    X, y, blood_group_dict, num_classes = preprocess_dataset(dataset_path, img_size)
    
    # Verify that we have the correct number of classes
    unique_labels = np.unique(y)
    print(f"Dataset contains labels: {unique_labels}")
    assert len(unique_labels) == num_classes, f"Expected {num_classes} classes but found {len(unique_labels)}"
    assert max(unique_labels) < num_classes, f"Maximum label value {max(unique_labels)} should be less than {num_classes}"
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    
    # Create efficient datasets
    train_dataset = create_dataset(X_train, y_train, batch_size=batch_size, is_training=True)
    val_dataset = create_dataset(X_val, y_val, batch_size=batch_size, is_training=False)
    
    # Build model
    model = build_fingerprint_model((img_size[0], img_size[1], 1), num_classes)
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # Model checkpoint to save best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'fingerprint_blood_group_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Training
    try:
        print(f"Starting training with batch size {batch_size}...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stopping, checkpoint]
        )
    except tf.errors.ResourceExhaustedError:
        # If memory error occurs, reduce batch size and recreate datasets
        print("Memory error encountered. Reducing batch size...")
        batch_size = batch_size // 2
        train_dataset = create_dataset(X_train, y_train, batch_size=batch_size, is_training=True)
        val_dataset = create_dataset(X_val, y_val, batch_size=batch_size, is_training=False)
        
        print(f"Retrying with batch size {batch_size}...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stopping, checkpoint]
        )
    
    # Save the blood group mapping
    with open('blood_group_mapping.json', 'w') as f:
        json.dump(blood_group_dict, f)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return model, blood_group_dict

# 5. Prediction function
def predict_blood_group(image_path, model, blood_group_dict, img_size=(128, 128)):
    """
    Predict blood group from a fingerprint image
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None, 0
            
        img = cv2.resize(img, img_size)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Normalize
        img = img / 255.0
        
        # Reshape for model input
        img = img.reshape(1, img_size[0], img_size[1], 1)
        
        # Make prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return blood_group_dict[str(predicted_class)], confidence
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, 0

# Example usage
if __name__ == "__main__":
    dataset_path = "dataset_blood_group"  # Update this to your dataset path
    
    # Train the model
    model, blood_group_dict = train_model(
        dataset_path,
        epochs=20,
        batch_size=8,
        img_size=(128, 128)
    )
    
    # Save the model
    model.save('fingerprint_blood_group_model.h5')
    print("Model saved successfully!")
    
    # Optional: Test the model on a sample image
    # test_image = "path_to_test_image.bmp"
    # predicted_blood_group, confidence = predict_blood_group(test_image, model, blood_group_dict)
    # print(f"Predicted blood group: {predicted_blood_group} with {confidence*100:.2f}% confidence")