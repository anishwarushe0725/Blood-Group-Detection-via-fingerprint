import numpy as np
import cv2
import tensorflow as tf

def preprocess_image(image_path):
    """
    Preprocess fingerprint image using the same steps as in training
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Resize image to 128x128
    img = cv2.resize(img, (128, 128))
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This was used during training
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Normalize pixel values (same as in training)
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Add channel dimension for grayscale
    img = np.expand_dims(img, axis=-1)
    
    print(f"Preprocessed image shape: {img.shape}")
    
    return img

def predict_blood_group(model, image_path):
    """
    Predict blood group from a fingerprint image
    """
    try:
        # Preprocess the image exactly as in training
        processed_img = preprocess_image(image_path)
        
        # Make prediction
        prediction = model.predict(processed_img)
        
        # Convert prediction to blood group label
        blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        
        # Get the index of the highest probability
        predicted_index = np.argmax(prediction[0])
        
        return blood_groups[predicted_index]
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Return a default value or re-raise the exception
        return "Error: Could not predict blood group"


