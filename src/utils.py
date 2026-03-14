import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess image for model prediction.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, target_size)
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0)

def get_crack_mask(image_path):
    """
    Process image to extract crack contours using edge detection.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Morphological operations to close gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    return dilated, image

def save_detection_result(image, mask, output_path):
    """
    Overlay mask on original image and save.
    """
    # Create a colored mask (e.g., red for cracks)
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 0, 255] # Red in BGR
    
    # Blend original image with mask
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    
    # Draw bounding boxes for contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 20: # Filter noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    cv2.imwrite(output_path, overlay)
    return len(contours)
