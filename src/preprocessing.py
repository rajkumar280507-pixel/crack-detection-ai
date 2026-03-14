import cv2
import numpy as np

def enhance_image(image):
    """
    Apply preprocessing to enhance crack features and reduce texture noise.
    Uses Bilateral Filter to preserve edges while smoothing texture.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. Edge-Preserving Smoothing: Bilateral Filter
    # Removes fine surface texture but keeps sharp crack edges
    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Contrast Enhancement: CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)

    return enhanced

def get_binary_mask(enhanced_image):
    """
    Segment crack regions using adaptive thresholding and tuned edge detection.
    """
    # 3. Adaptive Thresholding with larger block size for robustness
    thresh = cv2.adaptiveThreshold(
        enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 5
    )

    # 4. Tuned Canny Edge Detection
    edges = cv2.Canny(enhanced_image, 30, 90)

    # Combine for robust segmentation
    combined_mask = cv2.bitwise_or(thresh, edges)

    # 5. Morphological Closing to merge fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return closed
