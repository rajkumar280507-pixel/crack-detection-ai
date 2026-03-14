import tensorflow as tf
import cv2
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import enhance_image, get_binary_mask
from src.severity_analysis import analyze_crack_severity
from src.visualization import visualize_results

class ImprovedCrackDetector:
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
        
    def predict_classification(self, image_path):
        """
        AI-based classification (Crack vs No-Crack).
        """
        if self.model is None:
            return "N/A", 0.0
            
        # Load and resize for model
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
        input_data = np.expand_dims(img_resized, axis=0)
        
        prediction = self.model.predict(input_data)[0][0]
        label = "Crack" if prediction > 0.5 else "Non-Crack"
        return label, float(prediction)

    def analyze_image(self, image_path, output_path):
        """
        Complete pipeline: Preprocessing -> Analysis -> Visualization
        """
        # 1. Load Image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError("Image not found")

        # 2. AI Classification
        label, score = self.predict_classification(image_path)

        # 3. Image Processing (Localization)
        # Even if classification is "Non-Crack", we process to handle edge cases
        enhanced = enhance_image(original)
        mask = get_binary_mask(enhanced)
        
        # 4. Severity Analysis
        results = analyze_crack_severity(mask)
        results['classification_label'] = label
        results['confidence_score'] = score

        # 5. Visualization
        visualized = visualize_results(original, results)
        cv2.imwrite(output_path, visualized)
        
        return results
    def analyze_frame(self, frame):
        """
        Processes a single frame for real-time video feed.
        """
        if frame is None:
            return None, None

        # 1. Image Processing
        enhanced = enhance_image(frame)
        mask = get_binary_mask(enhanced)
        
        # 2. Severity Analysis
        results = analyze_crack_severity(mask)
        
        # 3. Visualization
        visualized = visualize_results(frame, results)
        return visualized, results

if __name__ == "__main__":
    # Example test
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'model.h5')
    detector = ImprovedCrackDetector(model_path)
    # results = detector.analyze_image('test.jpg', 'output.jpg')
