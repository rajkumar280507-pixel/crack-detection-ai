import cv2
import numpy as np

def visualize_results(image, analysis_results):
    """
    Generate professional visualization with skeleton path and major crack highlighting.
    """
    output = image.copy()
    contours = analysis_results.get('significant_contours', [])
    skeleton = analysis_results.get('skeleton')
    
    # 1. Overlay Skeleton (Single-pixel white path)
    if skeleton is not None:
        # Convert skeleton to BGR mask
        skeleton_bgr = np.zeros_like(image)
        skeleton_bgr[skeleton] = (255, 255, 255) # White path
        # Overlay with high visibility
        output = cv2.addWeighted(output, 1.0, skeleton_bgr, 1.0, 0)

    # 2. Highlight Major Crack Bounding Boxes
    for cnt in contours:
        # Green bounding box for the overall region
        x, y, w, h = cv2.boundingRect(cnt)
        # Only box major regions
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Red semi-transparent fill for the crack area
        overlay = output.copy()
        cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)

    # 3. Add Status Text
    severity = analysis_results.get('severity_level', 'N/A')
    color = (0, 0, 255) if severity == "Severe" else (0, 255, 255)
    cv2.putText(output, f"CRACK STATUS: {severity}", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

    return output
