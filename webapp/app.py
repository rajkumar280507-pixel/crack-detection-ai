from flask import Flask, render_template, request, redirect, url_for, Response
import os
import sys
import uuid

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detect_crack import ImprovedCrackDetector

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize detector
detector = None
try:
    detector = ImprovedCrackDetector(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load model. {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Output paths
        processed_filename = f"proc_{filename}"
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        
        # Improved Detection Pipeline
        if detector:
            report = detector.analyze_image(filepath, processed_path)
            label = report['classification_label']
            score = report['confidence_score']
            count = report['crack_count']
        else:
            label, score, count = "Model Not Loaded", 0, 0
            report = {
                "density_percentage": 0, 
                "estimated_length_px": 0, 
                "severity_level": "N/A",
                "recommended_action": "N/A"
            }
            # Just copy original to processed if no detector
            import shutil
            shutil.copy(filepath, processed_path)

        return render_template('index.html', 
                               original_img=filename,
                               processed_img=processed_filename,
                               label=label,
                               score=f"{score:.2f}",
                               count=count,
                               report=report)

from collections import deque

def gen_frames(cam_id=0):
    camera = cv2.VideoCapture(int(cam_id))
    severity_history = deque(maxlen=10)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if detector:
                processed_frame, results = detector.analyze_frame(frame)
                severity_history.append(results['severity_level'])
                
                # Get most frequent severity in history
                stable_severity = max(set(severity_history), key=severity_history.count)
                
                # Professional overlay
                cv2.rectangle(processed_frame, (0,0), (300, 50), (15, 23, 42), -1)
                cv2.putText(processed_frame, f"STATUS: {stable_severity}", (15, 33), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                processed_frame = frame

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    cam_id = request.args.get('cam', 0)
    return Response(gen_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # host='0.0.0.0' allows access from other devices (like your phone) on the same Wi-Fi
    app.run(host='0.0.0.0', debug=True, port=5000)
