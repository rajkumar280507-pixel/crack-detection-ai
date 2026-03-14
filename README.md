# Crack Detection AI – Intelligent Structural Inspection System

## 🏗️ Project Overview
**Crack Detection AI** is a professional-grade AI inspection platform designed for automated structural health monitoring. It leverages advanced Computer Vision (OpenCV) and Deep Learning (TensorFlow) to detect, segment, and analyze cracks in civil infrastructure like bridges, buildings, pavements, and tunnels.

Unlike basic classifiers, **Crack Detection AI** provides quantitative engineering metrics, including topological crack length, approximate width, and density-based hazard assessment.

## 🌟 Key Features
-   🔍 **Precision Detection:** Distinguishes structural cracks from surface pores, stains, and construction joints.
-   📐 **Quantitative Metrics:** Estimates crack length (skeleton-based) and average width (geometric-based).
-   📉 **Hazard Assessment:** Classifies severity into Mild, Moderate, Severe, or Critical based on ACI/Eurocode standards.
-   🎨 **Engineering Dashboard:** Side-by-side visualization with detection heatmaps and centerline skeletons.
-   📜 **Recommendation Engine:** Generates actionable repair advice (e.g., Monitoring, Epoxy Injection, Structural Patching).

## 🛠️ Technical Stack
-   **Backend:** Python 3.10+, Flask
-   **AI/ML:** TensorFlow/Keras (CNN), scikit-image (Skeletonization)
-   **Computer Vision:** OpenCV (Bilateral Filtering, Adaptive Thresholding, Hough Line Transform)
-   **Frontend:** HTML5, CSS3 (Modern Dashboard UI), Lucide Icons

## 📂 Project Structure
```text
crack_detection_ai/
│
├── src/
│   ├── preprocessing.py    # Texture suppression & adaptive segmentation
│   ├── detect_crack.py     # Inference pipeline & classification
│   ├── severity_analysis.py # Geometric metrics & recommendation engine
│   └── visualization.py    # Professional heatmap & skeleton overlays
├── webapp/                 # Civils.AI Flask Dashboard
├── models/                 # Pre-trained CNN (model.h5)
└── requirements.txt        # Managed dependencies
```

## 🚀 Quick Start
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Start the Dashboard:**
    ```bash
    python webapp/app.py
    ```
3.  **Access Platform:** Open `http://127.0.0.1:5000`

## 📊 Evaluation Logic
Civils.AI uses a multi-stage validation pipeline:
1.  **Stage 1:** AI Classification (Crack vs No-Crack).
2.  **Stage 2:** Bilateral feature enhancement and adaptive thresholding.
3.  **Stage 3:** Skeletonization for accurate topological path extraction.
4.  **Stage 4:** Mapping geometric data to Civil Engineering maintenance protocols.

## ⚖️ License
MIT License - Professional Use Recommended.
