@echo off
echo Setting up Structural Crack Detection System...
pip install -r requirements.txt
if not exist "models\model.h5" (
    echo Model not found. Training model now...
    python src/train_model.py
)
echo Starting Web Application...
python webapp/app.py
pause
