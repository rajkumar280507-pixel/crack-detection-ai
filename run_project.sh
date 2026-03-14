#!/bin/bash
echo "Setting up Structural Crack Detection System..."
pip install -r requirements.txt
if [ ! -f "models/model.h5" ]; then
    echo "Model not found. Training model now..."
    python3 src/train_model.py
fi
echo "Starting Web Application..."
python3 webapp/app.py
