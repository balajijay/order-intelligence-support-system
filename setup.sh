#!/bin/bash
# Exit immediately if any command encounters an error
set -e

echo "📦 Creating Python virtual environment..."
python3 -m venv venv

echo "🔌 Activating virtual environment..."
source venv/bin/activate

echo "🔄 Upgrading internal package manager..."
pip install --upgrade pip

echo "🧬 Installing Scientific Data Frameworks..."
pip install numpy pandas scikit-learn

echo "🧠 Installing Deep Learning Neural Networks..."
pip install torch torchvision matplotlib

echo "⚙️  Running pipeline training engine (Risk Model)..."
python3 return_risk.py

echo "🧪 Testing real-time transaction checking script..."
python3 predict_order.py

echo "🖼️  Downloading Fashion MNIST & training ResNet-18..."
python3 image_classify.py

echo "✅ System environment fully compiled and ready!"
