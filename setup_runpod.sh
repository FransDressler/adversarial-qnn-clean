#!/bin/bash
echo "🚀 RUNPOD SETUP FOR ADVERSARIAL QNN TRAINING"
echo "============================================="

echo "📦 Installing required packages..."
pip install torch numpy matplotlib quandela

echo "📁 Generating training data from CSV..."
python create_real_data.py

echo "🧪 Testing data loading..."
python compare_data.py

echo "✅ RunPod setup complete!"
echo "🎯 Ready to run: python optimized_resnet_training.py"