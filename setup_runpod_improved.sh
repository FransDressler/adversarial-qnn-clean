#!/bin/bash
echo "🚀 IMPROVED RUNPOD SETUP - OPTION A STRATEGY"
echo "============================================="

echo "📦 Installing required packages..."
pip install torch numpy matplotlib scikit-learn

echo "📦 Installing Quandela (from GitHub)..."
pip install git+https://github.com/Quandela/perceval.git || echo "⚠️  Quandela install failed, continuing..."

echo "📁 Generating proper training data from CSV..."
if [ -f "create_proper_data.py" ]; then
    echo "✅ Using proper data generator with MinMaxScaler"
    python create_proper_data.py
else
    echo "⚠️  Using fallback data generator"
    python create_real_data.py || python create_mini_data.py
fi

echo "🧪 Testing data loading..."
python3 -c "
import json
import numpy as np
try:
    with open('train_set.json', 'r') as f:
        data = json.load(f)
    X = np.array(data['X'])
    Y = np.array(data['Y'])
    print(f'✅ Training data loaded: X={X.shape}, Y={Y.shape}')

    with open('val_set.json', 'r') as f:
        data = json.load(f)
    X_val = np.array(data['X'])
    Y_val = np.array(data['Y'])
    print(f'✅ Validation data loaded: X={X_val.shape}, Y={Y_val.shape}')

    vol = np.std(Y, axis=1).mean()
    print(f'📊 Average volatility: {vol:.6f}')

    if hasattr('price_scaler.pkl', '__file__'):
        print(f'✅ Price scaler available for denormalization')

    print(f'🎯 Ready for IMPROVED training with Option A strategy!')
except Exception as e:
    print(f'❌ Data loading failed: {e}')
"

echo "✅ IMPROVED RunPod setup complete!"
echo "🚀 Ready to run IMPROVED training:"
echo ""
echo "🎯 IMPROVED STRATEGY:"
echo "   • Stronger adversarial loss (4.0 vs 2.0)"
echo "   • Weaker reconstruction loss (1.0 vs 3.0)"
echo "   • Higher discriminator LR (2e-3 vs 5e-4)"
echo "   • Multiple discriminator updates per epoch"
echo "   • Learning rate scheduling"
echo "   • Better early stopping"
echo ""
echo "📋 Available training options:"
echo "   python improved_resnet_training.py  # 🎯 RECOMMENDED: Option A strategy"
echo "   python optimized_resnet_training.py # 🔄 Original training (for comparison)"
echo ""
echo "📊 Files generated:"
ls -la *.py *.json *.pkl 2>/dev/null | head -10