#!/bin/bash
echo "🚀 RUNPOD SETUP FOR ADVERSARIAL QNN TRAINING"
echo "============================================="

echo "📦 Installing required packages..."
pip install torch numpy matplotlib scikit-learn

echo "📦 Installing Quandela (from GitHub)..."
pip install git+https://github.com/Quandela/perceval.git
# Note: If Perceval fails, we can continue without it for testing

echo "📁 Generating proper training data from CSV..."
if [ -f "Daten.csv" ]; then
    python create_proper_data.py
elif [ -f "create_proper_data.py" ]; then
    echo "✅ Using proper data generator"
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
except Exception as e:
    print(f'❌ Data loading failed: {e}')
"

echo "✅ RunPod setup complete!"
echo "🎯 Ready to run: python optimized_resnet_training.py"
echo ""
echo "📋 Available files:"
ls -la *.py *.json 2>/dev/null || echo "No data files yet"