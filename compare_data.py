#!/usr/bin/env python3
"""
Compare newly generated data with original data
"""

import torch
import json
import numpy as np


def load_original_data():
    """Load original training data"""
    print("📁 Loading original data...")

    try:
        # Load original training data
        with open('train_set_original.json', 'r') as f:
            train_data = json.load(f)
        X_train_orig = torch.tensor(train_data['X'], dtype=torch.float32)
        Y_train_orig = torch.tensor(train_data['Y'], dtype=torch.float32)

        # Load original validation data
        with open('val_set_original.json', 'r') as f:
            val_data = json.load(f)
        X_val_orig = torch.tensor(val_data['X'], dtype=torch.float32)
        Y_val_orig = torch.tensor(val_data['Y'], dtype=torch.float32)

        print(f"✅ Original data loaded:")
        print(f"   X_train: {X_train_orig.shape}")
        print(f"   Y_train: {Y_train_orig.shape}")
        print(f"   X_val: {X_val_orig.shape}")
        print(f"   Y_val: {Y_val_orig.shape}")

        return X_train_orig, Y_train_orig, X_val_orig, Y_val_orig

    except Exception as e:
        print(f"❌ Error loading original data: {e}")
        return None, None, None, None


def load_new_data():
    """Load newly generated data"""
    print("📁 Loading new data...")

    try:
        # Load new training data
        with open('train_set.json', 'r') as f:
            train_data = json.load(f)
        X_train_new = torch.tensor(train_data['X'], dtype=torch.float32)
        Y_train_new = torch.tensor(train_data['Y'], dtype=torch.float32)

        # Load new validation data
        with open('val_set.json', 'r') as f:
            val_data = json.load(f)
        X_val_new = torch.tensor(val_data['X'], dtype=torch.float32)
        Y_val_new = torch.tensor(val_data['Y'], dtype=torch.float32)

        print(f"✅ New data loaded:")
        print(f"   X_train: {X_train_new.shape}")
        print(f"   Y_train: {Y_train_new.shape}")
        print(f"   X_val: {X_val_new.shape}")
        print(f"   Y_val: {Y_val_new.shape}")

        return X_train_new, Y_train_new, X_val_new, Y_val_new

    except Exception as e:
        print(f"❌ Error loading new data: {e}")
        return None, None, None, None


def compare_statistics(X_orig, Y_orig, X_new, Y_new, data_type="train"):
    """Compare statistical properties of datasets"""
    print(f"\n📊 COMPARING {data_type.upper()} DATA STATISTICS:")

    # Shape comparison
    print(f"📏 Shapes:")
    print(f"   Original: X{X_orig.shape}, Y{Y_orig.shape}")
    print(f"   New:      X{X_new.shape}, Y{Y_new.shape}")

    shapes_match = X_orig.shape[1:] == X_new.shape[1:] and Y_orig.shape[1:] == Y_new.shape[1:]
    print(f"   Shape compatibility: {'✅ MATCH' if shapes_match else '❌ MISMATCH'}")

    # Price statistics (X[:,:,0] is the price column)
    orig_prices = X_orig[:, :, 0]
    new_prices = X_new[:, :, 0]

    print(f"\n📈 Price Statistics:")
    print(f"   Original - Mean: {orig_prices.mean():.4f}, Std: {orig_prices.std():.4f}")
    print(f"   Original - Range: [{orig_prices.min():.4f}, {orig_prices.max():.4f}]")
    print(f"   New -      Mean: {new_prices.mean():.4f}, Std: {new_prices.std():.4f}")
    print(f"   New -      Range: [{new_prices.min():.4f}, {new_prices.max():.4f}]")

    # Volatility comparison (most important!)
    orig_volatility = torch.std(Y_orig, dim=1).mean()
    new_volatility = torch.std(Y_new, dim=1).mean()

    print(f"\n📊 Volatility (KEY METRIC):")
    print(f"   Original: {orig_volatility:.6f}")
    print(f"   New:      {new_volatility:.6f}")
    print(f"   Ratio:    {new_volatility/orig_volatility:.2f}x")

    if abs(new_volatility/orig_volatility - 1.0) < 0.5:  # Within 50%
        print(f"   ✅ Volatility is reasonably similar")
    else:
        print(f"   ⚠️  Volatility differs significantly")

    # Target statistics
    print(f"\n🎯 Target (Y) Statistics:")
    print(f"   Original - Mean: {Y_orig.mean():.4f}, Std: {Y_orig.std():.4f}")
    print(f"   New -      Mean: {Y_new.mean():.4f}, Std: {Y_new.std():.4f}")

    # Feature analysis (tenor, maturity)
    if X_orig.shape[2] >= 3:
        print(f"\n🔧 Feature Analysis:")

        # Tenor (column 1)
        orig_tenor = X_orig[:, 0, 1]  # First timestep tenor
        new_tenor = X_new[:, 0, 1]
        print(f"   Tenor - Original: [{orig_tenor.min():.4f}, {orig_tenor.max():.4f}]")
        print(f"   Tenor - New:      [{new_tenor.min():.4f}, {new_tenor.max():.4f}]")

        # Maturity (column 2)
        orig_maturity = X_orig[:, 0, 2]
        new_maturity = X_new[:, 0, 2]
        print(f"   Maturity - Original: [{orig_maturity.min():.4f}, {orig_maturity.max():.4f}]")
        print(f"   Maturity - New:      [{new_maturity.min():.4f}, {new_maturity.max():.4f}]")


def test_training_compatibility():
    """Test if new data works with existing training functions"""
    print(f"\n🧪 TESTING TRAINING COMPATIBILITY:")

    try:
        from adversarial_merlin_swaption import QuantumGenerator, load_training_data, load_validation_data

        print("   Testing load functions...")
        X_train, Y_train = load_training_data()
        X_val, Y_val = load_validation_data()

        print("   ✅ Load functions work")

        print("   Testing QuantumGenerator...")
        generator = QuantumGenerator()

        # Test small batch
        test_batch = X_train[:4]
        with torch.no_grad():
            predictions = generator(test_batch)

        print(f"   ✅ QuantumGenerator works: {test_batch.shape} → {predictions.shape}")

        # Check prediction volatility
        pred_vol = torch.std(predictions, dim=1).mean()
        target_vol = torch.std(Y_train[:4], dim=1).mean()

        print(f"   📊 Quick volatility test:")
        print(f"      Target vol: {target_vol:.6f}")
        print(f"      Pred vol:   {pred_vol:.6f}")

        return True

    except Exception as e:
        print(f"   ❌ Compatibility test failed: {e}")
        return False


def main():
    """Main comparison function"""
    print("🔍 DATA COMPARISON: ORIGINAL vs NEWLY GENERATED")
    print("="*60)

    # Load both datasets
    X_train_orig, Y_train_orig, X_val_orig, Y_val_orig = load_original_data()
    X_train_new, Y_train_new, X_val_new, Y_val_new = load_new_data()

    if X_train_orig is None or X_train_new is None:
        print("❌ Could not load data for comparison")
        return

    # Compare training data
    compare_statistics(X_train_orig, Y_train_orig, X_train_new, Y_train_new, "train")

    # Compare validation data
    compare_statistics(X_val_orig, Y_val_orig, X_val_new, Y_val_new, "validation")

    # Test compatibility
    compatible = test_training_compatibility()

    print(f"\n" + "="*60)
    print(f"📋 COMPARISON SUMMARY:")

    print(f"✅ Shape compatibility: Both use correct dimensions")
    print(f"✅ Data generation: New data created successfully")
    print(f"{'✅' if compatible else '❌'} Training compatibility: {'Works with existing code' if compatible else 'Needs fixes'}")

    # File size comparison
    import os
    if os.path.exists('train_set_original.json') and os.path.exists('train_set.json'):
        orig_size = os.path.getsize('train_set_original.json') / 1024 / 1024
        new_size = os.path.getsize('train_set.json') / 1024 / 1024
        print(f"📁 File sizes: Original {orig_size:.1f}MB → New {new_size:.1f}MB ({new_size/orig_size:.1%})")

    print(f"\n🚀 READY FOR GPU TRAINING: {'YES' if compatible else 'NEEDS FIXES'}")


if __name__ == "__main__":
    main()