#!/usr/bin/env python3
"""
Test Current Model Predictions and Create Visualizations
Shows what the current adversarial QNN actually generates
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from adversarial_merlin_swaption import QuantumGenerator, load_validation_data
import os


def load_scaler():
    """Load the price scaler for denormalization"""
    try:
        with open('price_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except FileNotFoundError:
        print("⚠️  No price scaler found - predictions will be in normalized range [0-1]")
        return None


def denormalize_prices(normalized_prices, scaler):
    """Denormalize prices back to original scale"""
    if scaler is None:
        return normalized_prices

    # Handle both 1D and 2D arrays
    if normalized_prices.ndim == 1:
        normalized_prices = normalized_prices.reshape(-1, 1)
        denormalized = scaler.inverse_transform(normalized_prices).flatten()
    else:
        shape = normalized_prices.shape
        flat_prices = normalized_prices.reshape(-1, 1)
        denormalized = scaler.inverse_transform(flat_prices).reshape(shape)

    return denormalized


def load_best_model():
    """Load the best saved model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Look for saved models
    model_files = [f for f in os.listdir('.') if f.endswith('.pth') and 'best' in f]

    if model_files:
        model_path = model_files[0]  # Take first best model
        print(f"📂 Loading model from: {model_path}")

        generator = QuantumGenerator().to(device)
        checkpoint = torch.load(model_path, map_location=device)

        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)

        generator.eval()
        return generator, device, model_path
    else:
        print("⚠️  No best model found, using fresh generator")
        generator = QuantumGenerator().to(device)
        return generator, device, "fresh_model"


def test_predictions(generator, device, scaler, n_samples=6):
    """Test generator predictions on validation data"""
    print(f"🔍 Testing generator predictions...")

    # Load validation data
    X_val, Y_val = load_validation_data()
    X_val, Y_val = X_val.to(device), Y_val.to(device)

    print(f"   Validation data: X{X_val.shape}, Y{Y_val.shape}")

    # Select random samples
    indices = torch.randperm(len(X_val))[:n_samples]
    X_test = X_val[indices]
    Y_test = Y_val[indices]

    # Generate predictions
    with torch.no_grad():
        Y_pred = generator(X_test)

    # Move to CPU for plotting
    X_test = X_test.cpu().numpy()
    Y_test = Y_test.cpu().numpy()
    Y_pred = Y_pred.cpu().numpy()

    return X_test, Y_test, Y_pred


def plot_predictions(X_test, Y_test, Y_pred, scaler, model_name):
    """Create comprehensive prediction plots"""
    print(f"📊 Creating prediction plots...")

    n_samples = len(X_test)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(min(n_samples, 6)):
        ax = axes[i]

        # Extract data
        input_prices = X_test[i, :, 0]  # 60 input prices (normalized)
        true_targets = Y_test[i]        # 14 true future prices (normalized)
        pred_targets = Y_pred[i]        # 14 predicted future prices (normalized)

        # Get tenor/maturity info
        tenor = X_test[i, 0, 1] * 30.0    # Denormalize tenor
        maturity = X_test[i, 0, 2] * 30.0  # Denormalize maturity

        # Denormalize prices if scaler available
        if scaler:
            input_prices_orig = denormalize_prices(input_prices, scaler)
            true_targets_orig = denormalize_prices(true_targets, scaler)
            pred_targets_orig = denormalize_prices(pred_targets, scaler)
            ylabel = "Swaption Price (Original Scale)"
        else:
            input_prices_orig = input_prices
            true_targets_orig = true_targets
            pred_targets_orig = pred_targets
            ylabel = "Normalized Price [0-1]"

        # Create timeline
        input_days = np.arange(60)
        target_days = np.arange(60, 74)

        # Plot input sequence
        ax.plot(input_days, input_prices_orig, 'b-', linewidth=2,
                label='Historical (60 days)', alpha=0.8)

        # Plot true vs predicted targets
        ax.plot(target_days, true_targets_orig, 'g-', linewidth=2,
                label='True Future (14 days)', marker='o', markersize=4)
        ax.plot(target_days, pred_targets_orig, 'r--', linewidth=2,
                label='Predicted Future', marker='s', markersize=4)

        # Vertical line separator
        ax.axvline(x=59.5, color='gray', linestyle=':', alpha=0.7)

        # Calculate metrics
        mse = np.mean((true_targets - pred_targets) ** 2)
        true_vol = np.std(true_targets)
        pred_vol = np.std(pred_targets)
        vol_ratio = pred_vol / true_vol if true_vol > 0 else 0

        # Add correlation
        correlation = np.corrcoef(true_targets, pred_targets)[0, 1]

        # Formatting
        ax.set_title(f'Sample {i+1}: T{tenor:.0f}Y, M{maturity:.2f}Y\n'
                    f'Vol Ratio: {vol_ratio:.3f}, Corr: {correlation:.3f}')
        ax.set_xlabel('Days')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add statistics box
        stats_text = f'MSE: {mse:.5f}\nTrue Vol: {true_vol:.4f}\nPred Vol: {pred_vol:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'Adversarial QNN Predictions - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig('model_predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Prediction plots saved as 'model_predictions_analysis.png'")


def analyze_volatility_distribution(Y_test, Y_pred):
    """Analyze volatility distribution of real vs predicted"""
    print(f"📊 Analyzing volatility distributions...")

    # Calculate volatilities
    true_vols = np.std(Y_test, axis=1)
    pred_vols = np.std(Y_pred, axis=1)

    # Statistics
    print(f"\n📈 VOLATILITY ANALYSIS:")
    print(f"   True volatility:      {true_vols.mean():.6f} ± {true_vols.std():.6f}")
    print(f"   Predicted volatility: {pred_vols.mean():.6f} ± {pred_vols.std():.6f}")
    print(f"   Volatility ratio:     {pred_vols.mean() / true_vols.mean():.3f}")
    print(f"   Correlation:          {np.corrcoef(true_vols, pred_vols)[0,1]:.3f}")

    # Plot volatility distributions
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(true_vols, bins=20, alpha=0.7, label='True Volatilities', color='green')
    plt.hist(pred_vols, bins=20, alpha=0.7, label='Predicted Volatilities', color='red')
    plt.xlabel('Volatility')
    plt.ylabel('Frequency')
    plt.title('Volatility Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(true_vols, pred_vols, alpha=0.6, s=20)
    plt.plot([true_vols.min(), true_vols.max()],
             [true_vols.min(), true_vols.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('True Volatility')
    plt.ylabel('Predicted Volatility')
    plt.title('Volatility Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('volatility_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Volatility analysis saved as 'volatility_analysis.png'")


def main():
    """Main analysis function"""
    print("🔍 ADVERSARIAL QNN MODEL ANALYSIS")
    print("="*50)

    # Load components
    scaler = load_scaler()
    generator, device, model_name = load_best_model()

    print(f"🖥️  Device: {device}")
    print(f"📊 Model: {model_name}")

    # Test predictions
    X_test, Y_test, Y_pred = test_predictions(generator, device, scaler, n_samples=6)

    # Create plots
    plot_predictions(X_test, Y_test, Y_pred, scaler, model_name)

    # Analyze volatilities
    analyze_volatility_distribution(Y_test, Y_pred)

    # Overall assessment
    true_vol_mean = np.std(Y_test, axis=1).mean()
    pred_vol_mean = np.std(Y_pred, axis=1).mean()
    vol_ratio = pred_vol_mean / true_vol_mean

    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"   Volatility Ratio: {vol_ratio:.3f}")

    if vol_ratio > 0.8:
        print(f"   ✅ EXCELLENT: High volatility preservation!")
    elif vol_ratio > 0.5:
        print(f"   ✅ GOOD: Decent volatility preservation")
    elif vol_ratio > 0.3:
        print(f"   ⚠️  MODERATE: Some volatility loss")
    else:
        print(f"   ❌ POOR: Significant volatility loss (regression to mean)")

    print(f"\n📊 Next steps based on results:")
    if vol_ratio < 0.5:
        print(f"   - Increase adversarial loss weight")
        print(f"   - Strengthen discriminator")
        print(f"   - Reduce reconstruction loss weight")
    else:
        print(f"   - Current model shows good performance!")
        print(f"   - Consider fine-tuning hyperparameters")


if __name__ == "__main__":
    main()