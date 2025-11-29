#!/usr/bin/env python3
"""
Create Proper Training Data from Swaption CSV
- Each column = one tenor/maturity combination
- Split into 74-day sequences (60 input + 14 output)
- X: [60 days, 3 features] = [prices, tenor, maturity]
- Y: [14 days] = future prices
"""

import numpy as np
import json
import csv
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import MinMaxScaler
import pickle


def load_swaption_csv(csv_path="../Daten.csv"):
    """Load swaption data and parse tenor/maturity from headers"""
    print(f"📁 Loading swaption CSV from {csv_path}...")

    dates = []
    prices = []
    tenors = []
    maturities = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)  # Get header row

        # Parse tenor/maturity from column headers
        for col_name in header[1:]:  # Skip Date column
            # Extract tenor and maturity from "Tenor : X; Maturity : Y" format
            match = re.search(r'Tenor : (\d+); Maturity : ([\d.]+)', col_name)
            if match:
                tenor = int(match.group(1))
                maturity = float(match.group(2))
                tenors.append(tenor)
                maturities.append(maturity)

        print(f"   Found {len(tenors)} tenor/maturity combinations")
        print(f"   Tenor range: {min(tenors)} - {max(tenors)} years")
        print(f"   Maturity range: {min(maturities):.2f} - {max(maturities):.2f} years")

        # Read price data
        for row in reader:
            if len(row) < 2:
                continue
            date_str = row[0]
            price_values = []

            for val in row[1:]:
                if val.strip():
                    # Convert German decimal format to float
                    price_values.append(float(val.replace(',', '.')))

            if len(price_values) == len(tenors):
                dates.append(date_str)
                prices.append(price_values)

    prices_array = np.array(prices, dtype=np.float32)
    tenors_array = np.array(tenors, dtype=np.float32)
    maturities_array = np.array(maturities, dtype=np.float32)

    print(f"✅ CSV loaded:")
    print(f"   Dates: {len(dates)} (from {dates[0]} to {dates[-1]})")
    print(f"   Price matrix: {prices_array.shape}")
    print(f"   Price range: [{prices_array.min():.4f}, {prices_array.max():.4f}]")

    return dates, prices_array, tenors_array, maturities_array


def normalize_prices(prices):
    """
    Normalize swaption prices using MinMaxScaler
    Returns: normalized_prices, fitted_scaler
    """
    print(f"🔧 Normalizing prices with MinMaxScaler...")
    print(f"   Original price range: [{prices.min():.4f}, {prices.max():.4f}]")

    # Fit scaler on all price data
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_reshaped = prices.reshape(-1, 1)  # Flatten for scaler
    normalized_prices_flat = scaler.fit_transform(prices_reshaped)
    normalized_prices = normalized_prices_flat.reshape(prices.shape)

    print(f"   Normalized range: [{normalized_prices.min():.4f}, {normalized_prices.max():.4f}]")

    # Save scaler for later use
    with open('price_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ Scaler saved as 'price_scaler.pkl'")

    return normalized_prices, scaler


def denormalize_prices(normalized_prices, scaler_path='price_scaler.pkl'):
    """
    Denormalize prices back to original scale
    """
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Handle both 1D and 2D arrays
    if normalized_prices.ndim == 1:
        normalized_prices = normalized_prices.reshape(-1, 1)
        denormalized = scaler.inverse_transform(normalized_prices).flatten()
    else:
        shape = normalized_prices.shape
        flat_prices = normalized_prices.reshape(-1, 1)
        denormalized = scaler.inverse_transform(flat_prices).reshape(shape)

    return denormalized


def create_training_sequences(dates, prices, tenors, maturities, input_days=60, output_days=14):
    """
    Create training sequences from swaption price data

    For each tenor/maturity column:
    - Create sliding windows of 74 days total
    - X: [input_days, 3] = [price_sequence, tenor, maturity]
    - Y: [output_days] = future_prices (normalized)
    """

    n_dates, n_columns = prices.shape
    sequence_length = input_days + output_days

    print(f"🔄 Creating training sequences...")
    print(f"   Input days: {input_days}")
    print(f"   Output days: {output_days}")
    print(f"   Total sequence length needed: {sequence_length}")
    print(f"   Available dates: {n_dates}")

    if n_dates < sequence_length:
        raise ValueError(f"Not enough data! Need {sequence_length} days, have {n_dates}")

    X_sequences = []
    Y_sequences = []
    metadata = []  # Store which column and start date

    # For each tenor/maturity combination (column)
    for col_idx in range(n_columns):
        column_prices = prices[:, col_idx]
        tenor = tenors[col_idx]
        maturity = maturities[col_idx]

        # Create sliding windows
        max_start = n_dates - sequence_length + 1

        for start_idx in range(max_start):
            # Input sequence (60 days)
            input_prices = column_prices[start_idx:start_idx + input_days]

            # Output sequence (14 days after input)
            output_prices = column_prices[start_idx + input_days:start_idx + sequence_length]

            # Create 3D input: [60, 3] = [normalized_prices, tenor, maturity]
            X_sample = np.zeros((input_days, 3), dtype=np.float32)
            X_sample[:, 0] = input_prices  # Normalized price history
            X_sample[:, 1] = tenor / 30.0  # Normalized tenor (max 30 years)
            X_sample[:, 2] = maturity / 30.0  # Normalized maturity (max 30 years)

            X_sequences.append(X_sample)
            Y_sequences.append(output_prices)
            metadata.append({
                'column': col_idx,
                'tenor': tenor,
                'maturity': maturity,
                'start_date': dates[start_idx],
                'start_idx': start_idx
            })

    X_data = np.array(X_sequences, dtype=np.float32)
    Y_data = np.array(Y_sequences, dtype=np.float32)

    print(f"✅ Sequences created:")
    print(f"   Total sequences: {len(X_sequences)}")
    print(f"   X shape: {X_data.shape} (samples, days, features)")
    print(f"   Y shape: {Y_data.shape} (samples, future_days)")
    print(f"   Sequences per column: {max_start}")

    return X_data, Y_data, metadata


def plot_sample_sequences(X_data, Y_data, metadata, n_samples=4):
    """Plot random sample sequences to visualize the data (normalized and denormalized)"""
    print(f"📊 Plotting {n_samples} random sample sequences...")

    # Select random samples
    total_samples = len(X_data)
    sample_indices = np.random.choice(total_samples, n_samples, replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows: normalized and denormalized

    for i, sample_idx in enumerate(sample_indices):
        # Normalized plot
        ax_norm = axes[0, i]
        # Denormalized plot
        ax_denorm = axes[1, i]

        # Get sample data
        X_sample = X_data[sample_idx]
        Y_sample = Y_data[sample_idx]
        meta = metadata[sample_idx]

        # Extract normalized prices
        input_prices_norm = X_sample[:, 0]  # 60 days input (normalized)
        output_prices_norm = Y_sample  # 14 days output (normalized)

        # Denormalize prices
        input_prices_denorm = denormalize_prices(input_prices_norm)
        output_prices_denorm = denormalize_prices(output_prices_norm)

        # Create timeline
        input_days = np.arange(len(input_prices_norm))
        output_days = np.arange(len(input_prices_norm), len(input_prices_norm) + len(output_prices_norm))

        # Plot normalized data
        ax_norm.plot(input_days, input_prices_norm, 'b-', linewidth=2, label='Input (60 days)')
        ax_norm.plot(output_days, output_prices_norm, 'r-', linewidth=2, label='Target (14 days)')
        ax_norm.axvline(x=len(input_prices_norm)-1, color='gray', linestyle='--', alpha=0.7)
        ax_norm.set_title(f'Normalized - T{meta["tenor"]}Y, M{meta["maturity"]:.2f}Y')
        ax_norm.set_xlabel('Days')
        ax_norm.set_ylabel('Normalized Price [0-1]')
        ax_norm.legend()
        ax_norm.grid(True, alpha=0.3)

        # Plot denormalized data
        ax_denorm.plot(input_days, input_prices_denorm, 'b-', linewidth=2, label='Input (60 days)')
        ax_denorm.plot(output_days, output_prices_denorm, 'r-', linewidth=2, label='Target (14 days)')
        ax_denorm.axvline(x=len(input_prices_denorm)-1, color='gray', linestyle='--', alpha=0.7)
        ax_denorm.set_title(f'Original Scale - Sample {sample_idx}')
        ax_denorm.set_xlabel('Days')
        ax_denorm.set_ylabel('Swaption Price')
        ax_denorm.legend()
        ax_denorm.grid(True, alpha=0.3)

        # Statistics
        volatility = np.std(output_prices_norm)
        ax_norm.text(0.02, 0.98, f'Vol: {volatility:.4f}',
                    transform=ax_norm.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.suptitle('Swaption Training Sequences: 60 Input → 14 Target Days', fontsize=16)
    plt.tight_layout()
    plt.savefig('swaption_sequences_normalized.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Normalized sample plots saved as 'swaption_sequences_normalized.png'")


def analyze_data_statistics(X_data, Y_data, metadata):
    """Analyze the generated dataset"""
    print(f"\n📊 DATA STATISTICS:")

    # Shape analysis
    print(f"📏 Shapes:")
    print(f"   X: {X_data.shape} (samples, input_days, features)")
    print(f"   Y: {Y_data.shape} (samples, output_days)")

    # Price analysis
    input_prices = X_data[:, :, 0]  # All input prices
    output_prices = Y_data  # All output prices

    print(f"\n📈 Price Analysis:")
    print(f"   Input price range: [{input_prices.min():.4f}, {input_prices.max():.4f}]")
    print(f"   Output price range: [{output_prices.min():.4f}, {output_prices.max():.4f}]")
    print(f"   Input mean: {input_prices.mean():.4f}, std: {input_prices.std():.4f}")
    print(f"   Output mean: {output_prices.mean():.4f}, std: {output_prices.std():.4f}")

    # Volatility analysis (std of 14-day sequences)
    sequence_volatilities = np.std(Y_data, axis=1)
    print(f"\n📊 Volatility Analysis:")
    print(f"   Mean volatility: {sequence_volatilities.mean():.6f}")
    print(f"   Volatility std: {sequence_volatilities.std():.6f}")
    print(f"   Volatility range: [{sequence_volatilities.min():.6f}, {sequence_volatilities.max():.6f}]")

    # Tenor/Maturity distribution
    tenors = [meta['tenor'] for meta in metadata]
    maturities = [meta['maturity'] for meta in metadata]

    unique_tenors = sorted(set(tenors))
    unique_maturities = sorted(set(maturities))

    print(f"\n🔧 Feature Distribution:")
    print(f"   Unique tenors: {unique_tenors}")
    print(f"   Unique maturities: {unique_maturities}")
    print(f"   Tenor distribution: {np.bincount([unique_tenors.index(t) for t in tenors])}")


def save_training_data(X_data, Y_data, metadata, train_ratio=0.8):
    """Split and save training data"""
    print(f"\n💾 Splitting and saving data...")

    n_samples = len(X_data)
    n_train = int(n_samples * train_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Split data
    X_train = X_data[train_indices]
    Y_train = Y_data[train_indices]
    X_val = X_data[val_indices]
    Y_val = Y_data[val_indices]

    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")

    # Save training data
    train_data = {
        'X': X_train.tolist(),
        'Y': Y_train.tolist()
    }

    with open('train_set.json', 'w') as f:
        json.dump(train_data, f)

    # Save validation data
    val_data = {
        'X': X_val.tolist(),
        'Y': Y_val.tolist()
    }

    with open('val_set.json', 'w') as f:
        json.dump(val_data, f)

    print(f"✅ Data saved:")
    print(f"   train_set.json: ~{len(json.dumps(train_data))/1024/1024:.1f} MB")
    print(f"   val_set.json: ~{len(json.dumps(val_data))/1024/1024:.1f} MB")

    return X_train, Y_train, X_val, Y_val


def main():
    """Main function to create proper swaption training data"""
    print("🚀 PROPER SWAPTION DATA GENERATOR WITH NORMALIZATION")
    print("="*60)

    # Load CSV data
    dates, prices, tenors, maturities = load_swaption_csv()

    # Normalize prices
    normalized_prices, scaler = normalize_prices(prices)

    # Create training sequences with normalized prices
    X_data, Y_data, metadata = create_training_sequences(dates, normalized_prices, tenors, maturities)

    # Analyze statistics
    analyze_data_statistics(X_data, Y_data, metadata)

    # Plot sample sequences
    plot_sample_sequences(X_data, Y_data, metadata, n_samples=4)

    # Save data
    X_train, Y_train, X_val, Y_val = save_training_data(X_data, Y_data, metadata)

    print(f"\n✅ SUCCESS: Proper swaption training data created!")
    print(f"🎯 Ready for adversarial training with realistic market data")
    print(f"📊 Each sequence: 60 input days → 14 future days")
    print(f"🔧 Features: [price, tenor, maturity]")


if __name__ == "__main__":
    main()