#!/usr/bin/env python3
"""
Create Training Data from Original Daten.csv
Reads from real swaption CSV data and creates structured training sequences
"""

import numpy as np
import json
import csv


def load_csv_data(csv_path="../Daten.csv"):
    """Load and process the original CSV data"""
    print(f"📁 Loading CSV data from {csv_path}...")

    dates = []
    data = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)  # Skip header

        for row in reader:
            if len(row) < 2:
                continue
            date_str = row[0]
            values = [float(val.replace(',', '.')) for val in row[1:] if val]

            dates.append(date_str)
            data.append(values)

    data_array = np.array(data, dtype=np.float32)
    print(f"✅ CSV loaded: {len(dates)} dates, {data_array.shape[1]} tenor/maturity combinations")

    return dates, data_array


def create_training_sequences(dates, data_array, sequence_length=60, prediction_length=14):
    """
    Create training sequences from CSV data using sliding windows

    For each column (tenor/maturity combination), create sequences:
    - Input: 60 historical values + tenor + maturity
    - Output: 14 future values to predict
    """

    n_dates, n_columns = data_array.shape
    print(f"🔄 Creating training sequences...")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Prediction length: {prediction_length}")

    # Extract tenor/maturity info (simplified - use column index)
    X_sequences = []
    Y_sequences = []

    # For each column (tenor/maturity combination)
    for col_idx in range(n_columns):
        column_data = data_array[:, col_idx]

        # Create sliding windows
        max_start = n_dates - sequence_length - prediction_length

        if max_start <= 0:
            print(f"   ⚠️  Column {col_idx}: Not enough data for sequences (need {sequence_length + prediction_length}, have {n_dates})")
            continue

        for start_idx in range(max_start):
            # Input sequence (60 timesteps)
            input_seq = column_data[start_idx:start_idx + sequence_length]

            # Target sequence (14 future values)
            target_seq = column_data[start_idx + sequence_length:start_idx + sequence_length + prediction_length]

            # Create feature matrix [timesteps, 3 features]
            X_sample = np.zeros((sequence_length, 3), dtype=np.float32)
            X_sample[:, 0] = input_seq  # Price values
            X_sample[:, 1] = (col_idx % 14) / 13.0  # Normalized tenor proxy
            X_sample[:, 2] = (col_idx // 14) / 13.0  # Normalized maturity proxy

            X_sequences.append(X_sample)
            Y_sequences.append(target_seq)

    X_data = np.array(X_sequences, dtype=np.float32)
    Y_data = np.array(Y_sequences, dtype=np.float32)

    print(f"✅ Sequences created:")
    print(f"   Total sequences: {len(X_sequences)}")
    print(f"   X shape: {X_data.shape}")
    print(f"   Y shape: {Y_data.shape}")

    return X_data, Y_data


def split_data(X_data, Y_data, train_ratio=0.8):
    """Split data into training and validation sets"""

    n_samples = len(X_data)
    n_train = int(n_samples * train_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    X_train = X_data[train_indices]
    Y_train = Y_data[train_indices]
    X_val = X_data[val_indices]
    Y_val = Y_data[val_indices]

    print(f"📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")

    return X_train, Y_train, X_val, Y_val


def save_training_data(X_train, Y_train, X_val, Y_val):
    """Save training data in JSON format"""

    print("💾 Saving training data...")

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

    print(f"✅ Training data saved:")
    print(f"   train_set.json: ~{len(json.dumps(train_data))/1024/1024:.1f} MB")
    print(f"   val_set.json: ~{len(json.dumps(val_data))/1024/1024:.1f} MB")


def test_data_loading():
    """Test if the generated data works with our existing functions"""

    print("🧪 Testing data loading...")

    try:
        # Load training data
        with open('train_set.json', 'r') as f:
            train_data = json.load(f)
        X_train = np.array(train_data['X'], dtype=np.float32)
        Y_train = np.array(train_data['Y'], dtype=np.float32)

        # Load validation data
        with open('val_set.json', 'r') as f:
            val_data = json.load(f)
        X_val = np.array(val_data['X'], dtype=np.float32)
        Y_val = np.array(val_data['Y'], dtype=np.float32)

        print(f"✅ Loading test successful:")
        print(f"   Training data loaded: X={X_train.shape}, Y={Y_train.shape}")
        print(f"   Validation data loaded: X={X_val.shape}, Y={Y_val.shape}")

        # Test some basic properties
        train_vol = np.std(Y_train, axis=1).mean()
        val_vol = np.std(Y_val, axis=1).mean()

        print(f"📊 Loaded Data Properties:")
        print(f"   Train volatility: {train_vol:.6f}")
        print(f"   Val volatility: {val_vol:.6f}")

        return True

    except Exception as e:
        print(f"❌ Loading test failed: {e}")
        return False


def main():
    """Main function to create training data from CSV"""
    print("🚀 REAL TRAINING DATA GENERATOR FROM CSV")
    print("="*50)

    # Load CSV data
    dates, data_array = load_csv_data()

    # Create sequences
    X_data, Y_data = create_training_sequences(dates, data_array)

    if len(X_data) == 0:
        print("❌ No training sequences created!")
        return

    # Split into train/val
    X_train, Y_train, X_val, Y_val = split_data(X_data, Y_data)

    # Calculate statistics
    train_vol = np.std(Y_train, axis=1).mean()
    val_vol = np.std(Y_val, axis=1).mean()

    print(f"📊 Data Statistics:")
    print(f"   Train volatility: {train_vol:.6f}")
    print(f"   Val volatility: {val_vol:.6f}")
    print(f"   Train price range: [{X_train[:,:,0].min():.4f}, {X_train[:,:,0].max():.4f}]")
    print(f"   Future price range: [{Y_train.min():.4f}, {Y_train.max():.4f}]")

    # Save data
    save_training_data(X_train, Y_train, X_val, Y_val)

    # Test loading
    success = test_data_loading()

    if success:
        print(f"\n✅ SUCCESS: Real training data ready for GPU training!")
        print(f"🎯 Generated from {len(dates)} dates and {data_array.shape[1]} tenor/maturity combinations")
        print(f"📊 Total sequences: {len(X_data)}")
    else:
        print(f"\n❌ FAILED: Check data generation")


if __name__ == "__main__":
    main()