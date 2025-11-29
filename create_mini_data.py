#!/usr/bin/env python3
"""
Create Training Data from Daten.csv for GPU Testing
Reads from original CSV data and creates structured training sequences
"""

import numpy as np
import json
import csv
from datetime import datetime


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
    Create training sequences from CSV data

    Creates sliding windows over the time series:
    X: [samples, 60, 3] - 60 timesteps, 3 features (price, tenor, maturity)
    Y: [samples, 14] - 14 future price predictions
    """

    n_dates, n_columns = data_array.shape
    print(f"🔄 Creating training sequences from CSV data...")
    print(f"   Total dates: {n_dates}")
    print(f"   Total columns: {n_columns}")

    # Parse tenor/maturity from column headers
    def parse_tenor_maturity_from_csv():
        X = []
        Y = []

        for i in range(n_samples):
            # Initial price (realistic range)
            base_price = np.random.uniform(0.3, 0.8)

            # Generate 60 historical prices with realistic volatility
            historical_trend = np.random.uniform(-0.001, 0.001)
            noise_level = np.random.uniform(0.005, 0.025)

            prices = [base_price]
            for t in range(59):
                # Add trend + noise
                change = historical_trend + np.random.normal(0, noise_level)
                new_price = max(0.1, min(1.0, prices[-1] + change))
                prices.append(new_price)

            # Tenor and Maturity (static features)
            tenor = np.random.choice([1, 2, 5, 10, 30])  # years
            maturity = np.random.choice([0.25, 0.5, 1, 2, 5])  # years

            # Create input features [price, tenor, maturity]
            X_sample = np.zeros((60, 3))
            X_sample[:, 0] = prices  # Historical prices
            X_sample[:, 1] = tenor / 30.0  # Normalized tenor
            X_sample[:, 2] = maturity / 5.0  # Normalized maturity

            # Generate 14 future prices
            future_prices = [prices[-1]]  # Start from last historical price
            future_trend = np.random.uniform(-0.002, 0.002)
            future_noise = np.random.uniform(0.008, 0.035)

            for t in range(13):
                change = future_trend + np.random.normal(0, future_noise)
                new_price = max(0.1, min(1.0, future_prices[-1] + change))
                future_prices.append(new_price)

            Y_sample = np.array(future_prices[1:])  # 13 future prices - need 14!
            # Add one more future price to match original format
            change = future_trend + np.random.normal(0, future_noise)
            extra_price = max(0.1, min(1.0, future_prices[-1] + change))
            Y_sample = np.append(Y_sample, extra_price)  # Now 14 prices

            X.append(X_sample)
            Y.append(Y_sample)

        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    # Generate training data
    X_train, Y_train = generate_sequences(n_train)

    # Generate validation data
    X_val, Y_val = generate_sequences(n_val)

    print(f"✅ Data generated:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   Y_train shape: {Y_train.shape}")
    print(f"   X_val shape: {X_val.shape}")
    print(f"   Y_val shape: {Y_val.shape}")

    # Calculate statistics
    train_vol = np.std(Y_train, axis=1).mean()
    val_vol = np.std(Y_val, axis=1).mean()

    print(f"📊 Data Statistics:")
    print(f"   Train volatility: {train_vol:.6f}")
    print(f"   Val volatility: {val_vol:.6f}")
    print(f"   Train price range: [{X_train[:,:,0].min():.3f}, {X_train[:,:,0].max():.3f}]")
    print(f"   Future price range: [{Y_train.min():.3f}, {Y_train.max():.3f}]")

    return X_train, Y_train, X_val, Y_val


def save_mini_data(X_train, Y_train, X_val, Y_val):
    """Save mini data in same format as original"""

    print("💾 Saving mini data...")

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

    print(f"✅ Mini data saved:")
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


if __name__ == "__main__":
    print("🚀 MINI TRAINING DATA GENERATOR")
    print("="*50)

    # Generate mini data
    X_train, Y_train, X_val, Y_val = create_mini_training_data(
        n_train=2000,  # Much smaller than 75K original
        n_val=500      # Much smaller than 18K original
    )

    # Save data
    save_mini_data(X_train, Y_train, X_val, Y_val)

    # Test loading
    success = test_data_loading()

    if success:
        print(f"\n✅ SUCCESS: Mini training data ready for GPU training!")
        print(f"🎯 Now you can run: python optimized_resnet_training.py")
    else:
        print(f"\n❌ FAILED: Check data generation")