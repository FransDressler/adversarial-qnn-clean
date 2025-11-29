#!/usr/bin/env python3
"""
Adversarial MerLin-Style Swaption Predictor
Löst das "Regression to Mean" Problem mit GAN-style Training

Architecture: Generator (Decomposed QNN) vs Discriminator (Classical NN)
Generator: Erzeugt realistische Swaption-Sequences
Discriminator: Unterscheidet echte vs generierte Sequences
"""

import torch
import torch.nn as nn
import numpy as np
import json
from typing import Tuple, List, Dict
from scipy import signal
from scipy.ndimage import uniform_filter1d

# Import MerLin components
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.algorithms.layer import QuantumLayer, MeasurementStrategy
from merlin.core.computation_space import ComputationSpace
from merlin import LexGrouping


class SignalDecomposer:
    """Signal decomposition für Adversarial Training (copied from decomposed)"""

    def __init__(self, trend_window: int = 15, level_window: int = 7, trend_polyorder: int = 2):
        self.trend_window = trend_window
        self.level_window = level_window
        self.trend_polyorder = trend_polyorder

    def decompose_batch(self, prices_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Zerlegt Batch von Preis-Zeitreihen"""
        batch_size, seq_len = prices_batch.shape

        trends = torch.zeros_like(prices_batch)
        levels = torch.zeros_like(prices_batch)
        noise = torch.zeros_like(prices_batch)

        for batch_idx in range(batch_size):
            prices = prices_batch[batch_idx].detach().cpu().numpy()

            # Trend extraction
            if len(prices) >= self.trend_window:
                trend_smooth = signal.savgol_filter(
                    prices, window_length=self.trend_window,
                    polyorder=self.trend_polyorder, mode='nearest'
                )
            else:
                trend_smooth = uniform_filter1d(prices, size=3, mode='nearest')

            # Level extraction
            price_levels = np.zeros_like(prices)
            half_window = self.level_window // 2

            for i in range(len(prices)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(prices), i + half_window + 1)
                price_levels[i] = np.median(prices[start_idx:end_idx])

            # Noise extraction
            noise_component = prices - trend_smooth

            # Convert back to same device as input
            device = prices_batch.device
            trends[batch_idx] = torch.tensor(trend_smooth, dtype=torch.float32, device=device)
            levels[batch_idx] = torch.tensor(price_levels, dtype=torch.float32, device=device)
            noise[batch_idx] = torch.tensor(noise_component, dtype=torch.float32, device=device)

        return {'trend': trends, 'level': levels, 'noise': noise, 'original': prices_batch}

    def get_decomposition_features(self, decomposed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extrahiert Features aus decomposed Signalen"""
        trends = decomposed['trend']
        levels = decomposed['level']
        noise = decomposed['noise']

        # Trend features
        trend_slope = (trends[:, -1] - trends[:, 0]) / trends.shape[1]
        trend_recent = (trends[:, -7:].mean(dim=1) - trends[:, -14:-7].mean(dim=1))
        trend_accel = (trends[:, -5:].mean(dim=1) - trends[:, -10:-5].mean(dim=1)) - \
                     (trends[:, -15:-10].mean(dim=1) - trends[:, -20:-15].mean(dim=1))
        trend_curvature = torch.std(torch.diff(trends, dim=1), dim=1)

        # Ensure all tensors are on the same device as input
        device = trends.device
        trend_features = torch.stack([trend_slope, trend_recent, trend_accel, trend_curvature], dim=1).to(device)

        # Level features
        current_level = levels[:, -1]
        level_range = levels.max(dim=1)[0] - levels.min(dim=1)[0]
        level_stability = 1.0 / (1.0 + torch.std(levels, dim=1))
        level_position = (current_level - levels.min(dim=1)[0]) / (level_range + 1e-8)

        level_features = torch.stack([current_level, level_range, level_stability, level_position], dim=1).to(device)

        # Noise features
        noise_volatility = torch.std(noise, dim=1)
        noise_skewness = torch.mean(noise ** 3, dim=1) / (noise_volatility ** 3 + 1e-8)
        noise_kurtosis = torch.mean(noise ** 4, dim=1) / (noise_volatility ** 4 + 1e-8)
        noise_recent = torch.std(noise[:, -14:], dim=1)

        noise_features = torch.stack([noise_volatility, noise_skewness, noise_kurtosis, noise_recent], dim=1).to(device)

        return {
            'trend_features': trend_features,
            'level_features': level_features,
            'noise_features': noise_features
        }


class QuantumGenerator(nn.Module):
    """
    Quantum Generator (based on Decomposed MerLin)
    Generates realistic swaption price sequences
    """

    def __init__(self, window_size: int = 60, prediction_horizon: int = 14, n_photons: int = 4):
        super().__init__()

        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.n_photons = n_photons

        self.decomposer = SignalDecomposer()
        self._build_quantum_cores()
        self._build_merlin_processing()

    def _build_quantum_cores(self):
        """Build specialized quantum circuits"""

        # Trend QNN
        trend_builder = CircuitBuilder(n_modes=6)
        trend_builder.add_entangling_layer(trainable=True, name="trend_U1")
        trend_builder.add_angle_encoding(modes=list(range(4)), name="trend_input")
        trend_builder.add_rotations(trainable=True, name="trend_rotations")
        trend_builder.add_superpositions(depth=2)
        trend_builder.add_entangling_layer(trainable=True, name="trend_U2")

        self.trend_qnn = QuantumLayer(
            input_size=4, builder=trend_builder, n_photons=self.n_photons,
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
            computation_space=ComputationSpace.UNBUNCHED, dtype=torch.float32
        )

        # Level QNN
        level_builder = CircuitBuilder(n_modes=8)
        level_builder.add_entangling_layer(trainable=True, name="level_U1")
        level_builder.add_angle_encoding(modes=list(range(6)), name="level_input")
        level_builder.add_rotations(trainable=True, name="level_rotations")
        level_builder.add_superpositions(depth=3)
        level_builder.add_entangling_layer(trainable=True, name="level_U2")

        self.level_qnn = QuantumLayer(
            input_size=6, builder=level_builder, n_photons=self.n_photons + 1,
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
            computation_space=ComputationSpace.UNBUNCHED, dtype=torch.float32
        )

        # Noise QNN
        noise_builder = CircuitBuilder(n_modes=4)
        noise_builder.add_entangling_layer(trainable=True, name="noise_U1")
        noise_builder.add_angle_encoding(modes=list(range(4)), name="noise_input")
        noise_builder.add_rotations(trainable=True, name="noise_rotations")
        noise_builder.add_superpositions(depth=1)

        self.noise_qnn = QuantumLayer(
            input_size=4, builder=noise_builder, n_photons=3,
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
            computation_space=ComputationSpace.UNBUNCHED, dtype=torch.float32
        )

    def _build_merlin_processing(self):
        """Build post-quantum processing"""

        trend_output_size = self.trend_qnn.output_size
        level_output_size = self.level_qnn.output_size
        noise_output_size = self.noise_qnn.output_size

        # LexGrouping
        self.trend_grouping = LexGrouping(trend_output_size, self.prediction_horizon)
        self.level_grouping = LexGrouping(level_output_size, 3)
        self.noise_grouping = LexGrouping(noise_output_size, 5)

        # Assembly layers
        self.trend_assembly = nn.Linear(self.prediction_horizon, self.prediction_horizon)
        self.level_assembly = nn.Linear(3, 1)
        self.noise_assembly = nn.Linear(5, self.prediction_horizon)

        # Final combination
        total_components = self.prediction_horizon + 1 + self.prediction_horizon
        self.final_assembly = nn.Linear(total_components, self.prediction_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate realistic swaption sequences"""
        batch_size = x.shape[0]

        prices = x[:, :, 0]
        tenors = x[:, 0, 1]
        maturities = x[:, 0, 2]

        # Signal decomposition
        decomposed = self.decomposer.decompose_batch(prices)
        features = self.decomposer.get_decomposition_features(decomposed)

        # Quantum processing
        trend_features = features['trend_features']
        trend_quantum_output = self.trend_qnn(trend_features)

        level_features = features['level_features']
        level_input = torch.cat([level_features, tenors.unsqueeze(1), maturities.unsqueeze(1)], dim=1)
        level_quantum_output = self.level_qnn(level_input)

        noise_features = features['noise_features']
        noise_quantum_output = self.noise_qnn(noise_features)

        # Post-quantum processing
        trend_components = self.trend_grouping(trend_quantum_output)
        level_components = self.level_grouping(level_quantum_output)
        noise_components = self.noise_grouping(noise_quantum_output)

        # Assembly
        trend_predictions = self.trend_assembly(trend_components)
        base_level = self.level_assembly(level_components)
        level_broadcast = base_level.repeat(1, self.prediction_horizon)
        noise_predictions = self.noise_assembly(noise_components)

        # Final assembly
        combined_components = torch.cat([trend_predictions, base_level, noise_predictions], dim=1)
        final_predictions = self.final_assembly(combined_components)

        # Add baseline
        last_price = prices[:, -1].unsqueeze(1)
        predictions = last_price + final_predictions

        return predictions


class SequenceDiscriminator(nn.Module):
    """
    Classical Neural Network Discriminator
    Unterscheidet echte vs generierte Swaption-Sequences
    """

    def __init__(self, sequence_length: int = 14):
        super().__init__()

        self.sequence_length = sequence_length

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            # 1D Convolutions für sequence patterns
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(8),  # Compress to 8 features
            nn.Flatten()
        )

        # Statistical feature extraction
        self.stat_processor = nn.Sequential(
            nn.Linear(5, 16),  # mean, std, skew, kurt, trend
            nn.LeakyReLU(0.2),
            nn.Linear(16, 8)
        )

        # Pattern recognition layers
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 + 8, 128),  # conv features + stat features
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),  # Real vs Fake probability
            nn.Sigmoid()
        )

    def extract_statistical_features(self, sequences: torch.Tensor) -> torch.Tensor:
        """Extract statistical features from sequences"""
        batch_size = sequences.shape[0]

        # Calculate statistics
        mean_val = torch.mean(sequences, dim=1)
        std_val = torch.std(sequences, dim=1)

        # Skewness approximation
        centered = sequences - mean_val.unsqueeze(1)
        skew_val = torch.mean(centered ** 3, dim=1) / (std_val ** 3 + 1e-8)

        # Kurtosis approximation
        kurt_val = torch.mean(centered ** 4, dim=1) / (std_val ** 4 + 1e-8)

        # Trend (last vs first)
        trend_val = sequences[:, -1] - sequences[:, 0]

        # Stack all statistical features
        stat_features = torch.stack([mean_val, std_val, skew_val, kurt_val, trend_val], dim=1)

        return stat_features

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Classify sequences as real vs fake

        Args:
            sequences: [batch_size, 14] - Swaption price sequences

        Returns:
            probabilities: [batch_size, 1] - Probability of being real (0-1)
        """
        batch_size = sequences.shape[0]

        # Reshape for Conv1D
        sequences_reshaped = sequences.unsqueeze(1)  # [batch, 1, 14]

        # Extract convolutional features
        conv_features = self.feature_extractor(sequences_reshaped)  # [batch, 32*8]

        # Extract statistical features
        stat_features = self.extract_statistical_features(sequences)  # [batch, 5]
        stat_features = self.stat_processor(stat_features)  # [batch, 8]

        # Combine features
        combined_features = torch.cat([conv_features, stat_features], dim=1)

        # Classify
        probability = self.classifier(combined_features)

        return probability


class AdversarialLoss:
    """
    Adversarial Loss Functions für Generator und Discriminator
    """

    def __init__(self, lambda_adv: float = 1.0, lambda_recon: float = 10.0):
        self.lambda_adv = lambda_adv      # Adversarial loss weight
        self.lambda_recon = lambda_recon  # Reconstruction loss weight

    def generator_loss(self, fake_predictions: torch.Tensor, targets: torch.Tensor,
                      discriminator_fake_output: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Generator Loss = Reconstruction Loss + Adversarial Loss

        Generator wants to:
        1. Generate accurate predictions (reconstruction)
        2. Fool the discriminator (adversarial)
        """

        # 1. Reconstruction Loss (accuracy)
        recon_loss = torch.nn.functional.mse_loss(fake_predictions, targets)

        # 2. Adversarial Loss (fool discriminator)
        # Generator wants discriminator to output 1 (real) for fake data
        adv_loss = torch.nn.functional.binary_cross_entropy(
            discriminator_fake_output,
            torch.ones_like(discriminator_fake_output)
        )

        # 3. Combined Generator Loss
        total_loss = self.lambda_recon * recon_loss + self.lambda_adv * adv_loss

        return total_loss, {
            'recon_loss': recon_loss.item(),
            'adv_loss': adv_loss.item(),
            'total_loss': total_loss.item()
        }

    def discriminator_loss(self, discriminator_real_output: torch.Tensor,
                          discriminator_fake_output: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Discriminator Loss = Real Loss + Fake Loss

        Discriminator wants to:
        1. Output 1 for real sequences
        2. Output 0 for fake sequences
        """

        # Real sequences should be classified as 1
        real_loss = torch.nn.functional.binary_cross_entropy(
            discriminator_real_output,
            torch.ones_like(discriminator_real_output)
        )

        # Fake sequences should be classified as 0
        fake_loss = torch.nn.functional.binary_cross_entropy(
            discriminator_fake_output.detach(),  # Detach to avoid generator gradients
            torch.zeros_like(discriminator_fake_output)
        )

        # Combined Discriminator Loss
        total_loss = real_loss + fake_loss

        # Calculate accuracy
        real_accuracy = torch.mean((discriminator_real_output > 0.5).float())
        fake_accuracy = torch.mean((discriminator_fake_output < 0.5).float())
        total_accuracy = (real_accuracy + fake_accuracy) / 2.0

        return total_loss, {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'total_loss': total_loss.item(),
            'accuracy': total_accuracy.item()
        }


def load_training_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load training data"""
    with open('train_set.json', 'r') as f:
        train_data = json.load(f)
    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    Y_train = torch.tensor(train_data['Y'], dtype=torch.float32)
    print(f"Training data loaded: X={X_train.shape}, Y={Y_train.shape}")
    return X_train, Y_train


def load_validation_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load validation data"""
    with open('val_set.json', 'r') as f:
        val_data = json.load(f)
    X_val = torch.tensor(val_data['X'], dtype=torch.float32)
    Y_val = torch.tensor(val_data['Y'], dtype=torch.float32)
    print(f"Validation data loaded: X={X_val.shape}, Y={Y_val.shape}")
    return X_val, Y_val


if __name__ == "__main__":
    print("🔮 Adversarial MerLin-Style Swaption Predictor")
    print("🥊 Generator (Quantum) vs Discriminator (Classical)")
    print("="*60)

    # Initialize models
    generator = QuantumGenerator()
    discriminator = SequenceDiscriminator()

    print(f"📊 Generator Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"📊 Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Load test data
    X_train, Y_train = load_training_data()

    # Test forward pass
    with torch.no_grad():
        sample_input = X_train[:4]
        sample_targets = Y_train[:4]

        # Generator
        fake_predictions = generator(sample_input)
        print(f"\n🧪 Generator Test:")
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {fake_predictions.shape}")

        # Discriminator
        real_prob = discriminator(sample_targets)
        fake_prob = discriminator(fake_predictions)
        print(f"\n🧪 Discriminator Test:")
        print(f"   Real sequences probability: {real_prob.mean():.3f}")
        print(f"   Fake sequences probability: {fake_prob.mean():.3f}")

    print("\n✅ Adversarial Architecture Ready for Training!")
    print("🎯 Generator will learn to create realistic sequences that fool the discriminator")