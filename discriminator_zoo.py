#!/usr/bin/env python3
"""
Discriminator Zoo - Verschiedene bewährte Architekturen testen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResNet1DDiscriminator(nn.Module):
    """ResNet-style 1D CNN für Zeitreihen"""

    def __init__(self, sequence_length=14):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        # ResNet blocks
        self.res_block1 = self._make_res_block(64, 64)
        self.res_block2 = self._make_res_block(64, 128, stride=2)
        self.res_block3 = self._make_res_block(128, 128)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def _make_res_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, seq_len]

        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))

        # ResNet blocks with residual connections
        residual = x
        x = self.res_block1(x)
        x = F.relu(x + residual)

        x = self.res_block2(x)

        residual = x
        x = self.res_block3(x)
        x = F.relu(x + residual)

        # Global pooling + classification
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return self.sigmoid(x)


class TransformerDiscriminator(nn.Module):
    """Transformer-based discriminator für Sequenzen"""

    def __init__(self, sequence_length=14, d_model=64, nhead=4, num_layers=3):
        super().__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoding = self._create_positional_encoding(sequence_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def _create_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        # x: [batch, seq_len] -> [batch, seq_len, 1]
        x = x.unsqueeze(-1)

        # Project to d_model
        x = self.input_projection(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = x + self.pos_encoding.to(x.device)

        # Transformer
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # Global average pooling
        x = x.mean(dim=1)  # [batch, d_model]

        # Classification
        return self.classifier(x)


class LSTMDiscriminator(nn.Module):
    """LSTM-based discriminator für Sequenzen"""

    def __init__(self, sequence_length=14, hidden_size=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, seq_len] -> [batch, seq_len, 1]
        x = x.unsqueeze(-1)

        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)

        # Use last hidden state
        final_hidden = hidden[-1]  # [batch, hidden_size]

        return self.classifier(final_hidden)


class SimpleMLPDiscriminator(nn.Module):
    """Einfacher MLP - Baseline"""

    def __init__(self, sequence_length=14):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(sequence_length, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)


class WaveNetDiscriminator(nn.Module):
    """WaveNet-style dilated convolutions"""

    def __init__(self, sequence_length=14, channels=64):
        super().__init__()

        self.input_conv = nn.Conv1d(1, channels, 1)

        # Dilated convolutions with increasing dilation
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(channels, channels, 3, dilation=1, padding=1),
            nn.Conv1d(channels, channels, 3, dilation=2, padding=2),
            nn.Conv1d(channels, channels, 3, dilation=4, padding=4),
            nn.Conv1d(channels, channels, 3, dilation=8, padding=8)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv1d(channels, channels, 1) for _ in self.dilated_convs
        ])

        self.final_conv = nn.Conv1d(channels, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, seq_len]

        # Input projection
        x = self.input_conv(x)
        skip_connections = 0

        # Dilated convolutions
        for dilated_conv, skip_conv in zip(self.dilated_convs, self.skip_convs):
            residual = x
            x = F.tanh(dilated_conv(x))
            skip = skip_conv(x)
            skip_connections += skip
            x = x + residual  # Residual connection

        # Final layers
        x = F.relu(skip_connections)
        x = self.final_conv(x)
        x = self.global_pool(x).squeeze(-1)
        return self.sigmoid(x)


def test_discriminator_zoo():
    """Teste alle Discriminator Architekturen"""
    print("🦁 DISCRIMINATOR ZOO - ARCHITECTURE TESTING")
    print("="*60)

    from adversarial_merlin_swaption import QuantumGenerator, load_validation_data

    # Lade Daten
    X_val, Y_val = load_validation_data()
    generator = QuantumGenerator()

    test_size = 200
    X_test = X_val[:test_size]
    Y_real = Y_val[:test_size]

    # Generiere fake data
    with torch.no_grad():
        Y_fake = generator(X_test)

    print(f"📊 Test Data:")
    print(f"   Real volatility: {torch.std(Y_real, dim=1).mean():.6f}")
    print(f"   Fake volatility: {torch.std(Y_fake, dim=1).mean():.6f}")
    print(f"   Ratio: {torch.std(Y_fake, dim=1).mean() / torch.std(Y_real, dim=1).mean():.1f}x")

    # Teste verschiedene Discriminatoren
    discriminators = {
        "Simple MLP": SimpleMLPDiscriminator(),
        "ResNet1D": ResNet1DDiscriminator(),
        "LSTM": LSTMDiscriminator(),
        "Transformer": TransformerDiscriminator(),
        "WaveNet": WaveNetDiscriminator()
    }

    results = {}

    for name, discriminator in discriminators.items():
        print(f"\n🧪 Testing {name}...")

        # Parameter count
        param_count = sum(p.numel() for p in discriminator.parameters())
        print(f"   Parameters: {param_count:,}")

        # Optimizer
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        # Training steps
        best_acc = 0
        for step in range(10):
            optimizer.zero_grad()

            real_output = discriminator(Y_real)
            fake_output = discriminator(Y_fake)

            real_loss = criterion(real_output, torch.ones_like(real_output))
            fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
            total_loss = real_loss + fake_loss

            total_loss.backward()
            optimizer.step()

            # Calculate accuracy
            real_acc = (real_output > 0.5).float().mean()
            fake_acc = (fake_output < 0.5).float().mean()
            total_acc = (real_acc + fake_acc) / 2.0

            if total_acc > best_acc:
                best_acc = total_acc

        # Final test
        with torch.no_grad():
            real_final = discriminator(Y_real)
            fake_final = discriminator(Y_fake)

            separation = real_final.mean() - fake_final.mean()

        results[name] = {
            'accuracy': best_acc.item(),
            'separation': separation.item(),
            'parameters': param_count,
            'real_score': real_final.mean().item(),
            'fake_score': fake_final.mean().item()
        }

        print(f"   Best accuracy: {best_acc:.3f}")
        print(f"   Separation: {separation:.3f}")
        print(f"   Real/Fake: {real_final.mean():.3f}/{fake_final.mean():.3f}")

    # Summary
    print(f"\n" + "="*60)
    print(f"📊 DISCRIMINATOR ZOO RESULTS:")
    print(f"{'Architecture':<12} {'Params':<8} {'Accuracy':<8} {'Separation':<10} {'Status'}")
    print(f"-" * 60)

    for name, result in results.items():
        status = "✅ WORKS" if result['accuracy'] > 0.7 else "⚠️  WEAK" if result['accuracy'] > 0.6 else "❌ FAILS"
        print(f"{name:<12} {result['parameters']:<8,} {result['accuracy']:<8.3f} {result['separation']:<10.3f} {status}")

    # Best performer
    best = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 BEST: {best[0]} with {best[1]['accuracy']:.3f} accuracy")

    return results


if __name__ == "__main__":
    results = test_discriminator_zoo()