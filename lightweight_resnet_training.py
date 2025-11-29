#!/usr/bin/env python3
"""
Lightweight ResNet Adversarial Training - Kleine Batches, weniger Memory
"""

import torch
import torch.nn as nn
import time
from adversarial_merlin_swaption import QuantumGenerator, load_validation_data


class LightweightResNetDiscriminator(nn.Module):
    """Kleinere ResNet Version - weniger Parameter"""

    def __init__(self):
        super().__init__()

        # Viel kleinere ResNet
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)  # 64->32
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)  # 64->32
        self.bn2 = nn.BatchNorm1d(32)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),  # Kleiner
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, seq_len]

        # Simple conv layers
        x = torch.relu(self.bn1(self.conv1(x)))
        residual = x
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x + residual  # Residual

        # Global pooling + classification
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


def lightweight_training():
    print("🚀 LIGHTWEIGHT RESNET TRAINING")
    print("Small batches, fewer parameters, quick test")
    print("="*50)

    # Load SMALL data subset
    X_val, Y_val = load_validation_data()

    # VERY small training set
    train_size = 500  # Nur 500 samples!
    batch_size = 16   # Kleine batches

    X_train = X_val[:train_size]
    Y_train = Y_val[:train_size]
    X_val_small = X_val[train_size:train_size+200]
    Y_val_small = Y_val[train_size:train_size+200]

    print(f"📊 Data: {train_size} train, 200 val, batch_size={batch_size}")

    # Initialize models
    generator = QuantumGenerator()
    discriminator = LightweightResNetDiscriminator()

    print(f"📊 Generator: {sum(p.numel() for p in generator.parameters()):,} params")
    print(f"📊 Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} params")

    # Test forward pass first
    print(f"\n🧪 Testing forward pass...")

    try:
        with torch.no_grad():
            # Small test batch
            X_test = X_train[:4]  # Nur 4 samples
            Y_test = Y_train[:4]

            print(f"   Generator test...")
            fake = generator(X_test)
            print(f"   ✅ Generator works: {fake.shape}")

            print(f"   Discriminator test...")
            real_score = discriminator(Y_test)
            fake_score = discriminator(fake)
            print(f"   ✅ Discriminator works: real={real_score.mean():.3f}, fake={fake_score.mean():.3f}")

    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return

    # Optimizers
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    print(f"\n🔥 Starting lightweight training...")

    # Quick training loop
    for epoch in range(5):  # Nur 5 epochs
        epoch_start = time.time()

        # Process in small batches
        num_batches = len(X_train) // batch_size

        gen_loss_sum = 0.0
        disc_acc_sum = 0.0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            X_batch = X_train[start_idx:end_idx]
            Y_batch = Y_train[start_idx:end_idx]

            # Train discriminator
            disc_optimizer.zero_grad()

            with torch.no_grad():
                fake_batch = generator(X_batch)

            real_out = discriminator(Y_batch)
            fake_out = discriminator(fake_batch)

            # Simple BCE loss
            disc_loss = (torch.nn.functional.binary_cross_entropy(real_out, torch.ones_like(real_out)) +
                        torch.nn.functional.binary_cross_entropy(fake_out, torch.zeros_like(fake_out)))

            disc_loss.backward()
            disc_optimizer.step()

            # Calculate accuracy
            real_acc = (real_out > 0.5).float().mean()
            fake_acc = (fake_out < 0.5).float().mean()
            total_acc = (real_acc + fake_acc) / 2.0
            disc_acc_sum += total_acc.item()

            # Train generator
            gen_optimizer.zero_grad()

            fake_batch = generator(X_batch)
            fake_out = discriminator(fake_batch)

            # Combined generator loss
            recon_loss = torch.nn.functional.mse_loss(fake_batch, Y_batch)
            adv_loss = torch.nn.functional.binary_cross_entropy(fake_out, torch.ones_like(fake_out))
            gen_loss = 3.0 * recon_loss + 2.0 * adv_loss  # Simple weights

            gen_loss.backward()
            gen_optimizer.step()

            gen_loss_sum += gen_loss.item()

        # Average metrics
        avg_gen_loss = gen_loss_sum / num_batches
        avg_disc_acc = disc_acc_sum / num_batches

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/5: Gen Loss={avg_gen_loss:.4f}, Disc Acc={avg_disc_acc:.3f}, Time={epoch_time:.1f}s")

    # Final test
    print(f"\n📊 Final Test:")
    with torch.no_grad():
        test_fake = generator(X_val_small[:50])
        test_real_scores = discriminator(Y_val_small[:50])
        test_fake_scores = discriminator(test_fake)

        real_vol = torch.std(Y_val_small[:50], dim=1).mean()
        fake_vol = torch.std(test_fake, dim=1).mean()

        print(f"   Real volatility: {real_vol:.6f}")
        print(f"   Fake volatility: {fake_vol:.6f}")
        print(f"   Ratio: {fake_vol / real_vol:.3f}")
        print(f"   Real score: {test_real_scores.mean():.3f}")
        print(f"   Fake score: {test_fake_scores.mean():.3f}")

        if fake_vol / real_vol > 0.5:
            print(f"   ✅ Good volatility improvement!")
        else:
            print(f"   ⚠️  Still needs work")

if __name__ == "__main__":
    lightweight_training()