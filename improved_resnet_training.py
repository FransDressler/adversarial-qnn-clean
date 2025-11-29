#!/usr/bin/env python3
"""
IMPROVED ResNet Adversarial Training - Option A Strategy
Prevents early stagnation through better hyperparameter balance
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import time
import os

from adversarial_merlin_swaption import (
    QuantumGenerator,
    load_training_data,
    load_validation_data
)
from discriminator_zoo import ResNet1DDiscriminator


class ImprovedAdversarialLoss:
    """
    Improved loss with better adversarial/reconstruction balance
    """

    def __init__(self, lambda_adv=4.0, lambda_recon=1.0, lambda_vol=2.0):
        # STRONGER adversarial weight, WEAKER reconstruction
        self.lambda_adv = lambda_adv      # Increased: 2.0 → 4.0
        self.lambda_recon = lambda_recon  # Decreased: 3.0 → 1.0
        self.lambda_vol = lambda_vol      # Kept: 1.0 → 2.0

    def enhanced_generator_loss(self, fake_sequences, real_sequences, discriminator_output):
        """Enhanced generator loss with better balance"""
        batch_size = fake_sequences.size(0)

        # 1. Reconstruction loss (MSE) - REDUCED weight
        recon_loss = nn.MSELoss()(fake_sequences, real_sequences)

        # 2. Adversarial loss - INCREASED weight
        # Generator wants discriminator to think fakes are real (label = 1)
        real_labels = torch.ones(batch_size, 1, device=fake_sequences.device)
        adv_loss = nn.BCELoss()(discriminator_output, real_labels)

        # 3. Volatility preservation loss - INCREASED weight
        real_vol = torch.std(real_sequences, dim=1, keepdim=True)
        fake_vol = torch.std(fake_sequences, dim=1, keepdim=True)
        vol_loss = nn.MSELoss()(fake_vol, real_vol)

        # Combined loss with improved weights
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_adv * adv_loss +
            self.lambda_vol * vol_loss
        )

        loss_breakdown = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'adv_loss': adv_loss.item(),
            'volatility_loss': vol_loss.item()
        }

        return total_loss, loss_breakdown

    def discriminator_loss(self, real_output, fake_output):
        """Standard discriminator loss"""
        batch_size = real_output.size(0)

        # Real sequences should be classified as real (label = 1)
        real_labels = torch.ones(batch_size, 1, device=real_output.device)
        real_loss = nn.BCELoss()(real_output, real_labels)

        # Fake sequences should be classified as fake (label = 0)
        fake_labels = torch.zeros(batch_size, 1, device=fake_output.device)
        fake_loss = nn.BCELoss()(fake_output, fake_labels)

        total_loss = (real_loss + fake_loss) / 2

        # Calculate accuracy
        real_accuracy = ((real_output > 0.5).float() == real_labels).float().mean()
        fake_accuracy = ((fake_output < 0.5).float() == fake_labels).float().mean()
        accuracy = (real_accuracy + fake_accuracy) / 2

        return total_loss, {
            'total_loss': total_loss.item(),
            'accuracy': accuracy.item(),
            'real_accuracy': real_accuracy.item(),
            'fake_accuracy': fake_accuracy.item()
        }


def improved_adversarial_training():
    """
    IMPROVED adversarial training with better hyperparameter balance
    """
    print("🚀 IMPROVED ADVERSARIAL TRAINING - OPTION A STRATEGY")
    print("📊 Prevents early stagnation through better hyperparameter balance")
    print("🎯 Target: Break Nash equilibrium, achieve >0.7 volatility ratio")
    print("=" * 80)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Training on: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # IMPROVED configuration to prevent stagnation
    config = {
        'batch_size': 16,               # Keep small batches
        'num_epochs': 40,               # More epochs
        'batches_per_epoch': 300,       # Good coverage

        # LEARNING RATES - Better balance
        'lr_generator': 8e-4,           # DECREASED: 1e-3 → 8e-4
        'lr_discriminator': 2e-3,       # INCREASED: 5e-4 → 2e-3

        # TRAINING SCHEDULE - More discriminator updates
        'disc_updates_per_gen': 2,      # 2 disc updates per gen update
        'warmup_epochs': 3,             # Disc warmup period

        # LOSS WEIGHTS - Better adversarial balance
        'lambda_adv': 4.0,              # INCREASED: 2.0 → 4.0
        'lambda_recon': 1.0,            # DECREASED: 3.0 → 1.0
        'lambda_vol': 2.0,              # INCREASED: 1.0 → 2.0

        # LEARNING RATE SCHEDULE
        'lr_decay_epochs': [15, 25],    # Decay at these epochs
        'lr_decay_factor': 0.5,         # Multiply LR by this

        # EARLY STOPPING
        'patience': 10,                 # Stop if no improvement
        'min_improvement': 0.01         # Minimum improvement threshold
    }

    print(f"🔧 IMPROVED Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    # Load training data
    print(f"\n📁 Loading training data...")
    X_train, Y_train = load_training_data()
    X_val, Y_val = load_validation_data()

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)

    # Use substantial subset
    train_samples = min(config['batches_per_epoch'] * config['batch_size'], len(X_train))
    val_samples = min(50 * config['batch_size'], len(X_val))

    # Sample data
    train_indices = torch.randperm(len(X_train))[:train_samples]
    val_indices = torch.randperm(len(X_val))[:val_samples]

    X_train_subset = X_train[train_indices]
    Y_train_subset = Y_train[train_indices]
    X_val_subset = X_val[val_indices]
    Y_val_subset = Y_val[val_indices]

    # Create data loaders
    train_dataset = TensorDataset(X_train_subset, Y_train_subset)
    val_dataset = TensorDataset(X_val_subset, Y_val_subset)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"✅ Training data prepared:")
    print(f"   Train: {len(train_dataset):,} samples ({len(train_loader)} batches)")
    print(f"   Val: {len(val_dataset):,} samples ({len(val_loader)} batches)")

    # Initialize models
    generator = QuantumGenerator().to(device)
    discriminator = ResNet1DDiscriminator().to(device)

    print(f"📊 Generator Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"📊 ResNet1D Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"🖥️  Models moved to: {device}")

    # Loss function with improved weights
    loss_fn = ImprovedAdversarialLoss(
        lambda_adv=config['lambda_adv'],
        lambda_recon=config['lambda_recon'],
        lambda_vol=config['lambda_vol']
    )

    # Optimizers with IMPROVED learning rates
    gen_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config['lr_generator'],
        betas=(0.5, 0.999)
    )

    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config['lr_discriminator'],    # INCREASED LR
        betas=(0.5, 0.999)
    )

    # Learning rate schedulers
    gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        gen_optimizer,
        milestones=config['lr_decay_epochs'],
        gamma=config['lr_decay_factor']
    )

    disc_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        disc_optimizer,
        milestones=config['lr_decay_epochs'],
        gamma=config['lr_decay_factor']
    )

    # Training tracking
    best_vol_ratio = 0.0
    patience_counter = 0
    training_history = []

    run_name = f"improved_resnet_qnn_{int(time.time())}"
    print(f"🎯 Run: {run_name}")
    print()

    # Training loop with IMPROVED strategy
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()

        # ===== DISCRIMINATOR TRAINING (MULTIPLE UPDATES) =====
        discriminator.train()
        generator.eval()

        disc_metrics_sum = {'total_loss': 0.0, 'accuracy': 0.0}
        disc_batches = 0

        # IMPROVED: Multiple discriminator updates per generator update
        disc_updates = config['disc_updates_per_gen']
        if epoch < config['warmup_epochs']:
            disc_updates *= 2  # Extra updates during warmup

        for _ in range(disc_updates):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                disc_optimizer.zero_grad()

                with torch.no_grad():
                    fake_sequences = generator(inputs)

                # Real and fake discriminator outputs
                real_output = discriminator(targets)
                fake_output = discriminator(fake_sequences)

                disc_loss, disc_metrics = loss_fn.discriminator_loss(real_output, fake_output)
                disc_loss.backward()
                disc_optimizer.step()

                # Accumulate metrics
                for key, value in disc_metrics.items():
                    disc_metrics_sum[key] += value
                disc_batches += 1

        # Average discriminator metrics
        for key in disc_metrics_sum:
            disc_metrics_sum[key] /= disc_batches

        # ===== GENERATOR TRAINING (SINGLE UPDATE) =====
        generator.train()
        discriminator.eval()

        gen_metrics_sum = {'total_loss': 0.0, 'recon_loss': 0.0, 'adv_loss': 0.0, 'volatility_loss': 0.0}
        gen_batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            gen_optimizer.zero_grad()

            fake_sequences = generator(inputs)
            fake_output = discriminator(fake_sequences)

            gen_loss, gen_metrics = loss_fn.enhanced_generator_loss(fake_sequences, targets, fake_output)
            gen_loss.backward()
            gen_optimizer.step()

            # Accumulate metrics
            for key, value in gen_metrics.items():
                gen_metrics_sum[key] += value
            gen_batches += 1

        # Average generator metrics
        for key in gen_metrics_sum:
            gen_metrics_sum[key] /= gen_batches

        # ===== VALIDATION =====
        generator.eval()
        discriminator.eval()

        val_loss = 0.0
        val_real_vol = 0.0
        val_fake_vol = 0.0
        val_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                fake_sequences = generator(inputs)
                fake_output = discriminator(fake_sequences)
                gen_loss, _ = loss_fn.enhanced_generator_loss(fake_sequences, targets, fake_output)

                val_loss += gen_loss.item()

                # Calculate volatilities
                real_vol = torch.std(targets, dim=1).mean()
                fake_vol = torch.std(fake_sequences, dim=1).mean()
                val_real_vol += real_vol.item()
                val_fake_vol += fake_vol.item()
                val_batches += 1

        val_loss /= val_batches
        val_real_vol /= val_batches
        val_fake_vol /= val_batches
        vol_ratio = val_fake_vol / val_real_vol if val_real_vol > 0 else 0

        # Learning rate scheduling
        gen_scheduler.step()
        disc_scheduler.step()

        # Progress tracking
        epoch_time = time.time() - epoch_start
        progress = f"[{'█' * (epoch // 3)}{'░' * (config['num_epochs'] // 3 - epoch // 3)}]"

        print(f"{progress} Epoch {epoch+1:2d}/{config['num_epochs']} | "
              f"G: {gen_metrics_sum['total_loss']:.4f} "
              f"(R:{gen_metrics_sum['recon_loss']:.4f} A:{gen_metrics_sum['adv_loss']:.4f} V:{gen_metrics_sum['volatility_loss']:.4f}) | "
              f"D: {disc_metrics_sum['total_loss']:.2f} ({disc_metrics_sum['accuracy']:.3f}) | "
              f"Vol: {val_real_vol:.4f}→{val_fake_vol:.4f} ({vol_ratio:.3f}) | ",
              end="")

        # Best model tracking with IMPROVED criteria
        if vol_ratio > best_vol_ratio + config['min_improvement']:
            best_vol_ratio = vol_ratio
            patience_counter = 0

            # Save best model
            model_path = f"{run_name}_best_vol_{vol_ratio:.3f}_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                'vol_ratio': vol_ratio,
                'config': config
            }, model_path)

            print(f"💾 New best volatility! {vol_ratio:.3f} | {epoch_time:.1f}s")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"⏳ Early stop (patience) | {epoch_time:.1f}s")
                break
            else:
                print(f"⏳ Vol: {vol_ratio:.3f} ({patience_counter}/{config['patience']}) | {epoch_time:.1f}s")

        # Store history
        training_history.append({
            'epoch': epoch + 1,
            'gen_loss': gen_metrics_sum['total_loss'],
            'disc_loss': disc_metrics_sum['total_loss'],
            'disc_accuracy': disc_metrics_sum['accuracy'],
            'vol_ratio': vol_ratio,
            'val_loss': val_loss
        })

    print(f"\n✅ IMPROVED Training completed!")
    print(f"🎯 Best volatility ratio achieved: {best_vol_ratio:.3f}")

    if best_vol_ratio > 0.7:
        print(f"🏆 EXCELLENT: Volatility target achieved!")
    elif best_vol_ratio > 0.6:
        print(f"✅ GOOD: Strong improvement over baseline")
    else:
        print(f"📊 MODERATE: Some improvement, may need further tuning")

    # Save training history
    with open(f'{run_name}_history.json', 'w') as f:
        json.dump(training_history, f)

    return best_vol_ratio


if __name__ == "__main__":
    best_ratio = improved_adversarial_training()
    print(f"\nFinal best volatility ratio: {best_ratio:.3f}")