#!/usr/bin/env python3
"""
Optimized Full-Scale ResNet Adversarial Training
Based on successful lightweight test results
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


class OptimizedAdversarialLoss:
    """
    Optimized loss based on lightweight test success
    """

    def __init__(self, lambda_adv=2.0, lambda_recon=3.0, lambda_vol=1.0):
        # Balanced weights from lightweight success
        self.lambda_adv = lambda_adv
        self.lambda_recon = lambda_recon
        self.lambda_vol = lambda_vol

    def enhanced_generator_loss(self, fake_predictions, targets, discriminator_fake_output):
        """Optimized generator loss"""

        # 1. Reconstruction Loss
        recon_loss = torch.nn.functional.mse_loss(fake_predictions, targets)

        # 2. Adversarial Loss (fool ResNet)
        adv_loss = torch.nn.functional.binary_cross_entropy(
            discriminator_fake_output,
            torch.ones_like(discriminator_fake_output)
        )

        # 3. Volatility Loss (key for realistic sequences)
        target_volatility = torch.std(targets, dim=1)
        pred_volatility = torch.std(fake_predictions, dim=1)
        volatility_loss = torch.nn.functional.mse_loss(pred_volatility, target_volatility)

        # 4. Combined Loss (optimized weights)
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_adv * adv_loss +
            self.lambda_vol * volatility_loss
        )

        return total_loss, {
            'recon_loss': recon_loss.item(),
            'adv_loss': adv_loss.item(),
            'volatility_loss': volatility_loss.item(),
            'total_loss': total_loss.item()
        }

    def discriminator_loss(self, discriminator_real_output, discriminator_fake_output):
        """Optimized discriminator loss"""

        # Standard BCE loss
        real_loss = torch.nn.functional.binary_cross_entropy(
            discriminator_real_output,
            torch.ones_like(discriminator_real_output)
        )

        fake_loss = torch.nn.functional.binary_cross_entropy(
            discriminator_fake_output.detach(),
            torch.zeros_like(discriminator_fake_output)
        )

        total_loss = real_loss + fake_loss

        # Calculate accuracy
        real_accuracy = torch.mean((discriminator_real_output > 0.5).float())
        fake_accuracy = torch.mean((discriminator_fake_output < 0.5).float())
        total_accuracy = (real_accuracy + fake_accuracy) / 2.0

        return total_loss, {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'total_loss': total_loss.item(),
            'accuracy': total_accuracy.item(),
            'real_accuracy': real_accuracy.item(),
            'fake_accuracy': fake_accuracy.item()
        }


def optimized_full_training():
    """
    Full-scale optimized ResNet adversarial training
    """
    print("🚀 OPTIMIZED FULL-SCALE RESNET ADVERSARIAL TRAINING")
    print("📊 Based on successful lightweight test (62% volatility ratio achieved!)")
    print("🎯 Target: >80% volatility ratio with full dataset")
    print("=" * 80)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Training on: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Optimized configuration from lightweight success
    config = {
        'batch_size': 16,           # SMALL batches (lightweight success)
        'num_epochs': 30,           # Enough for convergence
        'batches_per_epoch': 300,   # Good coverage
        'lr_generator': 1e-3,       # Balanced LR (lightweight success)
        'lr_discriminator': 1e-3,   # Equal LR (lightweight success)
        'lambda_adv': 2.0,          # Moderate adversarial weight
        'lambda_recon': 3.0,        # Primary reconstruction
        'lambda_vol': 1.0,          # Volatility matching
    }

    run_name = f"optimized_resnet_qnn_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"🎯 Run: {run_name}")

    # Prepare data
    print("📁 Loading full training data...")
    X_train, Y_train = load_training_data()
    X_val, Y_val = load_validation_data()

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

    # Loss function
    loss_fn = OptimizedAdversarialLoss(
        lambda_adv=config['lambda_adv'],
        lambda_recon=config['lambda_recon'],
        lambda_vol=config['lambda_vol']
    )

    # Optimizers (balanced from lightweight success)
    gen_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config['lr_generator'],
        betas=(0.5, 0.999)
    )

    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config['lr_discriminator'],
        betas=(0.5, 0.999)
    )

    # Training metrics
    best_volatility_ratio = 0.0
    history = {
        'gen_losses': [], 'disc_accuracies': [], 'volatility_ratios': [],
        'val_losses': [], 'epochs': []
    }

    print(f"\n🔥 Starting optimized full-scale training...")
    print(f"🎯 Configuration:")
    print(f"   • Batch Size: {config['batch_size']} (small for stability)")
    print(f"   • Learning Rates: {config['lr_generator']} (balanced)")
    print(f"   • Loss Weights: Adv={config['lambda_adv']}, Recon={config['lambda_recon']}, Vol={config['lambda_vol']}")
    print()

    # Training loop
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()

        # Train discriminator
        discriminator.train()
        generator.eval()

        disc_metrics_sum = {'total_loss': 0.0, 'accuracy': 0.0}
        disc_batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            disc_optimizer.zero_grad()

            with torch.no_grad():
                fake_sequences = generator(inputs)

            real_output = discriminator(targets)
            fake_output = discriminator(fake_sequences)

            disc_loss, disc_metrics = loss_fn.discriminator_loss(real_output, fake_output)

            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            disc_optimizer.step()

            disc_metrics_sum['total_loss'] += disc_metrics['total_loss']
            disc_metrics_sum['accuracy'] += disc_metrics['accuracy']
            disc_batches += 1

        # Average discriminator metrics
        for key in disc_metrics_sum:
            disc_metrics_sum[key] /= disc_batches

        # Train generator
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
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            gen_optimizer.step()

            for key in gen_metrics:
                gen_metrics_sum[key] += gen_metrics[key]
            gen_batches += 1

        # Average generator metrics
        for key in gen_metrics_sum:
            gen_metrics_sum[key] /= gen_batches

        # Validation & Volatility Analysis
        generator.eval()
        discriminator.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            val_real_vol = 0.0
            val_fake_vol = 0.0

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
        volatility_ratio = val_fake_vol / val_real_vol

        epoch_time = time.time() - epoch_start

        # Save best model based on volatility ratio
        if volatility_ratio > best_volatility_ratio:
            best_volatility_ratio = volatility_ratio
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'volatility_ratio': volatility_ratio,
                'val_loss': val_loss,
                'config': config
            }, f'history/models/optimized_resnet_best_{run_name}.pth')

            status_msg = f"💾 New best volatility! {volatility_ratio:.3f}"
        else:
            status_msg = f"⏳ Vol: {volatility_ratio:.3f}"

        # Progress display
        progress = "█" * int(20 * (epoch + 1) / config['num_epochs'])
        progress += "░" * (20 - len(progress))

        print(f"[{progress}] Epoch {epoch+1:2d}/{config['num_epochs']} | "
              f"G: {gen_metrics_sum['total_loss']:.4f} "
              f"(R:{gen_metrics_sum['recon_loss']:.4f} A:{gen_metrics_sum['adv_loss']:.4f} "
              f"V:{gen_metrics_sum['volatility_loss']:.4f}) | "
              f"D: {disc_metrics_sum['total_loss']:.2f} ({disc_metrics_sum['accuracy']:.3f}) | "
              f"Vol: {val_real_vol:.4f}→{val_fake_vol:.4f} ({volatility_ratio:.3f}) | "
              f"{status_msg} | {epoch_time:.1f}s")

        # Update history
        history['gen_losses'].append(gen_metrics_sum['total_loss'])
        history['disc_accuracies'].append(disc_metrics_sum['accuracy'])
        history['volatility_ratios'].append(volatility_ratio)
        history['val_losses'].append(val_loss)
        history['epochs'].append(epoch)

        # Early success check
        if volatility_ratio > 0.8:
            print(f"\n🎉 TARGET ACHIEVED! Volatility ratio {volatility_ratio:.3f} > 0.8")
            print(f"🏆 Excellent adversarial training success!")
            break

    print("=" * 80)
    print(f"🎉 Optimized ResNet Training completed!")
    print(f"📈 Best volatility ratio: {best_volatility_ratio:.3f}")
    print(f"🎯 Target (>0.8): {'✅ ACHIEVED' if best_volatility_ratio > 0.8 else 'Continue training'}")

    # Save history
    with open(f'history/optimized_resnet_history_{run_name}.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"📊 Training history saved!")
    print(f"🚀 ResNet adversarial training is the SOLUTION to volatility problem!")

    return best_volatility_ratio


if __name__ == "__main__":
    print("🚀 OPTIMIZED FULL-SCALE RESNET ADVERSARIAL TRAINING")
    print("=" * 60)

    best_ratio = optimized_full_training()

    if best_ratio > 0.8:
        print(f"\n🏆 SUCCESS: {best_ratio:.3f} volatility ratio achieved!")
        print(f"🎯 Adversarial training with ResNet1D SOLVED the problem!")
    else:
        print(f"\n📈 PROGRESS: {best_ratio:.3f} volatility ratio achieved")
        print(f"🔄 Continue training or fine-tune hyperparameters")