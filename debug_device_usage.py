#!/usr/bin/env python3
"""
Debug GPU/CPU usage in quantum generator
"""

import torch
from adversarial_merlin_swaption import QuantumGenerator


def debug_device_usage():
    """Debug where the computational bottleneck is"""
    print("🔍 DEBUGGING DEVICE USAGE")
    print("="*50)

    # Device check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  PyTorch device: {device}")

    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create test input
    print(f"\n📊 Creating test input...")
    batch_size = 4
    X_test = torch.randn(batch_size, 60, 3).to(device)
    print(f"   Input tensor device: {X_test.device}")

    # Initialize generator
    print(f"\n🤖 Initializing QuantumGenerator...")
    generator = QuantumGenerator()

    # Check model device
    print(f"   Model parameters device: {next(generator.parameters()).device}")

    # Move to device
    generator = generator.to(device)
    print(f"   Model parameters after .to(device): {next(generator.parameters()).device}")

    # Test forward pass and monitor where time is spent
    print(f"\n⏱️  Testing forward pass...")
    import time

    # Test multiple passes to see consistent timing
    for i in range(3):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        with torch.no_grad():
            output = generator(X_test)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        print(f"   Pass {i+1}: {end_time - start_time:.3f}s, Output device: {output.device}")

    # Memory usage
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
        print(f"   Reserved:  {torch.cuda.memory_reserved(0) / 1e6:.1f} MB")

    return generator


def test_cpu_only_training():
    """Test if training works better on CPU only"""
    print(f"\n🧪 TESTING CPU-ONLY MODE")
    print("="*30)

    device = torch.device('cpu')
    print(f"🖥️  Forced CPU device: {device}")

    # Create test data
    X_test = torch.randn(8, 60, 3).to(device)
    Y_test = torch.randn(8, 14).to(device)

    # Initialize models
    generator = QuantumGenerator().to(device)

    # Quick training step
    print(f"🏃 Quick training step test...")
    import time
    start_time = time.time()

    output = generator(X_test)
    loss = torch.nn.MSELoss()(output, Y_test)
    print(f"   Forward pass: {time.time() - start_time:.3f}s")

    start_time = time.time()
    loss.backward()
    print(f"   Backward pass: {time.time() - start_time:.3f}s")

    print(f"   Loss: {loss.item():.6f}")
    print(f"   Output shape: {output.shape}")


def check_quandela_gpu_support():
    """Check if Quandela/Perceval supports GPU"""
    print(f"\n🔬 CHECKING QUANDELA GPU SUPPORT")
    print("="*40)

    try:
        # Try to access Quandela internals
        from merlinqnn.layers import QuantumLayer
        print("✅ QuantumLayer imported successfully")

        # Check if it has GPU support
        layer = QuantumLayer(input_size=2, builder=None, n_photons=2)
        print("✅ QuantumLayer instantiated")

        # Test device transfer
        test_input = torch.randn(2, 2)
        if torch.cuda.is_available():
            test_input_gpu = test_input.cuda()
            try:
                # This will likely fail
                result = layer(test_input_gpu)
                print(f"✅ QuantumLayer works on GPU: {result.device}")
            except Exception as e:
                print(f"❌ QuantumLayer GPU error: {e}")
                print("🔍 This explains why training is CPU-bound!")

    except Exception as e:
        print(f"❌ Quandela import error: {e}")


if __name__ == "__main__":
    # Debug device usage
    generator = debug_device_usage()

    # Test CPU-only
    test_cpu_only_training()

    # Check Quandela GPU support
    check_quandela_gpu_support()

    print(f"\n🎯 DIAGNOSIS:")
    print(f"   If QuantumLayer fails on GPU, then Quandela/Perceval")
    print(f"   only supports CPU, which explains the 100% CPU usage.")
    print(f"   Solution: Use CPU-only training or replace quantum layers.")