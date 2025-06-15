#!/usr/bin/env python3

"""
Pure local test script for NFSampler weight saving and loading functionality
Tests both direct NFSampler usage and BasicRasterizer integration
No API calls or requests - pure local testing
"""

import os
import sys
import tempfile
import numpy as np
import argparse
import time
from pathlib import Path

def test_direct_nf_sampler():
    """Test NFSampler save/load functionality directly"""
    print("=" * 60)
    print("Testing Direct NFSampler Save/Load")
    print("=" * 60)

    try:
        from nf_sampler import NFSampler
        import torch

        # Create temporary directory for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "test_nf_weights.pth")

            print("1. Creating and training original NFSampler...")

            # Create original sampler
            original_sampler = NFSampler(
                name="test_sampler",
                rng=None,
                num_flows=2,  # Smaller for faster testing
                hidden_units=32,
                hidden_layers=2,
                learning_rate=1e-3,
                epochs_per_fit=5,  # Fewer epochs for faster testing
                batch_size=16,
                history_size=100,
                latent_size=3,  # 3D test case
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # Register dimensions
            for _ in range(3):
                original_sampler.register_dimension()

            # Initialize
            original_sampler.reinitialize()

            # Generate some training data
            print("   Generating training data...")
            x_data = []
            x_pdf_data = []
            y_data = []

            for i in range(50):
                # Random 3D point
                x = np.random.uniform(0, 1, 3).tolist()
                # Uniform PDF
                pdf = 1.0
                # Synthetic performance (lower values near center)
                center = np.array([0.5, 0.5, 0.5])
                distance = np.linalg.norm(np.array(x) - center)
                performance = distance + 0.1 * np.random.randn()  # Add noise

                x_data.append(x)
                x_pdf_data.append(pdf)
                y_data.append(performance)

            # Train original sampler
            print("   Training original sampler...")
            original_sampler.add_data(x_data, x_pdf_data, y_data)
            original_sampler.fit()

            # Sample from original
            print("   Sampling from original...")
            original_samples = original_sampler.sample_primary(num_samples=10)
            original_likelihoods = [s.pdf for s in original_samples]

            print(f"   Original sampler trained with {len(x_data)} samples")
            print(f"   Original sample likelihoods: {[f'{l:.6f}' for l in original_likelihoods[:3]]}...")

            # Save weights
            print("\n2. Saving weights...")
            original_sampler.save_weights(weights_path)

            # Verify file exists
            if not os.path.exists(weights_path):
                print("❌ Weights file was not created!")
                return False

            file_size = os.path.getsize(weights_path) / 1024  # KB
            print(f"✅ Weights saved successfully ({file_size:.1f} KB)")

            # Create new sampler and load weights
            print("\n3. Creating new sampler and loading weights...")

            # Method 1: Load into existing sampler
            new_sampler = NFSampler(
                name="loaded_sampler",
                rng=None,
                num_flows=2,
                hidden_units=32,
                hidden_layers=2,
                learning_rate=1e-3,
                epochs_per_fit=5,
                batch_size=16,
                history_size=100,
                latent_size=3,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # Register dimensions
            for _ in range(3):
                new_sampler.register_dimension()

            # Load weights
            new_sampler.load_weights(weights_path, load_training_data=True)

            # Sample from loaded sampler
            print("   Sampling from loaded sampler...")
            loaded_samples = new_sampler.sample_primary(num_samples=10)
            loaded_likelihoods = [s.pdf for s in loaded_samples]

            print(f"   Loaded sampler has {new_sampler.all_x.shape[0] if new_sampler.all_x is not None else 0} training samples")
            print(f"   Loaded sample likelihoods: {[f'{l:.6f}' for l in loaded_likelihoods[:3]]}...")

            # Method 2: Load from file directly
            print("\n4. Testing load_from_file class method...")
            direct_loaded_sampler = NFSampler.load_from_file(weights_path)

            direct_samples = direct_loaded_sampler.sample_primary(num_samples=10)
            direct_likelihoods = [s.pdf for s in direct_samples]

            print(f"   Direct loaded sampler has {direct_loaded_sampler.all_x.shape[0] if direct_loaded_sampler.all_x is not None else 0} training samples")
            print(f"   Direct loaded sample likelihoods: {[f'{l:.6f}' for l in direct_likelihoods[:3]]}...")

            # Compare distributions
            print("\n5. Comparing distributions...")

            # Test same input on all samplers
            test_input = [0.3, 0.7, 0.5]
            original_likelihood = original_sampler.get_likelihood(test_input)
            loaded_likelihood = new_sampler.get_likelihood(test_input)
            direct_likelihood = direct_loaded_sampler.get_likelihood(test_input)

            print(f"   Test input {test_input}:")
            print(f"   Original likelihood: {original_likelihood:.6f}")
            print(f"   Loaded likelihood: {loaded_likelihood:.6f}")
            print(f"   Direct loaded likelihood: {direct_likelihood:.6f}")

            # Check if likelihoods are similar (within tolerance)
            tolerance = 1e-5
            if (abs(original_likelihood - loaded_likelihood) < tolerance and
                abs(original_likelihood - direct_likelihood) < tolerance):
                print("✅ Likelihoods match within tolerance!")
                return True
            else:
                print("❌ Likelihoods don't match!")
                print(f"   Difference (original vs loaded): {abs(original_likelihood - loaded_likelihood):.8f}")
                print(f"   Difference (original vs direct): {abs(original_likelihood - direct_likelihood):.8f}")
                return False

    except Exception as e:
        print(f"❌ Direct NFSampler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_rasterizer_integration():
    """Test NFSampler save/load through BasicRasterizer"""
    print("\n" + "=" * 60)
    print("Testing BasicRasterizer NFSampler Integration")
    print("=" * 60)

    try:
        from basic_rasterizer import BasicRasterizer
        import torch

        # Create temporary directory for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "rasterizer_nf_weights.pth")

            print("1. Creating BasicRasterizer with NFSampler...")

            # Create renderer with NFSampler enabled
            renderer = BasicRasterizer(
                render_width=256,
                render_height=256,
                verbose=False,  # Reduce output for testing
                use_nf_sampler=True,
                initial_samples=20,
                train_batch_size=15,
                predetermined_batch_size=10,
                max_training_history=100
            )

            print("   ✅ BasicRasterizer created")

            # Check NFSampler initialization
            if renderer.nf_sampler is None:
                print("❌ NFSampler not initialized!")
                return False

            print(f"   NFSampler dimensions: {renderer.nf_sampler.total_dimensions}")
            print(f"   NFSampler latent size: {renderer.nf_sampler.latent_size}")

            # Simulate training data (without actual rendering)
            print("\n2. Adding synthetic training data...")

            # Generate synthetic 5D training data (3D position + 2D spherical direction)
            for i in range(30):
                # Random 5D input (normalized position + spherical direction)
                input_5d = np.random.uniform(0, 1, 5)

                # Synthetic surface area (higher near center of position space)
                pos_center = np.array([0.5, 0.5, 0.5])
                pos_distance = np.linalg.norm(input_5d[:3] - pos_center)
                surface_area = 1.0 - pos_distance + 0.1 * np.random.randn()  # Higher is better

                # Uniform PDF
                sample_pdf = 1.0

                # Add to training data
                renderer.add_training_data(input_5d, surface_area, sample_pdf)

            print(f"   Added {len(renderer.training_data)} training samples")

            # Train NFSampler
            print("\n3. Training NFSampler...")
            renderer.train_nf_sampler()

            if not renderer.is_nf_trained:
                print("❌ NFSampler training failed!")
                return False

            print("   ✅ NFSampler training completed")

            # Get training stats
            print("\n4. Getting training statistics...")
            stats = renderer.get_nf_training_stats()

            print(f"   Is initialized: {stats['is_initialized']}")
            print(f"   Is trained: {stats['is_trained']}")
            print(f"   Training samples: {stats['training_samples']}")
            print(f"   Performance stats: {stats.get('performance_stats', {})}")

            # Sample from trained NFSampler
            print("\n5. Sampling from trained NFSampler...")
            original_samples = []
            for i in range(5):
                camera_pos, camera_dir, input_5d, sample_pdf = renderer.sample_camera_params_nf()
                original_samples.append({
                    'input_5d': input_5d.tolist(),
                    'pdf': sample_pdf,
                    'position': camera_pos.tolist(),
                    'direction': camera_dir.tolist()
                })

            print(f"   Generated {len(original_samples)} samples")


            # Save weights
            print("\n6. Saving NFSampler weights...")
            renderer.save_nf_weights(weights_path)

            # Verify file exists
            if not os.path.exists(weights_path):
                print("❌ Weights file was not created!")
                return False

            file_size = os.path.getsize(weights_path) / 1024  # KB
            print(f"   ✅ Weights saved successfully ({file_size:.1f} KB)")

            # Create new renderer and load weights
            print("\n7. Creating new renderer and loading weights...")

            new_renderer = BasicRasterizer(
                render_width=256,
                render_height=256,
                verbose=False,
                use_nf_sampler=True,
                initial_samples=20,
                train_batch_size=15,
                predetermined_batch_size=10,
                max_training_history=100
            )

            # Load weights
            new_renderer.load_nf_weights(weights_path, load_training_data=True)

            print(f"   ✅ Weights loaded into new renderer")
            print(f"   New renderer is_nf_trained: {new_renderer.is_nf_trained}")
            print(f"   New renderer training data: {len(new_renderer.training_data)} samples")

            # Get stats from loaded renderer
            loaded_stats = new_renderer.get_nf_training_stats()
            print(f"   Loaded training samples: {loaded_stats['training_samples']}")

            # Sample from loaded NFSampler
            print("\n8. Sampling from loaded NFSampler...")
            loaded_samples = []
            for i in range(5):
                camera_pos, camera_dir, input_5d, sample_pdf = new_renderer.sample_camera_params_nf()
                loaded_samples.append({
                    'input_5d': input_5d.tolist(),
                    'pdf': sample_pdf,
                    'position': camera_pos.tolist(),
                    'direction': camera_dir.tolist()
                })

            print(f"   Generated {len(loaded_samples)} samples")


            # Compare results
            print("\n9. Comparing results...")

            # Compare training sample counts
            original_count = stats['training_samples']
            loaded_count = loaded_stats['training_samples']

            if original_count == loaded_count:
                print(f"   ✅ Training sample counts match: {original_count}")
            else:
                print(f"   ❌ Training sample counts don't match: {original_count} vs {loaded_count}")
                return False

            # Compare likelihood for same input
            test_input_5d = [0.3, 0.7, 0.5, 0.2, 0.8]
            original_likelihood = renderer.nf_sampler.get_likelihood(test_input_5d)
            loaded_likelihood = new_renderer.nf_sampler.get_likelihood(test_input_5d)

            print(f"   Test input: {test_input_5d}")
            print(f"   Original likelihood: {original_likelihood:.6f}")
            print(f"   Loaded likelihood: {loaded_likelihood:.6f}")

            # Check if likelihoods match
            tolerance = 1e-5
            if abs(original_likelihood - loaded_likelihood) < tolerance:
                print("   ✅ Likelihoods match within tolerance!")
                return True
            else:
                print(f"   ❌ Likelihoods don't match! Difference: {abs(original_likelihood - loaded_likelihood):.8f}")
                return False

    except Exception as e:
        print(f"❌ BasicRasterizer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_conditions():
    """Test error conditions and edge cases"""
    print("\n" + "=" * 60)
    print("Testing Error Conditions and Edge Cases")
    print("=" * 60)

    try:
        from nf_sampler import NFSampler
        from basic_rasterizer import BasicRasterizer
        import torch

        print("1. Testing save before initialization...")
        sampler = NFSampler(latent_size=2)
        try:
            sampler.save_weights("test.pth")
            print("   ❌ Should have failed!")
            return False
        except Exception as e:
            print(f"   ✅ Correctly failed: {type(e).__name__}")

        print("\n2. Testing load non-existent file...")
        sampler.register_dimension()
        sampler.register_dimension()
        sampler.reinitialize()

        try:
            sampler.load_weights("non_existent_file.pth")
            print("   ❌ Should have failed!")
            return False
        except FileNotFoundError:
            print("   ✅ Correctly failed: FileNotFoundError")

        print("\n3. Testing dimension mismatch...")
        # Create and save a 2D sampler
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "2d_weights.pth")

            # Save 2D sampler
            sampler.save_weights(weights_path)

            # Try to load into 3D sampler
            sampler_3d = NFSampler(latent_size=3)
            for _ in range(3):
                sampler_3d.register_dimension()
            sampler_3d.reinitialize()

            try:
                sampler_3d.load_weights(weights_path)
                print("   ❌ Should have failed!")
                return False
            except ValueError as e:
                print(f"   ✅ Correctly failed: {type(e).__name__}")

        print("\n4. Testing BasicRasterizer without NFSampler...")
        renderer = BasicRasterizer(use_nf_sampler=False, verbose=False)

        try:
            renderer.save_nf_weights("test.pth")
            print("   ❌ Should have failed!")
            return False
        except Exception as e:
            print(f"   ✅ Correctly failed: {type(e).__name__}")

        print("\n5. Testing load_from_file with invalid file...")
        try:
            NFSampler.load_from_file("non_existent.pth")
            print("   ❌ Should have failed!")
            return False
        except FileNotFoundError:
            print("   ✅ Correctly failed: FileNotFoundError")

        print("\n✅ All error condition tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error condition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test NFSampler weight saving and loading (pure local)")
    parser.add_argument("--skip-direct", action="store_true",
                       help="Skip direct NFSampler testing")
    parser.add_argument("--skip-integration", action="store_true",
                       help="Skip BasicRasterizer integration testing")
    parser.add_argument("--skip-errors", action="store_true",
                       help="Skip error condition testing")

    args = parser.parse_args()

    print("NFSampler Weight Save/Load Test (Pure Local)")
    print("=" * 80)

    success = True

    # Test direct NFSampler functionality
    if not args.skip_direct:
        direct_success = test_direct_nf_sampler()
        success = success and direct_success
    else:
        print("Skipping direct NFSampler test")

    # Test BasicRasterizer integration
    if not args.skip_integration:
        integration_success = test_basic_rasterizer_integration()
        success = success and integration_success
    else:
        print("Skipping BasicRasterizer integration test")

    # Test error conditions
    if not args.skip_errors:
        error_success = test_error_conditions()
        success = success and error_success
    else:
        print("Skipping error condition test")

    # Final results
    print("\n" + "=" * 80)
    print("🎯 FINAL TEST RESULTS")
    print("=" * 80)

    if success:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("✅ NFSampler weight save/load functionality is working correctly")
        print("\nFeatures tested:")
        print("  ✅ Direct NFSampler save/load")
        print("  ✅ NFSampler.load_from_file() class method")
        print("  ✅ BasicRasterizer integration")
        print("  ✅ Training data persistence")
        print("  ✅ Model state preservation")
        print("  ✅ Error handling and validation")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("Check the detailed output above for specific issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
