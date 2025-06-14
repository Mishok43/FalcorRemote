#!/usr/bin/env python3

"""
Test script for the new API features:
1. Get scene list
2. Set current scene
3. Sample render with albedo
4. Training with custom parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from basic_rasterizer import BasicRasterizer
import numpy as np
import time

def test_scene_management():
    """Test scene list and scene setting"""
    print("=== Testing Scene Management ===")

    renderer = BasicRasterizer(verbose=True)
    renderer.setup_falcor()

    # Test get available scenes
    scenes = renderer.get_available_scenes()
    print(f"Available scenes: {len(scenes)}")

    existing_scenes = [name for name, info in scenes.items() if info['exists']]
    print(f"Existing scenes: {existing_scenes}")

    if not existing_scenes:
        print("No scenes found! Please check scene file paths.")
        return False

    # Test set current scene
    scene_name = existing_scenes[0]
    print(f"Setting scene to: {scene_name}")

    success = renderer.set_current_scene(scene_name)
    if success:
        renderer.setup_render_graph()
        print(f"✓ Scene '{scene_name}' loaded successfully")
        print(f"Scene bounds: {renderer.scene_bounds}")
        return renderer, scene_name
    else:
        print(f"✗ Failed to load scene '{scene_name}'")
        return False

def test_sample_render(renderer):
    """Test sample rendering with albedo"""
    print("\n=== Testing Sample Render ===")

    # Test random sampling
    print("Testing random sampling...")
    random_results = renderer.sample_render_positions(
        num_samples=3,
        nn_sampling=False,
        return_albedo=True
    )

    print(f"Random sampling results: {len(random_results)} samples")
    for i, result in enumerate(random_results):
        pos = result['camera_position_absolute']
        dir = result['camera_direction_absolute']
        area = result['surface_area']
        method = result['sampling_method']
        has_albedo = 'albedo_render' in result

        print(f"  Sample {i} ({method}): pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
              f"area={area:.6f}, albedo={has_albedo}")

    return random_results

def test_training(renderer):
    """Test NFSampler training with custom parameters"""
    print("\n=== Testing NFSampler Training ===")

    # Test training with custom parameters
    training_result = renderer.train_nf_sampler_steps(
        num_steps=20,
        learning_rate=0.001,
        epochs_per_fit=10,
        batch_size=16
    )

    print("Training result:")
    for key, value in training_result.items():
        if key == 'training_parameters_used':
            print(f"  {key}:")
            for param, param_value in value.items():
                print(f"    {param}: {param_value}")
        else:
            print(f"  {key}: {value}")

    return training_result

def test_nn_sampling(renderer):
    """Test neural network sampling after training"""
    print("\n=== Testing Neural Network Sampling ===")

    if not renderer.is_nf_trained:
        print("NFSampler not trained, skipping NN sampling test")
        return None

    # Test NN sampling
    print("Testing neural network sampling...")
    nn_results = renderer.sample_render_positions(
        num_samples=3,
        nn_sampling=True,
        return_albedo=False  # Skip albedo for faster testing
    )

    print(f"NN sampling results: {len(nn_results)} samples")
    for i, result in enumerate(nn_results):
        pos = result['camera_position_absolute']
        dir = result['camera_direction_absolute']
        area = result['surface_area']
        method = result['sampling_method']

        print(f"  Sample {i} ({method}): pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
              f"area={area:.6f}")

    return nn_results

def main():
    """Run all tests"""
    print("Testing New API Features")
    print("=" * 50)

    try:
        # Test 1: Scene management
        result = test_scene_management()
        if not result:
            print("Scene management test failed!")
            return

        renderer, scene_name = result

        # Test 2: Sample rendering
        random_results = test_sample_render(renderer)

        # Test 3: Training
        training_result = test_training(renderer)

        # Test 4: NN sampling (if training succeeded)
        if training_result.get('is_nf_trained', False):
            nn_results = test_nn_sampling(renderer)

        print("\n=== All Tests Completed Successfully! ===")
        print(f"Scene used: {scene_name}")
        print(f"Random samples: {len(random_results)}")
        print(f"Training completed: {training_result.get('training_completed', False)}")
        print(f"NFSampler trained: {training_result.get('is_nf_trained', False)}")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
