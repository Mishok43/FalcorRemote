#!/usr/bin/env python3

"""
Complete API workflow test script
Tests all new features: scene management, sample rendering, and training
"""

import requests
import json
import time
import argparse
import sys

def test_api_workflow(base_url):
    """Test the complete API workflow"""

    print(f"Testing API at: {base_url}")
    print("=" * 60)

    session_id = None

    try:
        # 1. Get available scenes
        print("1. Getting available scenes...")
        response = requests.get(f"{base_url}/scenes", headers={"ngrok-skip-browser-warning": "true"})
        if response.status_code != 200:
            print(f"❌ Failed to get scenes: {response.status_code}")
            return False

        scenes_data = response.json()
        print(f"✅ Found {scenes_data['total_scenes']} scenes")
        existing_scenes = list(scenes_data['existing_scenes'].keys())
        print(f"   Existing scenes: {existing_scenes}")

        if not existing_scenes:
            print("❌ No existing scenes found!")
            return False

        # 2. Setup renderer
        print("\n2. Setting up renderer...")
        setup_data = {
            "render_width": 512,
            "render_height": 512,
            "use_nf_sampler": True,
            "train_batch_size": 20,
            "verbose": True
        }

        response = requests.post(
            f"{base_url}/setup_renderer",
            json=setup_data,
            headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
        )

        if response.status_code != 200:
            print(f"❌ Failed to setup renderer: {response.status_code}")
            return False

        setup_result = response.json()
        session_id = setup_result['session_id']
        print(f"✅ Renderer setup started, session ID: {session_id}")
        # Wait for renderer to be ready
        print("   Waiting for renderer to be ready...")
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(2)
            response = requests.get(
                f"{base_url}/status/{session_id}",
                headers={"ngrok-skip-browser-warning": "true"}
            )

            if response.status_code == 200:
                status_data = response.json()
                if status_data.get('status') == 'ready':
                    print("✅ Renderer is ready!")
                    break
                elif status_data.get('status') == 'error':
                    print(f"❌ Renderer setup failed: {status_data.get('error', 'Unknown error')}")
                    return False
                else:
                    print(f"   Status: {status_data.get('status', 'unknown')}...")
            else:
                print(f"   Checking status... (attempt {i+1}/15)")
        else:
            print("❌ Timeout waiting for renderer to be ready")
            return False

        # 3. Set scene
        print(f"\n3. Setting scene to: {existing_scenes[0]}")
        scene_data = {
            "session_id": session_id,
            "scene_name": existing_scenes[0]
        }


        response = requests.post(
            f"{base_url}/set_scene",
            json=scene_data,
            headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
        )

        if response.status_code != 200:
            print(f"❌ Failed to set scene: {response.status_code}")
            return False

        scene_result = response.json()
        print(f"✅ Scene '{existing_scenes[0]}' loaded successfully")
        print(f"   Scene bounds: {scene_result.get('scene_bounds', {})}")

        # 4. Sample render (random sampling)
        print("\n4. Testing sample render (random sampling)...")
        sample_data = {
            "session_id": session_id,
            "num_samples": 2,
            "nn_sampling": False,
            "return_albedo": False  # Disable albedo to avoid memory issues
        }

        response = requests.post(
            f"{base_url}/sample_render",
            json=sample_data,
            headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
        )

        if response.status_code != 200:
            print(f"❌ Failed to start sample render: {response.status_code}")
            return False

        sample_result = response.json()
        job_id = sample_result['job_id']
        print(f"✅ Sample render job started: {job_id}")

        # Wait for sample render to complete
        print("   Waiting for sample render to complete...")
        for i in range(20):  # Wait up to 40 seconds
            time.sleep(2)
            response = requests.get(
                f"{base_url}/status/{job_id}",
                headers={"ngrok-skip-browser-warning": "true"}
            )

            if response.status_code == 200:
                job_status = response.json()
                if job_status.get('status') == 'completed':
                    results = job_status.get('results', [])
                    print(f"✅ Sample render completed! {len(results)} samples")

                    # Show sample results
                    for i, result in enumerate(results):
                        pos = result['camera_position_absolute']
                        area = result['surface_area']
                        method = result['sampling_method']
                        print(f"   Sample {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                              f"area={area:.6f}, method={method}")
                    break
                elif job_status.get('status') == 'error':
                    print(f"❌ Sample render failed: {job_status.get('error', 'Unknown error')}")
                    return False
                else:
                    print(f"   Status: {job_status.get('status', 'unknown')}...")
            else:
                print(f"   Checking job status... (attempt {i+1}/20)")
        else:
            print("❌ Timeout waiting for sample render to complete")
            return False

        # 5. Train NFSampler
        print("\n5. Training NFSampler...")
        train_data = {
            "session_id": session_id,
            "num_steps": 20,
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs_per_fit": 10
        }

        response = requests.post(
            f"{base_url}/train",
            json=train_data,
            headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
        )

        if response.status_code != 200:
            print(f"❌ Failed to start training: {response.status_code}")
            return False

        train_result = response.json()
        train_job_id = train_result['job_id']
        print(f"✅ Training job started: {train_job_id}")
        print(f"   Training parameters: {train_result.get('training_params', {})}")

        # Wait for training to complete
        print("   Waiting for training to complete...")
        for i in range(30):  # Wait up to 60 seconds
            time.sleep(2)
            response = requests.get(
                f"{base_url}/status/{train_job_id}",
                headers={"ngrok-skip-browser-warning": "true"}
            )

            if response.status_code == 200:
                train_status = response.json()
                if train_status.get('status') == 'completed':
                    train_info = train_status.get('training_result', {})
                    print(f"✅ Training completed!")
                    print(f"   Steps: {train_info.get('num_steps_requested', 0)}")
                    print(f"   Samples added: {train_info.get('num_samples_added', 0)}")
                    print(f"   Duration: {train_info.get('training_duration_seconds', 0):.2f}s")
                    print(f"   NFSampler trained: {train_info.get('is_nf_trained', False)}")
                    break
                elif train_status.get('status') == 'error':
                    print(f"❌ Training failed: {train_status.get('error', 'Unknown error')}")
                    return False
                else:
                    print(f"   Status: {train_status.get('status', 'unknown')}...")
            else:
                print(f"   Checking training status... (attempt {i+1}/30)")
        else:
            print("❌ Timeout waiting for training to complete")
            return False

        # 6. Sample render with neural network
        print("\n6. Testing sample render (neural network sampling)...")
        nn_sample_data = {
            "session_id": session_id,
            "num_samples": 2,
            "nn_sampling": True,
            "return_albedo": False
        }

        response = requests.post(
            f"{base_url}/sample_render",
            json=nn_sample_data,
            headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
        )

        if response.status_code != 200:
            print(f"❌ Failed to start NN sample render: {response.status_code}")
            return False

        nn_sample_result = response.json()
        nn_job_id = nn_sample_result['job_id']
        print(f"✅ NN sample render job started: {nn_job_id}")

        # Wait for NN sample render to complete
        print("   Waiting for NN sample render to complete...")
        for i in range(20):  # Wait up to 40 seconds
            time.sleep(2)
            response = requests.get(
                f"{base_url}/status/{nn_job_id}",
                headers={"ngrok-skip-browser-warning": "true"}
            )

            if response.status_code == 200:
                nn_job_status = response.json()
                if nn_job_status.get('status') == 'completed':
                    nn_results = nn_job_status.get('results', [])
                    print(f"✅ NN sample render completed! {len(nn_results)} samples")

                    # Show NN sample results
                    for i, result in enumerate(nn_results):
                        pos = result['camera_position_absolute']
                        area = result['surface_area']
                        method = result['sampling_method']
                        print(f"   Sample {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                              f"area={area:.6f}, method={method}")
                    break
                elif nn_job_status.get('status') == 'error':
                    print(f"❌ NN sample render failed: {nn_job_status.get('error', 'Unknown error')}")
                    return False
                else:
                    print(f"   Status: {nn_job_status.get('status', 'unknown')}...")
            else:
                print(f"   Checking NN job status... (attempt {i+1}/20)")
        else:
            print("❌ Timeout waiting for NN sample render to complete")
            return False

        # 7. Final session status
        print("\n7. Final session status...")
        response = requests.get(
            f"{base_url}/status/{session_id}",
            headers={"ngrok-skip-browser-warning": "true"}
        )

        if response.status_code == 200:
            final_status = response.json()
            print(f"✅ Session {session_id} final status:")
            print(f"   Current scene: {final_status.get('current_scene', 'None')}")
            print(f"   Total samples rendered: {final_status.get('total_samples_rendered', 0)}")
            print(f"   Total training steps: {final_status.get('total_training_steps', 0)}")

        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("✅ Scene management working")
        print("✅ Sample rendering working")
        print("✅ Custom training working")
        print("✅ Neural network sampling working")
        print("✅ All new API features functional!")

        return True

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if session_id:
            print(f"\n8. Cleaning up session {session_id}...")
            try:
                requests.post(
                    f"{base_url}/cleanup",
                    json={"session_id": session_id},
                    headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
                )
                print("✅ Session cleaned up")
            except:
                print("⚠️  Cleanup failed (server may have stopped)")

def main():
    parser = argparse.ArgumentParser(description="Test the complete API workflow")
    parser.add_argument("--url", "-u", default="http://localhost:5000",
                       help="Base URL of the API server (default: http://localhost:5000)")
    parser.add_argument("--ngrok", "-n",
                       help="Use ngrok URL (e.g., https://817a-37-16-65-130.ngrok-free.app)")

    args = parser.parse_args()

    if args.ngrok:
        base_url = args.ngrok
    else:
        base_url = args.url

    # Remove trailing slash
    base_url = base_url.rstrip('/')

    print("Adaptive Camera Sampling API - Complete Workflow Test")
    print("=" * 60)

    success = test_api_workflow(base_url)

    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
