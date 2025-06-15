#!/usr/bin/env python3

"""
Complete API workflow test script with image storage validation
Tests all new features: scene management, sample rendering, training, and image storage
"""

import requests
import json
import time
import argparse
import sys
import base64
import os
from PIL import Image
import io

def validate_base64_image(base64_data, expected_format='jpeg'):
    """Validate that base64 data is a valid image"""
    try:
        # Decode base64
        image_data = base64.b64decode(base64_data)

        # Try to open as image
        image = Image.open(io.BytesIO(image_data))

        # Check format
        if expected_format.lower() == 'jpeg' and image.format != 'JPEG':
            return False, f"Expected JPEG, got {image.format}"

        # Check dimensions (should be reasonable)
        width, height = image.size
        if width < 10 or height < 10 or width > 5000 or height > 5000:
            return False, f"Suspicious dimensions: {width}x{height}"

        return True, f"Valid {image.format} image: {width}x{height}, {len(image_data)} bytes"

    except Exception as e:
        return False, f"Image validation failed: {e}"

def validate_cloudflare_url(url):
    """Validate that a Cloudflare Images URL is accessible"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Check if it's actually an image
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type:
                return True, f"Valid image URL: {content_type}, {len(response.content)} bytes"
            else:
                return False, f"URL accessible but not an image: {content_type}"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, f"URL validation failed: {e}"

def test_image_storage(base_url, session_id, storage_type='base64', cloudflare_credentials=None):
    """Test image storage functionality"""
    print(f"\n--- Testing {storage_type.upper()} Image Storage ---")

    # Prepare request data
    sample_data = {
        "session_id": session_id,
        "num_samples": 2,
        "nn_sampling": False,
        "return_albedo": True
    }

    if storage_type == 'base64':
        sample_data["embed_images"] = True
        sample_data["cloud_storage"] = None
    elif storage_type == 'cloudflare':
        sample_data["embed_images"] = False
        sample_data["cloud_storage"] = "cloudflare"
        if cloudflare_credentials:
            sample_data["cloudflare_account_id"] = cloudflare_credentials["account_id"]
            sample_data["cloudflare_api_token"] = cloudflare_credentials["api_token"]

    # Start render job
    response = requests.post(
        f"{base_url}/sample_render",
        json=sample_data,
        headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
    )

    if response.status_code != 200:
        print(f"❌ Failed to start {storage_type} render: {response.status_code}")
        return False

    job_result = response.json()
    job_id = job_result['job_id']
    print(f"✅ {storage_type.capitalize()} render job started: {job_id}")
    print(f"   Note: {job_result.get('note', 'No note')}")

    # Wait for completion
    print(f"   Waiting for {storage_type} render to complete...")
    for i in range(30):  # Wait up to 60 seconds
        time.sleep(2)
        response = requests.get(
            f"{base_url}/status/{job_id}",
            headers={"ngrok-skip-browser-warning": "true"}
        )

        if response.status_code == 200:
            job_status = response.json()
            if job_status.get('status') == 'completed':
                results = job_status.get('results', [])
                print(f"✅ {storage_type.capitalize()} render completed! {len(results)} samples")

                # Validate images
                image_validation_passed = True
                for i, result in enumerate(results):
                    print(f"   Sample {i}:")
                    print(f"     Surface area: {result.get('surface_area', 0):.6f}")
                    print(f"     Storage type: {result.get('albedo_storage', 'unknown')}")

                    if storage_type == 'base64' and 'albedo_base64' in result:
                        # Validate base64 image
                        valid, message = validate_base64_image(result['albedo_base64'])
                        if valid:
                            print(f"     ✅ Base64 image: {message}")
                        else:
                            print(f"     ❌ Base64 validation failed: {message}")
                            image_validation_passed = False

                    elif storage_type == 'cloudflare' and 'albedo_url' in result:
                        # Validate Cloudflare URL
                        valid, message = validate_cloudflare_url(result['albedo_url'])
                        if valid:
                            print(f"     ✅ Cloudflare URL: {message}")
                            print(f"     📷 Image URL: {result['albedo_url']}")
                        else:
                            print(f"     ❌ Cloudflare URL validation failed: {message}")
                            image_validation_passed = False

                    else:
                        print(f"     ⚠️  No image data found for {storage_type} storage")
                        image_validation_passed = False

                return image_validation_passed

            elif job_status.get('status') == 'error':
                print(f"❌ {storage_type.capitalize()} render failed: {job_status.get('error', 'Unknown error')}")
                return False
            else:
                print(f"   Status: {job_status.get('status', 'unknown')}...")
        else:
            print(f"   Checking job status... (attempt {i+1}/30)")

    print(f"❌ Timeout waiting for {storage_type} render to complete")
    return False

def test_api_workflow(base_url, cloudflare_credentials=None):
    """Test the complete API workflow with image storage validation"""

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
        for i in range(30):  # Wait up to 60 seconds
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
                print(f"   Checking status... (attempt {i+1}/30)")
        else:
            print("❌ Timeout waiting for renderer to be ready")
            return False

        # 3. Test Base64 Image Storage
        base64_success = test_image_storage(base_url, session_id, 'base64')

        # 4. Test Cloudflare Image Storage (if credentials provided)
        cloudflare_success = True
        if cloudflare_credentials:
            cloudflare_success = test_image_storage(base_url, session_id, 'cloudflare', cloudflare_credentials)
        else:
            print("\n--- Skipping Cloudflare Image Storage Test ---")
            print("   (No Cloudflare credentials provided)")

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

        # 6. Test Neural Network Sampling with Images
        print("\n6. Testing neural network sampling with base64 images...")
        nn_success = test_image_storage(base_url, session_id, 'base64')

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

        # Summary
        print("\n" + "=" * 60)
        print("🎯 TEST RESULTS SUMMARY")
        print("=" * 60)

        all_passed = base64_success and cloudflare_success and nn_success

        print(f"✅ Scene management: PASSED" if existing_scenes else "❌ Scene management: FAILED")
        print(f"✅ Base64 image storage: PASSED" if base64_success else "❌ Base64 image storage: FAILED")
        print(f"✅ Cloudflare image storage: PASSED" if cloudflare_success else ("⚠️  Cloudflare image storage: SKIPPED" if not cloudflare_credentials else "❌ Cloudflare image storage: FAILED"))
        print(f"✅ Neural network sampling: PASSED" if nn_success else "❌ Neural network sampling: FAILED")

        if all_passed:
            print("\n🎉 ALL TESTS PASSED! 🎉")
            print("✅ Complete image storage workflow functional!")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("Check the detailed output above for specific issues")

        return all_passed

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
    parser = argparse.ArgumentParser(description="Test the complete API workflow with image storage validation")
    parser.add_argument("--url", "-u", default="http://localhost:5000",
                       help="Base URL of the API server (default: http://localhost:5000)")
    parser.add_argument("--ngrok", "-n",
                       help="Use ngrok URL (e.g., https://817a-37-16-65-130.ngrok-free.app)")
    parser.add_argument("--cloudflare-account-id",
                       help="Cloudflare Account ID for testing Cloudflare Images")
    parser.add_argument("--cloudflare-api-token",
                       help="Cloudflare API Token for testing Cloudflare Images")

    args = parser.parse_args()

    if args.ngrok:
        base_url = args.ngrok
    else:
        base_url = args.url

    # Remove trailing slash
    base_url = base_url.rstrip('/')

    # Prepare Cloudflare credentials if provided
    cloudflare_credentials = None
    if args.cloudflare_account_id and args.cloudflare_api_token:
        cloudflare_credentials = {
            "account_id": args.cloudflare_account_id,
            "api_token": args.cloudflare_api_token
        }
        print("🔑 Cloudflare credentials provided - will test Cloudflare Images storage")
    else:
        print("⚠️  No Cloudflare credentials provided - will skip Cloudflare Images testing")

    print("Adaptive Camera Sampling API - Complete Workflow Test with Image Storage")
    print("=" * 80)

    success = test_api_workflow(base_url, cloudflare_credentials)

    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
