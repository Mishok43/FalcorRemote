#!/usr/bin/env python3

"""
Example client for the Adaptive Camera Sampling API
"""

import requests
import json
import time
import numpy as np
from PIL import Image
import io

# API base URL
BASE_URL = "https://817a-37-16-65-130.ngrok-free.app/"

class AdaptiveCameraSamplingClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session_id = None

    def get_api_info(self):
        """Get API information and available endpoints"""
        response = requests.get(f"{self.base_url}/")
        print(response)
        return response.json()

    def get_available_scenes(self):
        """Get list of available scenes"""
        response = requests.get(f"{self.base_url}/scenes")
        return response.json()

    def setup_renderer(self, render_width=512, render_height=512, use_nf_sampler=True,
                      train_batch_size=50, enable_3d_viz=False, verbose=True):
        """Setup a renderer instance"""
        data = {
            "render_width": render_width,
            "render_height": render_height,
            "use_nf_sampler": use_nf_sampler,
            "train_batch_size": train_batch_size,
            "enable_3d_viz": enable_3d_viz,
            "verbose": verbose
        }

        response = requests.post(f"{self.base_url}/setup_renderer", json=data)
        result = response.json()

        if response.status_code == 200:
            self.session_id = result['session_id']
            print(f"Renderer setup started. Session ID: {self.session_id}")

            # Wait for setup to complete
            self.wait_for_renderer_ready()

        return result

    def wait_for_renderer_ready(self, timeout=60):
        """Wait for renderer to be ready"""
        if not self.session_id:
            raise ValueError("No session ID available")

        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status(self.session_id)
            if status['status'] == 'ready':
                print("Renderer is ready!")
                return True
            elif status['status'] == 'error':
                print(f"Renderer setup failed: {status.get('error', 'Unknown error')}")
                return False

            print(f"Renderer status: {status['status']}... waiting")
            time.sleep(2)

        print("Timeout waiting for renderer to be ready")
        return False

    def set_scene(self, scene_name):
        """Set current scene for the renderer"""
        if not self.session_id:
            raise ValueError("No session ID available. Call setup_renderer() first.")

        data = {
            "session_id": self.session_id,
            "scene_name": scene_name
        }

        response = requests.post(f"{self.base_url}/set_scene", json=data)
        result = response.json()

        if response.status_code == 200:
            print(f"Scene '{scene_name}' loaded successfully")
            print(f"Scene bounds: {result.get('scene_bounds', {})}")
        else:
            print(f"Failed to set scene: {result.get('error', 'Unknown error')}")

        return result

    def sample_render(self, num_samples=10, nn_sampling=False, return_albedo=True):
        """Sample render positions with albedo returns"""
        if not self.session_id:
            raise ValueError("No session ID available. Call setup_renderer() first.")

        data = {
            "session_id": self.session_id,
            "num_samples": num_samples,
            "nn_sampling": nn_sampling,
            "return_albedo": return_albedo
        }

        response = requests.post(f"{self.base_url}/sample_render", json=data)
        result = response.json()

        if response.status_code == 200:
            job_id = result['job_id']
            print(f"Sample render job started: {job_id}")

            # Wait for completion
            return self.wait_for_job_completion(job_id)
        else:
            print(f"Failed to start sample render: {result.get('error', 'Unknown error')}")
            return result

    def train_nf_sampler(self, num_steps=100, learning_rate=None, epochs_per_fit=None,
                        batch_size=None, hidden_units=None, hidden_layers=None):
        """Train NFSampler with custom parameters"""
        if not self.session_id:
            raise ValueError("No session ID available. Call setup_renderer() first.")

        data = {
            "session_id": self.session_id,
            "num_steps": num_steps
        }

        # Add optional training parameters
        if learning_rate is not None:
            data["learning_rate"] = learning_rate
        if epochs_per_fit is not None:
            data["epochs_per_fit"] = epochs_per_fit
        if batch_size is not None:
            data["batch_size"] = batch_size
        if hidden_units is not None:
            data["hidden_units"] = hidden_units
        if hidden_layers is not None:
            data["hidden_layers"] = hidden_layers

        response = requests.post(f"{self.base_url}/train", json=data)
        result = response.json()

        if response.status_code == 200:
            job_id = result['job_id']
            print(f"Training job started: {job_id}")
            print(f"Training parameters: {result.get('training_params', {})}")

            # Wait for completion
            return self.wait_for_job_completion(job_id)
        else:
            print(f"Failed to start training: {result.get('error', 'Unknown error')}")
            return result

    def wait_for_job_completion(self, job_id, timeout=300):
        """Wait for a job to complete"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status(job_id)

            if status['status'] == 'completed':
                print(f"Job {job_id} completed successfully!")
                return status
            elif status['status'] == 'error':
                print(f"Job {job_id} failed: {status.get('error', 'Unknown error')}")
                return status

            print(f"Job {job_id} status: {status['status']}... waiting")
            time.sleep(3)

        print(f"Timeout waiting for job {job_id} to complete")
        return None

    def get_status(self, identifier):
        """Get status of session or job"""
        response = requests.get(f"{self.base_url}/status/{identifier}")
        return response.json()

    def get_sessions(self):
        """Get list of active sessions"""
        response = requests.get(f"{self.base_url}/sessions")
        return response.json()

    def cleanup_session(self):
        """Cleanup current session"""
        if not self.session_id:
            return {"message": "No session to cleanup"}

        data = {"session_id": self.session_id}
        response = requests.post(f"{self.base_url}/cleanup", json=data)
        result = response.json()

        if response.status_code == 200:
            print(f"Session {self.session_id} cleaned up")
            self.session_id = None

        return result

    def save_albedo_image(self, albedo_data, albedo_shape, filename):
        """Save albedo render data as image"""
        try:
            # Convert list back to numpy array
            albedo_array = np.array(albedo_data, dtype=np.uint8)
            albedo_array = albedo_array.reshape(albedo_shape)

            # Convert to PIL Image and save
            image = Image.fromarray(albedo_array)
            image.save(filename)
            print(f"Albedo image saved as: {filename}")

        except Exception as e:
            print(f"Error saving albedo image: {e}")

def main():
    """Example usage of the new API features"""
    client = AdaptiveCameraSamplingClient(base_url=BASE_URL)

    try:
        # 1. Get API info
        print("=== API Information ===")
        api_info = client.get_api_info()
        print(f"API Version: {api_info['version']}")
        print(f"Available endpoints: {list(api_info['endpoints'].keys())}")

        # 2. Get available scenes
        print("\n=== Available Scenes ===")
        scenes_info = client.get_available_scenes()
        available_scenes = scenes_info['available_scenes']
        existing_scenes = scenes_info['existing_scenes']

        print(f"Total scenes: {scenes_info['total_scenes']}")
        print(f"Existing scenes: {len(existing_scenes)}")

        for scene_name, scene_info in available_scenes.items():
            status = "✓" if scene_info['exists'] else "✗"
            print(f"  {status} {scene_name}: {scene_info['description']}")

        if not existing_scenes:
            print("No scenes found! Please check scene file paths.")
            return

        # Choose first existing scene
        scene_name = list(existing_scenes.keys())[0]
        print(f"\nUsing scene: {scene_name}")

        # 3. Setup renderer
        print("\n=== Setting up Renderer ===")
        setup_result = client.setup_renderer(
            render_width=512,
            render_height=512,
            use_nf_sampler=True,
            train_batch_size=20,
            verbose=True
        )

        if not client.session_id:
            print("Failed to setup renderer")
            return

        # 4. Set scene
        print(f"\n=== Setting Scene: {scene_name} ===")
        scene_result = client.set_scene(scene_name)

        if 'error' in scene_result:
            print(f"Failed to set scene: {scene_result['error']}")
            return

        # 5. Sample render with random sampling
        print("\n=== Sample Render (Random Sampling) ===")
        random_results = client.sample_render(
            num_samples=10,
            nn_sampling=True,
            return_albedo=True
        )

        if random_results and 'results' in random_results:
            results = random_results['results']
            print(f"Random sampling completed: {len(results)} samples")

            print('results', results[0])

            # Save first albedo image
            if results and 'albedo_render' in results[0]:
                client.save_albedo_image(
                    results[0]['albedo_render'],
                    results[0]['albedo_shape'],
                    "random_sample_albedo.png"
                )

            # Show camera positions and surface areas
            for i, result in enumerate(results[:3]):  # Show first 3
                pos = result['camera_position_absolute']
                dir = result['camera_direction_absolute']
                area = result['surface_area']
                print(f"  Sample {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                      f"dir=({dir[0]:.2f}, {dir[1]:.2f}, {dir[2]:.2f}), area={area:.6f}")

        # 6. Train NFSampler
        print("\n=== Training NFSampler ===")
        training_result = client.train_nf_sampler(
            num_steps=50,
            learning_rate=0.001,
            epochs_per_fit=20,
            batch_size=32
        )

        if training_result and 'training_result' in training_result:
            train_info = training_result['training_result']
            print(f"Training completed:")
            print(f"  Steps: {train_info.get('num_steps_requested', 0)}")
            print(f"  Samples added: {train_info.get('num_samples_added', 0)}")
            print(f"  Total training samples: {train_info.get('total_training_samples', 0)}")
            print(f"  Duration: {train_info.get('training_duration_seconds', 0):.2f}s")
            print(f"  NFSampler trained: {train_info.get('is_nf_trained', False)}")

        # 7. Sample render with neural network sampling
        print("\n=== Sample Render (Neural Network Sampling) ===")
        nn_results = client.sample_render(
            num_samples=5,
            nn_sampling=True,
            return_albedo=True
        )

        if nn_results and 'results' in nn_results:
            results = nn_results['results']
            print(f"Neural network sampling completed: {len(results)} samples")

            # Save first albedo image
            if results and 'albedo_render' in results[0]:
                client.save_albedo_image(
                    results[0]['albedo_render'],
                    results[0]['albedo_shape'],
                    "nn_sample_albedo.png"
                )

            # Show camera positions and surface areas
            for i, result in enumerate(results[:3]):  # Show first 3
                pos = result['camera_position_absolute']
                dir = result['camera_direction_absolute']
                area = result['surface_area']
                method = result['sampling_method']
                print(f"  Sample {i} ({method}): pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                      f"dir=({dir[0]:.2f}, {dir[1]:.2f}, {dir[2]:.2f}), area={area:.6f}")

        # 8. Show session status
        print("\n=== Session Status ===")
        session_status = client.get_status(client.session_id)
        print(f"Session ID: {client.session_id}")
        print(f"Current scene: {session_status.get('current_scene', 'None')}")
        print(f"Total samples rendered: {session_status.get('total_samples_rendered', 0)}")
        print(f"Total training steps: {session_status.get('total_training_steps', 0)}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n=== Cleanup ===")
        client.cleanup_session()

def quick_test():
    """Quick test to verify API is working"""
    print("Quick API Test")
    print("=" * 20)

    # Just check if API is accessible
    result = requests.get(f"https://817a-37-16-65-130.ngrok-free.app/")
    if result:
        print("✓ API is accessible")
        print(result)
        # print(f"  Message: {result['message']}")
        return True
    else:
        print("✗ API is not accessible")
        return False

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()
        print("\nTo run a quick test only, use: python api_client_example.py --quick")
