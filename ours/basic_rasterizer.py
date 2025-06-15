#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import argparse
import shutil
from pathlib import Path
import json
from typing import Tuple, List, Dict, Optional
import datetime
import asyncio
import tempfile
from PIL import Image

# Add the scripts directory to path for common module
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(parent_dir, 'scripts', 'inv-rendering'))

# Add the llm-experiments directory for async_check_boolean
llm_experiments_dir = os.path.join(script_dir, 'llm-experiments')
sys.path.append(llm_experiments_dir)

import torch
import falcor
from falcor import float3, uint2
import common
from surface_area_worker import SurfaceAreaCalculator
from nf_sampler import NFSampler
from predetermined_sampler import PredeterminedSampler

# Import async Mistral functionality
try:
    from async_check_boolean import AsyncMistralVisionAPI
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("Warning: async_check_boolean module not available. Mistral integration disabled.")

# 3D visualization imports (optional)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: 3D visualization libraries not available. Install matplotlib and plotly for 3D likelihood visualization.")


class BasicRasterizer:
    def __init__(self, render_width: int = 512, render_height: int = 512, verbose: bool = True,
                 use_nf_sampler: bool = True, initial_samples: int = 100, train_batch_size: int = 50,
                 predetermined_batch_size: int = 50, max_training_history: int = 2000,
                 enable_3d_viz: bool = False, viz_every_n_steps: int = 50, viz_resolution: int = 20,
                 # Simple Mistral integration
                 enable_mistral: bool = False, target_object: str = "target",
                 use_direct_tensor: bool = True, debug_folder: Optional[str] = None,
                 warmup_frames: int = 3):
        self.render_width = render_width
        self.render_height = render_height
        self.verbose = verbose
        self.use_nf_sampler = use_nf_sampler
        self.initial_samples = initial_samples
        self.train_batch_size = train_batch_size
        self.predetermined_batch_size = predetermined_batch_size
        self.max_training_history = max_training_history

        # 3D visualization parameters
        self.enable_3d_viz = enable_3d_viz and VISUALIZATION_AVAILABLE
        self.viz_every_n_steps = viz_every_n_steps
        self.viz_resolution = viz_resolution
        self.viz_step_counter = 0

        if self.enable_3d_viz and not VISUALIZATION_AVAILABLE:
            print("Warning: 3D visualization requested but libraries not available. Disabling 3D visualization.")
            self.enable_3d_viz = False

        # Simple Mistral integration
        self.enable_mistral = enable_mistral and MISTRAL_AVAILABLE
        self.target_object = target_object
        self.use_direct_tensor = use_direct_tensor
        self.debug_folder = debug_folder
        self.mistral_api = None
        self.target_found = False
        self.inference_iterations = 0

        # Warm-up frames for inference
        self.warmup_frames = warmup_frames

        if self.enable_mistral:
            try:
                self.mistral_api = AsyncMistralVisionAPI(debug_folder=self.debug_folder)
                approach = "direct tensor processing (no disk I/O)" if self.use_direct_tensor else "disk-based (temp files)"
                debug_info = f" with debug folder: {self.debug_folder}" if self.debug_folder else ""
                print(f"Mistral API enabled for target: '{target_object}' using {approach}{debug_info}")

                # Create target folder and clear any existing images
                self.target_folder = Path("target_images")
                if self.target_folder.exists():
                    shutil.rmtree(self.target_folder)
                    if self.verbose:
                        print(f"🗑️  Cleared existing target images folder")

                self.target_folder.mkdir(exist_ok=True)
                if self.verbose:
                    print(f"📁 Target images folder ready: {self.target_folder}")

            except Exception as e:
                print(f"Warning: Failed to initialize Mistral API: {e}")
                self.enable_mistral = False

        # Initialize Falcor components
        self.testbed = None
        self.scene = None
        self.render_graph = None
        self.pathtracer_render_graph = None  # MinimalPathTracer render graph for inference
        self.device = None

        # Keep alive flag to prevent garbage collection
        self._keep_alive = False

        # Scene management
        self.current_scene_path = None
        self.available_scenes = {}
        self.scene_bounds = None

        # Performance tracking
        self.render_times = []
        self.surface_areas = []

        # Surface area calculator
        self.surface_area_calculator = SurfaceAreaCalculator(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            auto_pixel_area_threshold=True,
            verbose=verbose
        )

        # NFSampler for adaptive sampling
        self.nf_sampler = None
        self.training_data = []  # Store (5D_input, surface_area) pairs
        self.sample_count = 0
        self.is_nf_trained = False

        # PredeterminedSampler for batch efficiency
        self.predetermined_sampler = None
        self.samples_until_retrain = 0  # Counter to track when to retrain

        if self.use_nf_sampler:
            self.setup_nf_sampler()

    def is_valid(self) -> bool:
        """Check if the renderer is still valid and operational"""
        return (self.testbed is not None and
                hasattr(self, '_keep_alive') and
                self._keep_alive)

    def get_available_scenes(self) -> Dict[str, Dict]:
        """Get list of available scenes"""
        default_scenes = {
            "cornell_box": {
                "path": "D:/Models/CornellBox/cornell_box.pyscene",
                "description": "Classic Cornell Box scene",
                "exists": False
            },
            "bistro_exterior": {
                "path": "D:/Models/Bistro/BistroExterior.fbx",
                "description": "Bistro exterior scene",
                "exists": False
            }

        }




        # Check which scenes actually exist
        for scene_name, scene_info in default_scenes.items():
            scene_info["exists"] = os.path.exists(scene_info["path"])

        self.available_scenes = default_scenes
        return self.available_scenes

    def set_current_scene(self, scene_name: str) -> bool:
        """Set the current scene by name"""
        if not self.available_scenes:
            self.get_available_scenes()

        if scene_name not in self.available_scenes:
            if self.verbose:
                print(f"Scene '{scene_name}' not found in available scenes")
            return False

        scene_info = self.available_scenes[scene_name]
        if not scene_info["exists"]:
            if self.verbose:
                print(f"Scene file does not exist: {scene_info['path']}")
            return False

        try:
            self.load_scene(scene_info["path"])
            if self.verbose:
                print(f"Successfully set current scene to: {scene_name}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"Failed to load scene '{scene_name}': {e}")
            return False

    def sample_render_positions(self, num_samples: int, nn_sampling: bool = False,
                              return_albedo: bool = True) -> List[Dict]:
        """
        Sample X random positions and render, returning albedo + camera info

        Args:
            num_samples: Number of positions to sample
            nn_sampling: Whether to use neural network sampling (if trained)
            return_albedo: Whether to return albedo renders

        Returns:
            List of dictionaries containing render results
        """
        if self.scene is None:
            raise RuntimeError("No scene loaded. Call set_current_scene() first.")

        results = []

        for i in range(num_samples):
            if self.verbose:
                print(f"Sampling position {i+1}/{num_samples}")

            # Sample camera parameters
            if nn_sampling and self.is_nf_trained:
                camera_pos, camera_dir, input_5d, sample_pdf = self.sample_camera_params_nf()
                sampling_method = "neural_network"
            else:
                camera_pos, camera_dir, input_5d, sample_pdf = self.sample_camera_params_random()
                sampling_method = "random"

            # Set camera
            self.set_camera(camera_pos, camera_dir)

            # Prepare camera info
            camera_info = {
                "sample_id": i,
                "position": camera_pos.tolist(),
                "direction": camera_dir.tolist(),
                "target": (camera_pos + camera_dir).tolist(),
                "input_5d": input_5d.tolist(),
                "scene_bounds": {
                    "min": self.scene_bounds[0].tolist(),
                    "max": self.scene_bounds[1].tolist()
                }
            }

            # Render frame
            outputs, render_time = self.render_frame(camera_info)

            # Prepare result
            result = {
                'sample_id': i,
                'camera_position_absolute': camera_pos.tolist(),
                'camera_direction_absolute': camera_dir.tolist(),
                'camera_position_normalized': self.normalize_position(camera_pos).tolist(),
                'camera_direction_normalized': self.direction_to_spherical(camera_dir).tolist(),
                'surface_area': outputs.get('surface_area', 0.0),
                'render_time': render_time,
                'sampling_method': sampling_method,
                'surface_area_stats': outputs.get('surface_area_stats', {})
            }

            # Add albedo render if requested
            if return_albedo and 'diffuse' in outputs:
                # Convert tensor to numpy and then to list for JSON serialization
                albedo_array = outputs['diffuse'].cpu().numpy()
                # Convert to uint8 for smaller size
                albedo_uint8 = (np.clip(albedo_array, 0, 1) * 255).astype(np.uint8)
                result['albedo_render'] = albedo_uint8.tolist()
                result['albedo_shape'] = albedo_uint8.shape

            results.append(result)

        return results

    def train_nf_sampler_steps(self, num_steps: int, **training_params) -> Dict:
        """
        Train the NFSampler for a specified number of steps with custom parameters

        Args:
            num_steps: Number of training steps to perform
            **training_params: Additional training parameters to override defaults

        Returns:
            Dictionary with training results and statistics
        """
        if not self.use_nf_sampler:
            return {"error": "NFSampler is disabled"}

        if self.scene is None:
            return {"error": "No scene loaded"}

        # Update NFSampler parameters if provided
        if training_params:
            if 'learning_rate' in training_params:
                self.nf_sampler.learning_rate = training_params['learning_rate']
            if 'epochs_per_fit' in training_params:
                self.nf_sampler.epochs_per_fit = training_params['epochs_per_fit']
            if 'batch_size' in training_params:
                self.nf_sampler.batch_size = training_params['batch_size']
            if 'hidden_units' in training_params:
                self.nf_sampler.hidden_units = training_params['hidden_units']
            if 'hidden_layers' in training_params:
                self.nf_sampler.hidden_layers = training_params['hidden_layers']

        training_start_time = time.time()
        initial_training_data_size = len(self.training_data)

        # Collect training data by sampling
        for step in range(num_steps):
            if self.verbose and step % max(1, num_steps // 10) == 0:
                print(f"Training step {step+1}/{num_steps}")

            # Sample camera parameters
            camera_pos, camera_dir, input_5d, sample_pdf = self.sample_camera_params_random()

            # Set camera and render
            self.set_camera(camera_pos, camera_dir)

            camera_info = {
                "sample_id": step,
                "position": camera_pos.tolist(),
                "direction": camera_dir.tolist(),
                "target": (camera_pos + camera_dir).tolist(),
                "input_5d": input_5d.tolist(),
                "scene_bounds": {
                    "min": self.scene_bounds[0].tolist(),
                    "max": self.scene_bounds[1].tolist()
                }
            }

            outputs, render_time = self.render_frame(camera_info)
            surface_area = outputs.get('surface_area', 0.0)

            # Add to training data
            self.add_training_data(input_5d, surface_area, sample_pdf)

        training_end_time = time.time()
        training_duration = training_end_time - training_start_time

        # Final training statistics
        final_training_data_size = len(self.training_data)
        new_samples_added = final_training_data_size - initial_training_data_size

        result = {
            "training_completed": True,
            "num_steps_requested": num_steps,
            "num_samples_added": new_samples_added,
            "total_training_samples": final_training_data_size,
            "training_duration_seconds": training_duration,
            "is_nf_trained": self.is_nf_trained,
            "training_parameters_used": {
                "learning_rate": getattr(self.nf_sampler, 'learning_rate', None),
                "epochs_per_fit": getattr(self.nf_sampler, 'epochs_per_fit', None),
                "batch_size": getattr(self.nf_sampler, 'batch_size', None),
                "hidden_units": getattr(self.nf_sampler, 'hidden_units', None),
                "hidden_layers": getattr(self.nf_sampler, 'hidden_layers', None)
            }
        }

        if self.verbose:
            print(f"Training completed: {new_samples_added} new samples in {training_duration:.2f}s")
            print(f"NFSampler trained: {self.is_nf_trained}")

        return result

    def setup_nf_sampler(self):
        """Initialize the 5D NFSampler"""
        if self.verbose:
            print("Setting up 5D NFSampler (3D position + 2D spherical direction)")

        self.nf_sampler = NFSampler(
            name="camera_sampler",
            rng=None,
            num_flows=4,
            hidden_units=128,
            hidden_layers=2,
            learning_rate=1e-3,
            epochs_per_fit=25,
            batch_size=64,
            history_size=1000,
            latent_size=5,  # 5D: 3 for position + 2 for spherical direction
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Register 5 dimensions with the sampler
        for i in range(5):
            self.nf_sampler.register_dimension()

        self.nf_sampler.reinitialize()

    def normalize_position(self, position: np.ndarray) -> np.ndarray:
        """Convert absolute position to normalized [0,1] based on scene bounds"""
        min_bounds, max_bounds = self.scene_bounds
        bounds_size = max_bounds - min_bounds

        # Add small epsilon to avoid division by zero
        bounds_size = np.maximum(bounds_size, 1e-6)

        normalized = (position - min_bounds) / bounds_size
        return np.clip(normalized, 0.0, 1.0)

    def denormalize_position(self, normalized_pos: np.ndarray) -> np.ndarray:
        """Convert normalized [0,1] position back to absolute coordinates"""
        if self.scene_bounds is None:
            # Use default bounds if no scene is loaded
            self.scene_bounds = (np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0]))

        min_bounds, max_bounds = self.scene_bounds
        bounds_size = max_bounds - min_bounds

        # Add margin for better camera placement
        margin_factor = 0.2
        margin = bounds_size * margin_factor
        extended_min = min_bounds - margin
        extended_max = max_bounds + margin
        extended_size = extended_max - extended_min

        absolute_pos = extended_min + normalized_pos * extended_size
        return absolute_pos

    def direction_to_spherical(self, direction: np.ndarray) -> np.ndarray:
        """Convert 3D direction vector to 2D spherical coordinates [0,1]"""
        # Normalize direction
        direction = direction / np.linalg.norm(direction)

        # Convert to spherical coordinates
        theta = np.arctan2(direction[1], direction[0])  # azimuth [-pi, pi]
        phi = np.arccos(np.clip(direction[2], -1, 1))    # polar [0, pi]

        # Normalize to [0,1]
        theta_norm = (theta + np.pi) / (2 * np.pi)  # [0,1]
        phi_norm = phi / np.pi                       # [0,1]

        return np.array([theta_norm, phi_norm])

    def spherical_to_direction(self, spherical: np.ndarray) -> np.ndarray:
        """Convert 2D spherical coordinates [0,1] back to 3D direction vector"""
        theta_norm, phi_norm = spherical

        # Convert back to spherical coordinates
        theta = theta_norm * 2 * np.pi - np.pi  # [-pi, pi]
        phi = phi_norm * np.pi                  # [0, pi]

        # Convert to Cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        direction = np.array([x, y, z])
        return direction / np.linalg.norm(direction)  # Normalize

    def setup_falcor(self):
        """Initialize Falcor testbed and device"""
        if self.verbose:
            print(f"Setting up Falcor with resolution {self.render_width}x{self.render_height}")

        self.testbed = common.create_testbed([self.render_width, self.render_height])
        self.device = self.testbed.device
        self.testbed.render_graph = self.render_graph
        self._keep_alive = True  # Mark as active to prevent garbage collection

    def load_scene(self, scene_path: str):
        """Load a scene file"""
        if self.verbose:
            print(f"Loading scene: {scene_path}")

        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene file not found: {scene_path}")

        scene_path = Path(scene_path)
        self.scene = common.load_scene(
            self.testbed,
            scene_path,
            self.render_width / self.render_height
        )

        # Calculate scene bounding box
        self.scene_bounds = self._calculate_scene_bounds()

        if self.verbose:
            print(f"Scene bounds: min={self.scene_bounds[0]}, max={self.scene_bounds[1]}")

    def _calculate_scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the bounding box of the scene"""
        # Get scene bounds from Falcor
        if hasattr(self.scene, 'bounds'):
            bounds = self.scene.bounds
            min_bounds = np.array([bounds.minPoint.x, bounds.minPoint.y, bounds.minPoint.z])
            max_bounds = np.array([bounds.maxPoint.x, bounds.maxPoint.y, bounds.maxPoint.z])
        else:
            # Default bounds if not available
            min_bounds = np.array([-5.0, -5.0, -5.0])
            max_bounds = np.array([5.0, 5.0, 5.0])

        return min_bounds, max_bounds

    def setup_render_graph(self):
        """Setup the rasterization render graph"""
        if self.verbose:
            print("Setting up rasterization render graph")

        # Create render graph for basic rasterization
        self.render_graph = self.testbed.create_render_graph("BasicRasterizer")

        # Use GBufferRaster for rasterization (not ray tracing)
        # Disable culling to capture both front and back faces for signed surface area
        gbuffer_pass = self.render_graph.create_pass(
            "GBufferRaster",
            "GBufferRaster",
            {
                "samplePattern": "Center",
                "sampleCount": 1,
                "useAlphaTest": True,
                "forceCullMode": True,  # Force override of default cull mode
                "cull": "None"          # Disable culling to see both front and back faces
            }
        )

        # Mark outputs we want to capture
        self.render_graph.mark_output("GBufferRaster.diffuseOpacity")  # Diffuse
        self.render_graph.mark_output("GBufferRaster.depth")  # Depth
        self.render_graph.mark_output("GBufferRaster.linearZ")  # Linear Z and derivatives
        self.render_graph.mark_output("GBufferRaster.guideNormalW")  # World space normals for back-face detection

        # Assign render graph to testbed
        self.testbed.render_graph = self.render_graph

    def setup_pathtracer_render_graph(self):
        """Setup the PathTracer render graph with AccumulatePass + OptixDenoiser for inference with Mistral API"""
        if self.verbose:
            print("Setting up PathTracer + AccumulatePass + OptixDenoiser render graph for inference")

        # Create PathTracer render graph
        self.pathtracer_render_graph = self.testbed.create_render_graph("PathTracer")

        # VBufferRT pass
        vbuffer_rt = self.pathtracer_render_graph.create_pass(
            "VBufferRT",
            "VBufferRT",
            {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True}
        )

        # PathTracer pass (full path tracer with albedo and normal outputs)
        path_tracer = self.pathtracer_render_graph.create_pass(
            "PathTracer",
            "PathTracer",
            {'maxBounces': 1}
        )

        # AccumulatePass (for temporal accumulation)
        accumulate_pass = self.pathtracer_render_graph.create_pass(
            "AccumulatePass",
            "AccumulatePass",
            {'enabled': True, 'precisionMode': 'Single'}
        )

        # ToneMapper (after accumulation, before denoising)
        tone_mapper = self.pathtracer_render_graph.create_pass(
            "ToneMapper",
            "ToneMapper",
            {'autoExposure': False, 'exposureCompensation': 0.0}
        )

        # OptixDenoiser Pass (after ToneMapper)
        optix_denoiser = self.pathtracer_render_graph.create_pass(
            "OptixDenoiser",
            "OptixDenoiser"
        )

        # Connect the passes: VBufferRT -> PathTracer -> AccumulatePass -> ToneMapper -> OptixDenoiser
        self.pathtracer_render_graph.add_edge("VBufferRT.vbuffer", "PathTracer.vbuffer")
        self.pathtracer_render_graph.add_edge("VBufferRT.viewW", "PathTracer.viewW")

        # Path tracer to accumulate pass
        self.pathtracer_render_graph.add_edge("PathTracer.color", "AccumulatePass.input")

        # Accumulate pass to tone mapper
        self.pathtracer_render_graph.add_edge("AccumulatePass.output", "ToneMapper.src")

        # Connect OptixDenoiser inputs (tone mapped color + auxiliary buffers for better denoising)
        # PathTracer outputs albedo and guideNormal unlike MinimalPathTracer
        self.pathtracer_render_graph.add_edge("ToneMapper.dst", "OptixDenoiser.color")
        self.pathtracer_render_graph.add_edge("PathTracer.albedo", "OptixDenoiser.albedo")
        self.pathtracer_render_graph.add_edge("PathTracer.guideNormal", "OptixDenoiser.normal")
        self.pathtracer_render_graph.add_edge("VBufferRT.mvec", "OptixDenoiser.mvec")

        # Mark the OptixDenoiser output as final
        self.pathtracer_render_graph.mark_output("OptixDenoiser.output")

    def switch_to_pathtracer_rendering(self):
        """Switch to PathTracer rendering for inference"""
        if self.pathtracer_render_graph is None:
            self.setup_pathtracer_render_graph()

        self.testbed.render_graph = self.pathtracer_render_graph
        if self.verbose:
            print("Switched to PathTracer + AccumulatePass + OptixDenoiser rendering")

    def switch_to_basic_rendering(self):
        """Switch back to basic rasterization for training"""
        self.testbed.render_graph = self.render_graph
        if self.verbose:
            print("Switched to basic rasterization")

    def sample_camera_params_random(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Sample random camera parameters and return (position, direction, 5D_input, pdf)"""
        # Sample random position
        min_bounds, max_bounds = self.scene_bounds
        bounds_size = max_bounds - min_bounds
        margin_factor = 0.2
        margin = bounds_size * margin_factor
        extended_min = min_bounds - margin
        extended_max = max_bounds + margin
        extended_size = extended_max - extended_min

        position = np.random.uniform(extended_min, extended_max)

        # Calculate PDF for position (uniform over extended volume)
        position_pdf = 1.0 / np.prod(extended_size)

        # Sample random direction on sphere using uniform sampling
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        theta = 2 * np.pi * u1
        phi = np.arccos(2 * u2 - 1)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        direction = np.array([x, y, z])
        direction = direction / np.linalg.norm(direction)

        # Calculate PDF for direction (uniform on unit sphere)
        # Surface area of unit sphere is 4π, so PDF = 1/(4π)
        direction_pdf = 1.0 / (4.0 * np.pi)

        # Combined PDF (assuming independence)
        combined_pdf = position_pdf * direction_pdf

        # Convert to 5D input for NFSampler
        normalized_pos = self.normalize_position(position)
        spherical_dir = self.direction_to_spherical(direction)
        input_5d = np.concatenate([normalized_pos, spherical_dir])

        return position, direction, input_5d, combined_pdf

    def sample_camera_params_nf(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Sample camera parameters using NFSampler"""
        if not self.is_nf_trained:
            return self.sample_camera_params_random()

        # Sample from NFSampler
        sample = self.nf_sampler.sample_primary(num_samples=1)[0]
        input_5d = np.array(sample.value)
        pdf = sample.pdf

        # Convert 5D input back to position and direction
        normalized_pos = input_5d[:3]
        spherical_dir = input_5d[3:5]

        position = self.denormalize_position(normalized_pos)
        direction = self.spherical_to_direction(spherical_dir)

        return position, direction, input_5d, pdf

    def set_camera(self, position: np.ndarray, direction: np.ndarray):
        """Set camera position and direction"""
        # Calculate target point
        target = position + direction

        # Set camera parameters
        self.scene.camera.position = float3(position[0], position[1], position[2])
        self.scene.camera.target = float3(target[0], target[1], target[2])
        self.scene.camera.up = float3(0, 1, 0)  # Standard up vector

    def depth_to_world_coordinates(self, depth_tensor, camera_info):
        """Convert depth buffer to world coordinates using camera parameters"""
        try:
            h, w = depth_tensor.shape
            device = depth_tensor.device

            # Extract camera parameters
            camera_pos = torch.tensor(camera_info['position'], device=device, dtype=torch.float32)
            camera_dir = torch.tensor(camera_info['direction'], device=device, dtype=torch.float32)

            # Create coordinate grids
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing='ij'
            )

            # Convert to normalized device coordinates
            x_ndc = (x_coords - w/2) / (w/2)
            y_ndc = (y_coords - h/2) / (h/2)

            # Estimate focal length based on typical camera setup
            focal_length_px = min(w, h) * 0.8  # Rough estimate

            # Calculate camera rays
            ray_x = x_ndc / focal_length_px * w
            ray_y = -y_ndc / focal_length_px * h
            ray_z = -torch.ones_like(ray_x)

            # Normalize ray direction
            ray_dir = torch.stack([ray_x, ray_y, ray_z], dim=-1)
            ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)

            # Calculate world coordinates
            world_coords = camera_pos.unsqueeze(0).unsqueeze(0) + ray_dir * depth_tensor.unsqueeze(-1)

            # Create valid mask
            valid_mask = (depth_tensor > 0.001) & (depth_tensor < 0.999)

            return world_coords, valid_mask

        except Exception as e:
            print(f"Error in depth to world conversion: {e}")
            return None, None

    def render_frame(self, camera_info: dict) -> dict:
        """Render a single frame and return the output buffers with surface area estimation"""
        start_time = time.time()

        # Execute rendering

        self.scene.renderSettings.useEnvLight = True
        self.scene.renderSettings.useEmissiveLights = False

        # Check which render graph is active to determine warm-up frames
        is_pathtracer = (self.testbed.render_graph == self.pathtracer_render_graph)

        if is_pathtracer:
            # PathTracer + OptixDenoiser rendering needs warm-up frames for inference
            warmup_frames = getattr(self, 'warmup_frames', 3)
            if self.verbose and warmup_frames > 1:
                print(f"    Executing {warmup_frames} warm-up frames for PathTracer + AccumulatePass + OptixDenoiser...")
            for _ in range(warmup_frames):
                self.testbed.frame()

        else:
            # Basic rasterization - single frame is sufficient
            self.testbed.frame()

        # Get outputs and convert to PyTorch via numpy
        outputs = {}

        # Check which render graph is active
        is_pathtracer = (self.testbed.render_graph == self.pathtracer_render_graph)

        if is_pathtracer:
            # PathTracer + OptixDenoiser rendering - get OptixDenoiser output for Mistral
            optix_buffer = self.testbed.render_graph.get_output("OptixDenoiser.output")
            if optix_buffer is not None:
                optix_numpy = optix_buffer.to_numpy()

                # Ensure values are in [0,1] range for proper display
                optix_numpy = np.clip(optix_numpy, 0.0, 1.0)

                outputs['diffuse'] = torch.from_numpy(optix_numpy)

            # For PathTracer, we don't calculate surface area (used for training only)
            surface_area = 0.0
            surface_area_stats = {}
        else:
            # Basic rasterization - get all buffers for surface area calculation
            # Get diffuse buffer - keep on CPU to save GPU memory
            diffuse_buffer = self.render_graph.get_output("GBufferRaster.diffuseOpacity")
            if diffuse_buffer is not None:
                diffuse_numpy = diffuse_buffer.to_numpy()[:, :, :3]  # RGB only
                # Keep on CPU to save GPU memory
                outputs['diffuse'] = torch.from_numpy(diffuse_numpy)

            # Get depth buffer - temporarily move to GPU for processing, then back to CPU
            depth_buffer = self.render_graph.get_output("GBufferRaster.depth")
            depth_tensor_gpu = None
            if depth_buffer is not None:
                depth_numpy = depth_buffer.to_numpy()
                if torch.cuda.is_available():
                    depth_tensor_gpu = torch.from_numpy(depth_numpy).cuda()
                else:
                    depth_tensor_gpu = torch.from_numpy(depth_numpy)
                # Store CPU version to save memory
                outputs['depth'] = torch.from_numpy(depth_numpy)

            # Get linear Z buffer - keep on CPU
            linear_z_buffer = self.render_graph.get_output("GBufferRaster.linearZ")
            if linear_z_buffer is not None:
                linear_z_numpy = linear_z_buffer.to_numpy()
                outputs['linear_z'] = torch.from_numpy(linear_z_numpy)

            # Get normal buffer for back-face detection - temporarily move to GPU for processing
            normal_buffer = self.render_graph.get_output("GBufferRaster.guideNormalW")
            normal_tensor_gpu = None
            if normal_buffer is not None:
                normal_numpy = normal_buffer.to_numpy()[:, :, :3]  # Only XYZ components
                if torch.cuda.is_available():
                    normal_tensor_gpu = torch.from_numpy(normal_numpy).cuda()
                else:
                    normal_tensor_gpu = torch.from_numpy(normal_numpy)
                # Store CPU version to save memory
                outputs['normals'] = torch.from_numpy(normal_numpy)

            # Calculate surface area using depth buffer
            surface_area = 0.0
            surface_area_stats = {}

            if depth_tensor_gpu is not None:
                try:
                    # Use simple surface area calculation by default (counts visible surfaces)
                    surface_area = self.calculate_simple_surface_area(depth_tensor_gpu)
                    surface_area_stats = {
                        'method': 'simple_surface_count',
                        'visible_surface_ratio': surface_area,
                        'total_pixels': depth_tensor_gpu.numel(),
                        'visible_pixels': int(surface_area * depth_tensor_gpu.numel())
                    }

                    if self.verbose:
                        print(f"    Simple surface area (visible ratio): {surface_area:.6f}")
                        print(f"    Visible pixels: {surface_area_stats['visible_pixels']}/{surface_area_stats['total_pixels']}")

                    # Optional: Use complex surface area calculation if needed (commented out by default)
                    # Convert depth to world coordinates (using GPU tensor for processing)
                    # world_coords, valid_mask = self.depth_to_world_coordinates(depth_tensor_gpu, camera_info)
                    # if world_coords is not None and valid_mask is not None:
                    #     # Estimate surface area using the surface area calculator with signed area
                    #     surface_area, surface_area_stats, final_valid_mask = self.surface_area_calculator.estimate_surface_area_signed(
                    #         world_coords, valid_mask, normal_tensor_gpu, camera_info
                    #     )

                    # Clean up GPU tensors immediately
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                except Exception as e:
                    print(f"Error calculating surface area: {e}")
                    surface_area = 0.0

            # Clean up GPU tensors
            if depth_tensor_gpu is not None:
                del depth_tensor_gpu
            if normal_tensor_gpu is not None:
                del normal_tensor_gpu
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Track timing and surface area
        render_time = time.time() - start_time
        self.render_times.append(render_time)
        self.surface_areas.append(surface_area)

        # Add surface area info to outputs (scalars only, no tensors)
        outputs['surface_area'] = surface_area
        outputs['surface_area_stats'] = surface_area_stats

        return outputs, render_time

    def add_training_data(self, input_5d: np.ndarray, surface_area: float, sample_pdf: float):
        """Add training data to NFSampler"""
        if not self.use_nf_sampler:
            return

        self.training_data.append((input_5d.tolist(), surface_area, sample_pdf))

        # Limit training data history to prevent memory accumulation
        if len(self.training_data) > self.max_training_history:
            self.training_data = self.training_data[-self.max_training_history:]

        # Decrement samples until retrain
        if self.samples_until_retrain > 0:
            self.samples_until_retrain -= 1

        # Train when we have enough data and every 50 samples after that
        if len(self.training_data) >= self.train_batch_size and (
            not self.is_nf_trained or self.samples_until_retrain == 0
        ):
            self.train_nf_sampler()
            # Generate next batch of predetermined samples
            self.generate_predetermined_samples()

    def train_nf_sampler(self):
        """Train the NFSampler with accumulated data"""
        if not self.use_nf_sampler or len(self.training_data) < self.train_batch_size:
            return

        if self.verbose:
            print(f"\nTraining NFSampler with {len(self.training_data)} samples...")

        # Prepare training data
        x_data = [data[0] for data in self.training_data]  # 5D inputs
        y_data = [data[1] for data in self.training_data]  # Surface areas
        x_pdf_data = [data[2] for data in self.training_data]  # Actual PDF values used for sampling

        # Add data to NFSampler (note: NFSampler wants to minimize y, but we want to maximize surface area)
        # So we use negative surface area as the loss

        try:
            # Save weights before training (backup)
            backup_file = "nf_weights_backup_before_fit.pth"
            if self.is_nf_trained:
                try:
                    self.save_nf_weights(backup_file)
                    if self.verbose:
                        print(f"💾 Backup saved before training: {backup_file}")
                except Exception as backup_e:
                    if self.verbose:
                        print(f"⚠️  Warning: Could not save backup: {backup_e}")

            # Add data and fit
            self.nf_sampler.add_data(x_data, x_pdf_data, y_data)
            self.nf_sampler.fit()
            self.is_nf_trained = True

            # Save weights after successful training
            checkpoint_file = f"nf_weights_after_fit_{len(self.training_data)}.pth"
            try:
                self.save_nf_weights(checkpoint_file)
                if self.verbose:
                    print(f"✅ Weights saved after successful fit: {checkpoint_file}")
            except Exception as save_e:
                if self.verbose:
                    print(f"⚠️  Warning: Could not save weights after fit: {save_e}")

            if self.verbose:
                print(f"NFSampler training complete. Model is now active for sampling.")

            # Show 3D visualization if enabled and it's time
            if self.should_show_3d_viz():
                if self.verbose:
                    print("Generating 3D likelihood visualization...")
                self.visualize_3d_likelihood(len(self.training_data))

            # Clean up GPU memory after training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error training NFSampler: {e}")

            # Try to recover from backup if available
            backup_file = "nf_weights_backup_before_fit.pth"
            if os.path.exists(backup_file) and self.is_nf_trained:
                try:
                    print(f"🔄 Attempting to recover from backup: {backup_file}")
                    self.load_nf_weights(backup_file, load_training_data=False)
                    print(f"✅ Successfully recovered from backup")
                except Exception as recovery_e:
                    print(f"❌ Failed to recover from backup: {recovery_e}")
                    # Reset training status if recovery fails
                    self.is_nf_trained = False
            else:
                # No backup available or not previously trained
                self.is_nf_trained = False
                print(f"⚠️  No backup available. NFSampler training disabled.")

            # Clean up GPU memory even on error
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    print(f"🧹 GPU memory cleared after error")
                except Exception as cleanup_e:
                    print(f"⚠️  Could not clear GPU memory: {cleanup_e}")

            # Reset CUDA context if needed
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    print(f"🔄 CUDA context synchronized")
            except Exception as sync_e:
                print(f"⚠️  Could not synchronize CUDA: {sync_e}")

    def generate_predetermined_samples(self, batch_size: int = None):
        """Generate a batch of predetermined samples from NFSampler for efficiency"""
        if not self.is_nf_trained:
            return

        if batch_size is None:
            batch_size = self.predetermined_batch_size

        try:
            # Generate batch of samples from NFSampler
            nf_samples = self.nf_sampler.sample_primary(num_samples=batch_size)

            # Create PredeterminedSampler from these samples
            self.predetermined_sampler = PredeterminedSampler.from_samples(nf_samples)

            # Set samples until retrain counter
            self.samples_until_retrain = batch_size

            if self.verbose:
                print(f"Generated {batch_size} predetermined samples from NFSampler")

        except Exception as e:
            print(f"Error generating predetermined samples: {e}")
            self.predetermined_sampler = None

    def sample_camera_params_predetermined(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Sample camera parameters from predetermined samples (batch efficient)"""
        if self.predetermined_sampler is None or self.predetermined_sampler.remaining_samples() == 0:
            # Fall back to regular NF sampling
            return self.sample_camera_params_nf()

        try:
            # Get PDF for current sample
            pdf = self.predetermined_sampler.sample()

            # Get 5D input values
            pos_norm = np.array([
                self.predetermined_sampler.get(),  # x
                self.predetermined_sampler.get(),  # y
                self.predetermined_sampler.get()   # z
            ])

            spherical = np.array([
                self.predetermined_sampler.get(),  # theta
                self.predetermined_sampler.get()   # phi
            ])

            # Convert to world coordinates
            camera_pos = self.denormalize_position(pos_norm)
            camera_dir = self.spherical_to_direction(spherical)

            # Create 5D input vector for training
            input_5d = np.concatenate([pos_norm, spherical])

            return camera_pos, camera_dir, input_5d, pdf

        except Exception as e:
            print(f"Error in predetermined sampling: {e}")
            # Fall back to regular NF sampling
            return self.sample_camera_params_nf()

    def render_samples_training(self, num_samples: int):
        """Training mode: render samples to train NFSampler"""
        if self.verbose:
            print(f"TRAINING MODE: {num_samples} samples")
            if self.use_nf_sampler:
                print(f"NFSampler will be used after {self.train_batch_size} samples")
                print(f"Weights will be saved every 1000 samples")

        results = []

        for sample_id in range(num_samples):
            if self.verbose:
                # Determine sampling method before sampling
                if self.use_nf_sampler and self.is_nf_trained:
                    if self.predetermined_sampler and self.predetermined_sampler.remaining_samples() > 0:
                        method_preview = "NF-batch"
                    else:
                        method_preview = "NF-guided"
                else:
                    method_preview = "Random"
                print(f"Training sample {sample_id + 1}/{num_samples} ({method_preview})")

            # Sample camera parameters (adaptive or random)
            if self.use_nf_sampler and self.is_nf_trained:
                # Use predetermined samples for efficiency if available
                if self.predetermined_sampler and self.predetermined_sampler.remaining_samples() > 0:
                    camera_pos, camera_dir, input_5d, sample_pdf = self.sample_camera_params_predetermined()
                    sampling_method = "NF-batch"
                else:
                    camera_pos, camera_dir, input_5d, sample_pdf = self.sample_camera_params_nf()
                    sampling_method = "NF-guided"
            else:
                camera_pos, camera_dir, input_5d, sample_pdf = self.sample_camera_params_random()
                sampling_method = "Random"

            # Set camera
            self.set_camera(camera_pos, camera_dir)

            # Prepare camera info for surface area calculation
            camera_info = {
                "sample_id": sample_id,
                "position": camera_pos.tolist(),
                "direction": camera_dir.tolist(),
                "target": (camera_pos + camera_dir).tolist(),
                "input_5d": input_5d.tolist(),
                "scene_bounds": {
                    "min": self.scene_bounds[0].tolist(),
                    "max": self.scene_bounds[1].tolist()
                }
            }

            # Render frame with surface area calculation
            outputs, render_time = self.render_frame(camera_info)
            surface_area = outputs.get('surface_area', 0.0)

            # Add training data to NFSampler
            try:
                self.add_training_data(input_5d, surface_area, sample_pdf)
            except Exception as training_e:
                if self.verbose:
                    print(f"⚠️  Error adding training data for sample {sample_id}: {training_e}")
                    print(f"    Continuing with next sample...")
                # Continue with next sample instead of crashing

            # Store results - only keep essential data, not large tensors
            result = {
                'sample_id': sample_id,
                'camera_info': camera_info,
                'render_time': render_time,
                'surface_area': surface_area,
                'surface_area_stats': outputs.get('surface_area_stats', {}),
                'sampling_method': sampling_method
            }

            results.append(result)

            # Save weights every 1000 samples if NFSampler is trained
            if self.use_nf_sampler and self.is_nf_trained and (sample_id + 1) % 1000 == 0:
                checkpoint_file = f"nf_weights_checkpoint_{sample_id + 1}.pth"
                try:
                    self.save_nf_weights(checkpoint_file)
                    if self.verbose:
                        print(f"✅ Checkpoint saved: {checkpoint_file} (after {sample_id + 1} samples)")
                except Exception as e:
                    if self.verbose:
                        print(f"❌ Failed to save checkpoint: {e}")

            # Periodic memory cleanup every 10 samples
            if (sample_id + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.verbose:
                print(f"  Render time: {render_time:.3f}s")
                print(f"  Surface area: {surface_area:.6f}")
                if self.use_nf_sampler:
                    print(f"  Training data: {len(self.training_data)}/{self.train_batch_size}")

        # Save final weights
        if self.use_nf_sampler and self.is_nf_trained:
            final_weights_file = f"nf_weights_final_{num_samples}.pth"
            try:
                self.save_nf_weights(final_weights_file)
                if self.verbose:
                    print(f"✅ Final weights saved: {final_weights_file}")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed to save final weights: {e}")

        self._print_performance_stats()
        return results

    def tensor_to_temp_image(self, tensor: torch.Tensor) -> str:
        """Convert tensor to temporary image file (for disk-based approach)"""
        # Ensure tensor is on CPU
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # Convert to numpy and ensure correct range [0, 1] -> [0, 255]
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
            np_array = (torch.clamp(tensor, 0, 1) * 255).byte().numpy()
        else:
            np_array = tensor.numpy()

        # Ensure correct shape (H, W, C)
        if len(np_array.shape) == 3 and np_array.shape[2] >= 3:
            np_array = np_array[:, :, :3]  # Take RGB only

        # Convert to PIL Image and save to temp file
        pil_image = Image.fromarray(np_array, mode='RGB')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        pil_image.save(temp_file.name, quality=95)
        return temp_file.name



    async def check_target_in_image(self, tensor: torch.Tensor, sample_id: int) -> bool:
        """Check if target object is in the rendered image (supports both direct tensor and disk-based approaches)"""
        if not self.enable_mistral:
            return False

        try:
            if self.use_direct_tensor:
                # Optimized approach: send tensor directly to Mistral (no disk I/O)
                result = await self.mistral_api.check_object_presence_from_tensor_async(
                    tensor, self.target_object, sample_id=sample_id
                )
            else:
                # Traditional approach: save to temp file first
                temp_image_path = self.tensor_to_temp_image(tensor)
                try:
                    result = await self.mistral_api.check_object_presence_async(
                        temp_image_path, self.target_object, sample_id=sample_id
                    )
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

            self.inference_iterations += 1

            if self.verbose:
                method = "direct-tensor" if self.use_direct_tensor else "disk-based"
                print(f"  Sample {sample_id} ({method}): {result.response_text} -> {result.is_present}")

            return result.is_present

        except Exception as e:
            if self.verbose:
                method = "direct-tensor" if self.use_direct_tensor else "disk-based"
                print(f"  Error checking target in sample {sample_id} ({method}): {e}")



            return False

    def save_target_image(self, tensor: torch.Tensor, sample_id: int) -> str:
        """Save the potential target image with sample_id in filename"""
        # Ensure tensor is on CPU
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # Convert to numpy and ensure correct range [0, 1] -> [0, 255]
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
            np_array = (torch.clamp(tensor, 0, 1) * 255).byte().numpy()
        else:
            np_array = tensor.numpy()

        # Handle different tensor shapes
        if len(np_array.shape) == 3 and np_array.shape[2] >= 3:
            np_array = np_array[:, :, :3]  # Take RGB only

        # Convert to PIL Image and save with sample_id in filename
        pil_image = Image.fromarray(np_array, mode='RGB')
        filepath = self.target_folder / f"target_sample_{sample_id:04d}.jpeg"
        pil_image.save(filepath, quality=95)

        return str(filepath)

    async def render_single_sample_async(self, sample_id: int):
        """Render a single sample asynchronously"""
        if self.verbose:
            method = "NF-guided" if (self.use_nf_sampler and self.is_nf_trained) else "Random"
            print(f"Inference sample {sample_id + 1} ({method})")

        # Sample camera parameters using trained model
        if self.use_nf_sampler and self.is_nf_trained:
            camera_pos, camera_dir, input_5d, sample_pdf = self.sample_camera_params_nf()
            sampling_method = "NF-guided"
        else:
            camera_pos, camera_dir, input_5d, sample_pdf = self.sample_camera_params_random()
            sampling_method = "Random"

        # Set camera and render
        self.set_camera(camera_pos, camera_dir)

        camera_info = {
            "sample_id": sample_id,
            "position": camera_pos.tolist(),
            "direction": camera_dir.tolist(),
            "target": (camera_pos + camera_dir).tolist(),
            "input_5d": input_5d.tolist(),
            "scene_bounds": {
                "min": self.scene_bounds[0].tolist(),
                "max": self.scene_bounds[1].tolist()
            }
        }


        outputs, render_time = self.render_frame(camera_info)

        # Check for target if Mistral is enabled
        target_found = False
        if self.enable_mistral and 'diffuse' in outputs:
            target_found = await self.check_target_in_image(outputs['diffuse'], sample_id)

            if target_found:
                # Save the potential target image with sample_id in filename
                saved_path = self.save_target_image(outputs['diffuse'], sample_id)

                print(f"\n🎯 POTENTIAL TARGET FOUND! 🎯")
                print(f"Sample {sample_id}: Mistral detected '{self.target_object}' (inference #{self.inference_iterations})")
                print(f"Saved to: {saved_path}")
                print("Continuing search to find more candidates...\n")

                # Keep track of how many we found but don't stop
                if not hasattr(self, 'targets_found_count'):
                    self.targets_found_count = 0
                self.targets_found_count += 1
            else:
                print(f"Sample {sample_id}: Target not found (inference #{self.inference_iterations})")

        result = {
            'sample_id': sample_id,
            'camera_info': camera_info,
            'render_time': render_time,
            'surface_area': outputs.get('surface_area', 0.0),
            'sampling_method': sampling_method,
            'target_found': target_found
        }

        if self.verbose:
            print(f"  Render time: {render_time:.3f}s")
            if self.enable_mistral:
                print(f"  Inference iterations: {self.inference_iterations}")

        return result

    async def render_samples_inference(self, num_samples: int):
        """Inference mode: render samples with sliding window concurrency"""
        if self.verbose:
            print(f"INFERENCE MODE: {num_samples} samples")
            print(f"Looking for target: '{self.target_object}'")
            print(f"Using pretrained NFSampler: {self.is_nf_trained}")

        # Switch to PathTracer rendering for better quality inference images
        if self.enable_mistral:
            self.switch_to_pathtracer_rendering()

        # Set sliding window size
        window_size = 16 if self.enable_mistral else 1
        if self.verbose:
            print(f"Sliding window size: {window_size}")

        results = []
        tasks = []

        for sample_id in range(num_samples):
            # Create async task for this sample
            task = asyncio.create_task(self.render_single_sample_async(sample_id))
            tasks.append(task)

            # Process completed tasks when window is full or at the end
            if len(tasks) >= window_size or sample_id == num_samples - 1:
                # Wait for all tasks in current window to complete
                completed_results = await asyncio.gather(*tasks)
                results.extend(completed_results)

                # Clear tasks for next window
                tasks = []

        # Process any remaining tasks
        if tasks:
            completed_results = await asyncio.gather(*tasks)
            results.extend(completed_results)

        # Switch back to basic rendering after inference
        if self.enable_mistral:
            self.switch_to_basic_rendering()

        # Print final summary
        if self.enable_mistral:
            targets_found = getattr(self, 'targets_found_count', 0)
            print(f"\n📊 INFERENCE COMPLETE 📊")
            print(f"Total samples processed: {len(results)}")
            print(f"Total inference iterations: {self.inference_iterations}")
            print(f"Potential targets found: {targets_found}")
            if targets_found > 0:
                print(f"Target images saved in: {self.target_folder}")
                print("Review all saved images to verify which ones actually contain the target!")
            else:
                print("No potential targets detected by Mistral.")

        return results

    def render_samples(self, num_samples: int):
        """Backward compatibility wrapper"""
        return self.render_samples_training(num_samples)

    def _print_performance_stats(self):
        """Print performance statistics"""
        if len(self.render_times) > 0:
            avg_time = np.mean(self.render_times)
            min_time = np.min(self.render_times)
            max_time = np.max(self.render_times)
            total_time = np.sum(self.render_times)

            print(f"\nPerformance Statistics:")
            print(f"  Total samples: {len(self.render_times)}")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Average time per sample: {avg_time:.3f}s")
            print(f"  Min time: {min_time:.3f}s")
            print(f"  Max time: {max_time:.3f}s")
            print(f"  Samples per second: {len(self.render_times) / total_time:.2f}")

        if len(self.surface_areas) > 0:
            avg_area = np.mean(self.surface_areas)
            min_area = np.min(self.surface_areas)
            max_area = np.max(self.surface_areas)
            total_area = np.sum(self.surface_areas)

            # Calculate absolute area statistics for comparison
            abs_areas = [abs(area) for area in self.surface_areas]
            avg_abs_area = np.mean(abs_areas)
            total_abs_area = np.sum(abs_areas)

            print(f"\nSigned Surface Area Statistics:")
            print(f"  Average signed surface area: {avg_area:.6f}")
            print(f"  Min signed surface area: {min_area:.6f}")
            print(f"  Max signed surface area: {max_area:.6f}")
            print(f"  Total signed surface area: {total_area:.6f}")
            print(f"  Average absolute surface area: {avg_abs_area:.6f}")
            print(f"  Total absolute surface area: {total_abs_area:.6f}")

            if self.use_nf_sampler:
                random_samples = [r for r in self.surface_areas[:self.train_batch_size]]
                nf_samples = [r for r in self.surface_areas[self.train_batch_size:]]

                if len(random_samples) > 0 and len(nf_samples) > 0:
                    print(f"\nAdaptive Sampling Analysis:")
                    print(f"  Random sampling avg: {np.mean(random_samples):.6f}")
                    print(f"  NF-guided sampling avg: {np.mean(nf_samples):.6f}")
                    improvement = (np.mean(nf_samples) - np.mean(random_samples)) / abs(np.mean(random_samples)) * 100
                    print(f"  Improvement: {improvement:.1f}%")

                    # Also show absolute area improvement
                    random_abs = [abs(r) for r in random_samples]
                    nf_abs = [abs(r) for r in nf_samples]
                    abs_improvement = (np.mean(nf_abs) - np.mean(random_abs)) / np.mean(random_abs) * 100
                    print(f"  Absolute area improvement: {abs_improvement:.1f}%")

    def sample_3d_likelihood_grid(self, resolution: int = 20, directions_per_point: int = 100):
        """
        Sample likelihood values on a 3D grid for visualization

        Args:
            resolution: Grid resolution (NxNxN)
            directions_per_point: Number of random directions to sample per grid point

        Returns:
            tuple: (grid_positions, likelihood_values)
        """
        if not self.is_nf_trained:
            return None, None

        if self.verbose:
            print(f"Sampling 3D likelihood grid ({resolution}^3 = {resolution**3} points, {directions_per_point} directions each)")

        # Create 3D grid of positions in normalized space [0,1]^3
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        z = np.linspace(0, 1, resolution)

        grid_positions = []
        likelihood_values = []

        total_points = resolution ** 3
        processed = 0

        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    pos_norm = np.array([x[i], y[j], z[k]])

                    # Sample multiple random directions for this position
                    direction_likelihoods = []

                    for _ in range(directions_per_point):
                        # Sample random direction on unit sphere
                        u1 = np.random.uniform(0, 1)
                        u2 = np.random.uniform(0, 1)
                        theta = 2 * np.pi * u1
                        phi = np.arccos(2 * u2 - 1)

                        # Convert to spherical coordinates [0,1]^2
                        theta_norm = (theta + np.pi) / (2 * np.pi)
                        phi_norm = phi / np.pi
                        spherical_dir = np.array([theta_norm, phi_norm])

                        # Create 5D input
                        input_5d = np.concatenate([pos_norm, spherical_dir])

                        try:
                            # Get likelihood from NFSampler
                            likelihood = self.nf_sampler.get_likelihood(input_5d.tolist())
                            direction_likelihoods.append(likelihood)
                        except Exception as e:
                            # If likelihood calculation fails, use 0
                            direction_likelihoods.append(0.0)

                    # Average likelihood across all directions for this position
                    avg_likelihood = np.mean(direction_likelihoods) if direction_likelihoods else 0.0

                    # Convert normalized position to world coordinates for visualization
                    world_pos = self.denormalize_position(pos_norm)

                    grid_positions.append(world_pos)
                    likelihood_values.append(avg_likelihood)

                    processed += 1
                    if self.verbose and processed % (total_points // 10) == 0:
                        print(f"  Progress: {processed}/{total_points} ({100*processed/total_points:.1f}%)")

        return np.array(grid_positions), np.array(likelihood_values)

    def visualize_3d_likelihood(self, step: int):
        """
        Create and display 3D likelihood visualization

        Args:
            step: Current training step for title
        """
        if not self.enable_3d_viz or not self.is_nf_trained:
            return

        try:
            # Sample likelihood grid
            positions, likelihoods = self.sample_3d_likelihood_grid(
                resolution=self.viz_resolution,
                directions_per_point=50  # Reduced for faster computation
            )

            if positions is None or likelihoods is None:
                return

            # Normalize likelihoods for better visualization
            if np.max(likelihoods) > np.min(likelihoods):
                likelihoods_norm = (likelihoods - np.min(likelihoods)) / (np.max(likelihoods) - np.min(likelihoods))
            else:
                likelihoods_norm = likelihoods

            # Create 3D scatter plot with Plotly
            fig = go.Figure(data=go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=likelihoods_norm,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Normalized Likelihood"),
                    cmin=0,
                    cmax=1
                ),
                text=[f'Pos: ({x:.2f}, {y:.2f}, {z:.2f})<br>Likelihood: {l:.4f}'
                      for (x, y, z), l in zip(positions, likelihoods)],
                hovertemplate='%{text}<extra></extra>'
            ))

            # Add scene bounds visualization
            min_bounds, max_bounds = self.scene_bounds

            # Create wireframe box for scene bounds
            box_x = [min_bounds[0], max_bounds[0], max_bounds[0], min_bounds[0], min_bounds[0],
                     min_bounds[0], max_bounds[0], max_bounds[0], min_bounds[0], min_bounds[0],
                     max_bounds[0], max_bounds[0], min_bounds[0], min_bounds[0], max_bounds[0], max_bounds[0]]
            box_y = [min_bounds[1], min_bounds[1], max_bounds[1], max_bounds[1], min_bounds[1],
                     min_bounds[1], min_bounds[1], max_bounds[1], max_bounds[1], min_bounds[1],
                     min_bounds[1], max_bounds[1], max_bounds[1], min_bounds[1], min_bounds[1], max_bounds[1]]
            box_z = [min_bounds[2], min_bounds[2], min_bounds[2], min_bounds[2], min_bounds[2],
                     max_bounds[2], max_bounds[2], max_bounds[2], max_bounds[2], max_bounds[2],
                     min_bounds[2], min_bounds[2], max_bounds[2], max_bounds[2], max_bounds[2], max_bounds[2]]

            fig.add_trace(go.Scatter3d(
                x=box_x, y=box_y, z=box_z,
                mode='lines',
                line=dict(color='red', width=2),
                name='Scene Bounds',
                showlegend=True
            ))

            fig.update_layout(
                title=f'3D Likelihood Visualization - Training Step {step}<br>Higher likelihood = Better camera positions',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='cube'
                ),
                width=800,
                height=600
            )

            # Show the plot
            fig.show()

            if self.verbose:
                print(f"3D likelihood visualization displayed for step {step}")
                print(f"Likelihood range: {np.min(likelihoods):.6f} to {np.max(likelihoods):.6f}")

        except Exception as e:
            print(f"Error creating 3D visualization: {e}")

    def should_show_3d_viz(self) -> bool:
        """Check if we should show 3D visualization at this step"""
        if not self.enable_3d_viz or not self.is_nf_trained:
            return False

        self.viz_step_counter += 1
        return self.viz_step_counter % self.viz_every_n_steps == 0

    def save_nf_weights(self, filepath: str):
        """
        Save NFSampler weights and training data

        Args:
            filepath: Path to save the weights (should end with .pth or .pt)
        """
        if not self.use_nf_sampler:
            raise Exception("NFSampler is not enabled")

        if self.nf_sampler is None:
            raise Exception("NFSampler is not initialized")

        self.nf_sampler.save_weights(filepath)
        if self.verbose:
            print(f"NFSampler weights saved to: {filepath}")

    def load_nf_weights(self, filepath: str, load_training_data: bool = True):
        """
        Load NFSampler weights and optionally training data

        Args:
            filepath: Path to the saved weights file
            load_training_data: Whether to load the training data as well
        """
        if not self.use_nf_sampler:
            raise Exception("NFSampler is not enabled")

        if self.nf_sampler is None:
            self.setup_nf_sampler()

        self.nf_sampler.load_weights(filepath, load_training_data)

        # Update training status
        if self.nf_sampler.all_x is not None and self.nf_sampler.all_x.shape[0] > 0:
            self.is_nf_trained = True
            # Update training data list for compatibility
            self.training_data = []
            for i in range(self.nf_sampler.all_x.shape[0]):
                x_data = self.nf_sampler.all_x[i].cpu().numpy().tolist()
                y_data = self.nf_sampler.all_y_unnormalized[i].cpu().numpy().item()
                pdf_data = self.nf_sampler.all_x_pdf[i].cpu().numpy().item()
                self.training_data.append((x_data, y_data, pdf_data))

        if self.verbose:
            print(f"NFSampler weights loaded from: {filepath}")
            print(f"NFSampler trained: {self.is_nf_trained}")

    def get_nf_training_stats(self) -> Dict:
        """
        Get statistics about the NFSampler training data

        Returns:
            Dictionary with training statistics
        """
        if not self.use_nf_sampler or self.nf_sampler is None:
            return {"error": "NFSampler not available"}

        stats = {
            "is_initialized": self.nf_sampler.is_initialized,
            "is_trained": self.is_nf_trained,
            "total_dimensions": self.nf_sampler.total_dimensions,
            "latent_size": self.nf_sampler.latent_size,
            "training_samples": 0,
            "config": {
                "hidden_units": self.nf_sampler.hidden_units,
                "hidden_layers": self.nf_sampler.hidden_layers,
                "learning_rate": self.nf_sampler.learning_rate,
                "num_flows": self.nf_sampler.num_flows,
                "epochs_per_fit": self.nf_sampler.epochs_per_fit,
                "batch_size": self.nf_sampler.batch_size,
                "history_size": self.nf_sampler.history_size,
                "device": str(self.nf_sampler.device)
            }
        }

        if self.nf_sampler.all_x is not None:
            stats["training_samples"] = self.nf_sampler.all_x.shape[0]

            # Calculate statistics on training data
            y_data = self.nf_sampler.all_y_unnormalized.cpu().numpy()
            stats["performance_stats"] = {
                "min_performance": float(np.min(y_data)),
                "max_performance": float(np.max(y_data)),
                "mean_performance": float(np.mean(y_data)),
                "std_performance": float(np.std(y_data))
            }

        return stats

    def calculate_simple_surface_area(self, depth_tensor, depth_epsilon=0.001):
        """
        Simple surface area calculation that counts visible surfaces (pixels with valid depth).
        This maximizes sampling of camera locations that show the most surfaces rather than sky.

        Args:
            depth_tensor: Depth buffer tensor [H, W]
            depth_epsilon: Minimum depth threshold to consider a pixel as showing a surface

        Returns:
            float: Number of visible surface pixels (normalized by total pixels)
        """
        try:
            if depth_tensor is None:
                return 0.0

            # Count pixels with depth greater than epsilon (showing actual surfaces, not sky)
            valid_depth_mask = depth_tensor > depth_epsilon
            visible_surface_pixels = torch.sum(valid_depth_mask).item()

            # Normalize by total pixels to get a ratio [0, 1]
            total_pixels = depth_tensor.numel()
            surface_ratio = visible_surface_pixels / total_pixels if total_pixels > 0 else 0.0

            return surface_ratio

        except Exception as e:
            if self.verbose:
                print(f"Error in simple surface area calculation: {e}")
            return 0.0


async def main():
    parser = argparse.ArgumentParser(
        description="Basic Falcor Rasterizer with NFSampler and Mistral API",
        epilog="""
            Examples:
            # Use optimized direct tensor processing (default, fastest):
            python basic_rasterizer.py --enable_mistral --target_object "car" --mode inference

            # Use traditional disk-based approach (slower but more compatible):
            python basic_rasterizer.py --enable_mistral --target_object "car" --mode inference --use_disk_io

            # Full training + inference workflow with direct tensor processing:
            python basic_rasterizer.py --mode both --enable_mistral --target_object "thread with a needle"

            # Enable debug mode to save all requests for analysis:
            python basic_rasterizer.py --enable_mistral --target_object "car" --mode inference --debug_folder "mistral_debug"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene-file",  default="C:/Users/devmi/OneDrive/Documents/Bistro_v5_2/Bistro_v5_2/BistroOur.fbx", help="Path to scene file (.pyscene, .fbx, .gltf, etc.)")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to render")
    parser.add_argument("--width", type=int, default=512, help="Render width")
    parser.add_argument("--height", type=int, default=512, help="Render height")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no_nf", action="store_true", help="Disable NFSampler (use random sampling only)")
    parser.add_argument("--train_batch_size", type=int, default=100, help="Minimum samples before training NFSampler")
    parser.add_argument("--predetermined_batch_size", type=int, default=50, help="Number of samples to pre-generate from NFSampler for efficiency")
    parser.add_argument("--max_training_history", type=int, default=300, help="Maximum number of training samples to keep in memory")
    parser.add_argument("--enable_3d_viz", action="store_true", help="Enable 3D likelihood visualization")
    parser.add_argument("--viz_every_n_steps", type=int, default=200, help="Show 3D visualization every N training steps")
    parser.add_argument("--viz_resolution", type=int, default=25, help="Resolution of 3D visualization grid (NxNxN)")

    # Simple workflow arguments
    parser.add_argument("--mode", choices=["train", "inference", "both"], default="both",
                       help="Mode: 'train' (only training), 'inference' (only inference), 'both' (train then inference)")
    parser.add_argument("--weights_file", type=str, default="nf_weights.pth", help="Path to save/load NFSampler weights")
    parser.add_argument("--enable_mistral", action="store_true", help="Enable Mistral API for target detection")
    parser.add_argument("--target_object", type=str, default="target", help="Object to search for with Mistral API")
    parser.add_argument("--use_disk_io", action="store_true", help="Use disk-based approach (save temp files) instead of direct tensor processing")
    parser.add_argument("--debug_folder", type=str, default=None, help="Save all Mistral requests (images + prompts) to this folder for debugging")
    parser.add_argument("--warmup_frames", type=int, default=3, help="Number of warm-up frames for inference rendering (PathTracer + AccumulatePass + OptixDenoiser)")

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Create renderer
    renderer = BasicRasterizer(
        render_width=args.width,
        render_height=args.height,
        verbose=args.verbose,
        use_nf_sampler=not args.no_nf,
        train_batch_size=args.train_batch_size,
        predetermined_batch_size=args.predetermined_batch_size,
        max_training_history=args.max_training_history,
        enable_3d_viz=args.enable_3d_viz,
        viz_every_n_steps=args.viz_every_n_steps,
        viz_resolution=args.viz_resolution,
        enable_mistral=args.enable_mistral,
        target_object=args.target_object,
        use_direct_tensor=not args.use_disk_io,  # Invert the flag: --use_disk_io disables direct tensor
        debug_folder=args.debug_folder,
        warmup_frames=args.warmup_frames
    )

    try:
        # Setup Falcor
        renderer.setup_falcor()
        renderer.load_scene(args.scene_file)
        renderer.setup_render_graph()
        renderer.setup_pathtracer_render_graph()

        results = []

        # PHASE 1: TRAINING
        if args.mode in ["train", "both"]:
            print("\n" + "="*50)
            print("PHASE 1: TRAINING NFSampler")
            print("="*50)

            training_results = renderer.render_samples_training(args.num_samples)
            results.extend(training_results)

            if renderer.use_nf_sampler and renderer.is_nf_trained:
                renderer.save_nf_weights(args.weights_file)
                print(f"✅ NFSampler trained and saved to: {args.weights_file}")
            else:
                print("❌ NFSampler training failed")

                # PHASE 2: INFERENCE
        if args.mode in ["inference", "both"]:
            print("\n" + "="*50)
            print("PHASE 2: INFERENCE WITH MISTRAL API")
            print("="*50)

            # Load weights if doing inference only
            if args.mode == "inference":
                if os.path.exists(args.weights_file):
                    renderer.load_nf_weights(args.weights_file)
                    print(f"✅ Loaded weights from: {args.weights_file}")
                else:
                    print(f"❌ Weights file not found: {args.weights_file}")
                    print("⚠️  Continuing with random sampling (no NFSampler)")

            if not args.enable_mistral:
                print("Warning: Mistral API not enabled. Skipping inference.")
            else:
                # Run inference with target detection
                inference_samples = min(args.num_samples, 100) if args.mode == "both" else args.num_samples
                inference_results = await renderer.render_samples_inference(inference_samples)
                results.extend(inference_results)

        print(f"\nTotal samples processed: {len(results)}")
        return results

    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = asyncio.run(main())
    if results is not None:
        print(f"Results contain {len(results)} samples with adaptive camera sampling and surface area data")
