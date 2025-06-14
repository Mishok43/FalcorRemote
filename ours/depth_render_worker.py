import bpy
import bmesh
import mathutils
import os
import json
import datetime
import shutil
import random
import numpy as np
import argparse
import sys
from mathutils import Vector, Euler
from config import setup_default_render_settings

class DepthRenderWorker:
    def __init__(self, verbose=False):
        self.scene = bpy.context.scene
        self.cameras = []
        self.original_render_settings = {}
        self.scene_bbox = None
        self.verbose = verbose
        
    def clear_scene(self):
        """Clear all objects"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
    
    def prepare_output_directory(self, output_dir):
        """Create output directory and clear all existing content"""
        try:
            # Convert to absolute path - handle both relative and absolute paths
            if os.path.isabs(output_dir):
                abs_output_dir = output_dir
            else:
                # Get current working directory and join with relative path
                current_dir = os.getcwd()
                abs_output_dir = os.path.join(current_dir, output_dir)
            
            # Normalize the path
            abs_output_dir = os.path.normpath(abs_output_dir)
            
            # If directory exists, remove all contents
            if os.path.exists(abs_output_dir):
                if self.verbose:
                    print(f"Clearing existing content in: {abs_output_dir}")
                shutil.rmtree(abs_output_dir)
            
            # Create the directory
            os.makedirs(abs_output_dir, exist_ok=True)
            if self.verbose:
                print(f"Created output directory: {abs_output_dir}")
            
            return abs_output_dir
            
        except Exception as e:
            print(f"Error preparing output directory: {e}")
            return None
        
    def get_scene_bounding_box(self):
        """Calculate the bounding box of all mesh objects in the scene"""
        min_coords = Vector((float('inf'), float('inf'), float('inf')))
        max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))
        
        # Find all mesh objects (excluding cameras and lights)
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        
        if not mesh_objects:
            # Default bounding box if no mesh objects
            return Vector((-5, -5, -1)), Vector((5, 5, 10))
        
        for obj in mesh_objects:
            # Get world matrix
            matrix_world = obj.matrix_world
            
            # Get bounding box corners in local space
            bbox_corners = [matrix_world @ Vector(corner) for corner in obj.bound_box]
            
            # Update min/max coordinates
            for corner in bbox_corners:
                min_coords.x = min(min_coords.x, corner.x)
                min_coords.y = min(min_coords.y, corner.y)
                min_coords.z = min(min_coords.z, corner.z)
                max_coords.x = max(max_coords.x, corner.x)
                max_coords.y = max(max_coords.y, corner.y)
                max_coords.z = max(max_coords.z, corner.z)
        
        return min_coords, max_coords
    
    def normalize_position(self, position, bbox_min, bbox_max):
        """Normalize position to [0,1] range based on bounding box"""
        bbox_size = bbox_max - bbox_min
        normalized = Vector((
            (position.x - bbox_min.x) / bbox_size.x if bbox_size.x != 0 else 0.5,
            (position.y - bbox_min.y) / bbox_size.y if bbox_size.y != 0 else 0.5,
            (position.z - bbox_min.z) / bbox_size.z if bbox_size.z != 0 else 0.5
        ))
        return normalized
    
    def sample_uniform_sphere_direction(self):
        """Sample a uniform random direction on unit sphere"""
        # Use numpy for better random number generation
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        
        # Convert to spherical coordinates
        theta = 2 * np.pi * u1  # azimuth angle
        phi = np.arccos(2 * u2 - 1)  # polar angle
        
        # Convert to Cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        return Vector((x, y, z))
    
    def sample_random_camera_position(self, bbox_min, bbox_max, margin_factor=0.1):
        """Sample a random camera position within the scene bounding box with margin"""
        bbox_size = bbox_max - bbox_min
        
        # Add margin around bounding box for better camera placement
        margin = bbox_size * margin_factor
        extended_min = bbox_min - margin
        extended_max = bbox_max + margin
        
        # Sample uniformly within extended bounding box
        x = np.random.uniform(extended_min.x, extended_max.x)
        y = np.random.uniform(extended_min.y, extended_max.y)
        z = np.random.uniform(extended_min.z, extended_max.z)
        
        return Vector((x, y, z))
    
    def calculate_view_direction(self, camera):
        """Calculate the view direction vector from camera"""
        # Get camera's world matrix
        camera_matrix = camera.matrix_world
        
        # Camera's forward direction is negative Z in its local space
        local_forward = Vector((0, 0, -1))
        
        # Transform to world space (only rotation, not translation)
        world_forward = camera_matrix.to_3x3() @ local_forward
        world_forward.normalize()
        
        return world_forward
    
    def generate_timestamp(self):
        """Generate timestamp string for file naming"""
        import time
        # Add a small delay and use more precise timing
        time.sleep(0.001)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        # Add additional uniqueness with a counter if needed
        if hasattr(self, '_timestamp_counter'):
            self._timestamp_counter += 1
        else:
            self._timestamp_counter = 0
        
        return f"{timestamp}_{self._timestamp_counter:03d}"
    
    def export_camera_config(self, camera, camera_index, output_dir, timestamp):
        """Export camera configuration to JSON file"""
        try:
            # Get scene bounding box
            bbox_min, bbox_max = self.scene_bbox
            
            # Camera position (absolute)
            camera_pos = camera.location.copy()
            
            # Normalized camera position
            normalized_pos = self.normalize_position(camera_pos, bbox_min, bbox_max)
            
            # View direction
            view_direction = self.calculate_view_direction(camera)
            
            # Camera properties
            camera_data = camera.data
            
            # Compile camera configuration
            config = {
                "timestamp": timestamp,
                "camera_info": {
                    "name": camera.name,
                    "index": camera_index,
                    "sample_id": camera_index  # For random sampling
                },
                "position": {
                    "absolute": {
                        "x": camera_pos.x,
                        "y": camera_pos.y,
                        "z": camera_pos.z
                    },
                    "normalized": {
                        "x": normalized_pos.x,
                        "y": normalized_pos.y,
                        "z": normalized_pos.z
                    }
                },
                "view_direction": {
                    "x": view_direction.x,
                    "y": view_direction.y,
                    "z": view_direction.z
                },
                "rotation": {
                    "euler": {
                        "x": camera.rotation_euler.x,
                        "y": camera.rotation_euler.y,
                        "z": camera.rotation_euler.z
                    }
                },
                "camera_properties": {
                    "lens": camera_data.lens,
                    "sensor_width": camera_data.sensor_width,
                    "sensor_height": camera_data.sensor_height,
                    "clip_start": camera_data.clip_start,
                    "clip_end": camera_data.clip_end
                },
                "scene_bounding_box": {
                    "min": {
                        "x": bbox_min.x,
                        "y": bbox_min.y,
                        "z": bbox_min.z
                    },
                    "max": {
                        "x": bbox_max.x,
                        "y": bbox_max.y,
                        "z": bbox_max.z
                    },
                    "size": {
                        "x": bbox_max.x - bbox_min.x,
                        "y": bbox_max.y - bbox_min.y,
                        "z": bbox_max.z - bbox_min.z
                    }
                },
                "sampling_info": {
                    "method": "uniform_random",
                    "position_sampling": "uniform_within_extended_bbox",
                    "direction_sampling": "uniform_sphere"
                }
            }
            
            # Save to JSON file with timestamp naming
            config_filename = f"{timestamp}_config.json"
            config_path = os.path.join(output_dir, config_filename)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            if self.verbose:
                print(f"Camera config exported: {config_filename}")
            return config_path
            
        except Exception as e:
            print(f"Error exporting camera config: {e}")
            return None
    
    def setup_basic_scene(self):
        """Setup basic scene with lighting and simple objects"""
        # Add lighting
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
        sun = bpy.context.object
        sun.data.energy = 5.0
        
        # Create ground
        bpy.ops.mesh.primitive_plane_add(size=15, location=(0, 0, 0))
        ground = bpy.context.object
        ground.name = "Ground"
        
        # Create simple objects at different depths
        objects = [
            {"type": "cube", "location": (0, 0, 1), "size": 1},
            {"type": "sphere", "location": (0, 0, 3), "size": 0.8},
            {"type": "cylinder", "location": (0, 0, 5), "size": 0.6},
            {"type": "cone", "location": (2, 2, 2), "size": 0.5},
            {"type": "cube", "location": (-3, 1, 2), "size": 0.7},
            {"type": "sphere", "location": (3, -2, 1.5), "size": 0.6},
        ]
        
        for obj_data in objects:
            if obj_data["type"] == "cube":
                bpy.ops.mesh.primitive_cube_add(
                    size=obj_data["size"], 
                    location=obj_data["location"]
                )
            elif obj_data["type"] == "sphere":
                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=obj_data["size"]/2, 
                    location=obj_data["location"]
                )
            elif obj_data["type"] == "cylinder":
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=obj_data["size"]/2, 
                    depth=obj_data["size"], 
                    location=obj_data["location"]
                )
            elif obj_data["type"] == "cone":
                bpy.ops.mesh.primitive_cone_add(
                    radius1=obj_data["size"]/2, 
                    depth=obj_data["size"], 
                    location=obj_data["location"]
                )
            
            obj = bpy.context.object
            obj.name = f"{obj_data['type'].title()}_{len([o for o in bpy.context.scene.objects if obj_data['type'] in o.name.lower()])}"
            
            # Add simple material
            mat = bpy.data.materials.new(name=f"{obj.name}_Material")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            # Random color
            color = (random.random(), random.random(), random.random(), 1)
            nodes["Principled BSDF"].inputs[0].default_value = color
            obj.data.materials.append(mat)
    
    def create_random_cameras(self, num_cameras=10):
        """Create cameras with random positions and orientations"""
        print(f"Creating {num_cameras} random cameras...")
        
        # Calculate scene bounding box once
        self.scene_bbox = self.get_scene_bounding_box()
        bbox_min, bbox_max = self.scene_bbox
        bbox_center = (bbox_min + bbox_max) * 0.5
        bbox_size = bbox_max - bbox_min
        
        print(f"Scene bounding box: min={bbox_min}, max={bbox_max}")
        print(f"Scene center: {bbox_center}, size: {bbox_size}")
        
        for i in range(num_cameras):
            # Sample random camera position
            camera_pos = self.sample_random_camera_position(bbox_min, bbox_max)
            
            # Sample random look-at direction
            look_at_dir = self.sample_uniform_sphere_direction()
            
            # Calculate look-at target point (some distance in the direction)
            look_distance = bbox_size.length * 0.5  # Look at reasonable distance
            look_at_target = camera_pos + look_at_dir * look_distance
            
            # Create camera
            bpy.ops.object.camera_add(location=camera_pos)
            camera = bpy.context.object
            camera.name = f"RandomCamera_{i:03d}"
            
            # Point camera towards look-at target
            direction = look_at_target - camera_pos
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()
            
            # Set camera properties
            camera.data.lens = 50
            camera.data.clip_start = 0.1
            camera.data.clip_end = max(100, bbox_size.length * 2)  # Adjust clip end based on scene size
            
            self.cameras.append(camera)
        
        if self.verbose:
            print(f"Created {len(self.cameras)} random cameras")
    
    def save_render_settings(self):
        """Save current render settings for restoration later"""
        self.original_render_settings = {
            'resolution_x': self.scene.render.resolution_x,
            'resolution_y': self.scene.render.resolution_y,
            'resolution_percentage': self.scene.render.resolution_percentage,
            'engine': self.scene.render.engine,
            'view_transform': self.scene.view_settings.view_transform,
            'film_transparent': self.scene.render.film_transparent,
            'view_settings_look': self.scene.view_settings.look,
            'use_nodes': self.scene.use_nodes,
            'use_compositing': self.scene.render.use_compositing,
        }
        
        # Store sequencer colorspace if it exists
        if hasattr(self.scene, 'sequencer_colorspace_settings'):
            self.original_render_settings['colorspace'] = self.scene.sequencer_colorspace_settings.name
    
    def restore_render_settings(self):
        """Restore original render settings"""
        if not self.original_render_settings:
            return
            
        self.scene.render.resolution_x = self.original_render_settings['resolution_x']
        self.scene.render.resolution_y = self.original_render_settings['resolution_y']
        self.scene.render.resolution_percentage = self.original_render_settings['resolution_percentage']
        self.scene.render.engine = self.original_render_settings['engine']
        self.scene.view_settings.view_transform = self.original_render_settings['view_transform']
        self.scene.render.film_transparent = self.original_render_settings['film_transparent']
        self.scene.view_settings.look = self.original_render_settings['view_settings_look']
        self.scene.use_nodes = self.original_render_settings['use_nodes']
        self.scene.render.use_compositing = self.original_render_settings['use_compositing']
        
        if 'colorspace' in self.original_render_settings and hasattr(self.scene, 'sequencer_colorspace_settings'):
            self.scene.sequencer_colorspace_settings.name = self.original_render_settings['colorspace']
    
    def setup_depth_compositor_once(self, output_dir):
        """Setup compositor nodes for depth map export (one-time setup)"""
        # Enable compositor nodes
        self.scene.use_nodes = True
        self.scene.render.use_compositing = True
        
        # Enable Z pass
        bpy.context.view_layer.use_pass_z = True
        
        # Set up compositor settings for depth export
        self.scene.view_settings.view_transform = 'Raw'
        self.scene.view_settings.look = 'None'
        self.scene.render.film_transparent = True
        
        # Set colorspace if available
        if hasattr(self.scene, 'sequencer_colorspace_settings'):
            self.scene.sequencer_colorspace_settings.name = 'Non-Color'
        
        # Get or create compositor node tree
        if not self.scene.node_tree:
            return None
            
        tree = self.scene.node_tree
        
        # Clear existing nodes
        for node in tree.nodes:
            tree.nodes.remove(node)
        
        # Create Render Layers node
        render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')
        render_layers_node.location = (-400, 0)
        
        # Create Invert node
        invert_node = tree.nodes.new(type='CompositorNodeInvert')
        invert_node.location = (-100, -200)
        
        # Create Normalize node
        normalize_node = tree.nodes.new(type='CompositorNodeNormalize')
        normalize_node.location = (100, -200)
        
        # Create Set Alpha node
        set_alpha_node = tree.nodes.new(type='CompositorNodeSetAlpha')
        set_alpha_node.location = (300, 0)
        
        # Create File Output node for depth map
        file_output_node = tree.nodes.new(type='CompositorNodeOutputFile')
        file_output_node.label = 'Depth Output'
        file_output_node.base_path = output_dir
        file_output_node.location = (600, 0)
        
        # Set output properties
        file_output_node.format.file_format = 'PNG'
        file_output_node.format.color_mode = 'RGBA'
        file_output_node.format.color_depth = '16'
        
        # Clear default file slots and add our custom ones
        file_output_node.file_slots.clear()
        file_output_node.file_slots.new("Depth")
        
        # Connect nodes for depth pipeline
        tree.links.new(render_layers_node.outputs["Depth"], invert_node.inputs["Color"])
        tree.links.new(render_layers_node.outputs["Alpha"], set_alpha_node.inputs["Alpha"])
        tree.links.new(invert_node.outputs["Color"], normalize_node.inputs["Value"])
        tree.links.new(normalize_node.outputs["Value"], set_alpha_node.inputs["Image"])
        tree.links.new(set_alpha_node.outputs["Image"], file_output_node.inputs["Depth"])
        
        # Store reference to file output node for later updates
        self.depth_file_output_node = file_output_node
        
        return True
    
    def update_depth_output_filename(self, timestamp):
        """Update the depth output filename without recreating compositor"""
        if hasattr(self, 'depth_file_output_node') and self.depth_file_output_node:
            old_path = self.depth_file_output_node.file_slots[0].path
            new_path = f"{timestamp}_depth"
            self.depth_file_output_node.file_slots[0].path = new_path
            
            # Verify the base path is still correct
            if not self.depth_file_output_node.base_path:
                print(f"  Warning: base_path was reset, restoring it")
                self.depth_file_output_node.base_path = self.current_output_dir
            
            # Force scene update to ensure Blender recognizes the change
            bpy.context.view_layer.update()
            
            print(f"  Updated depth output: '{old_path}' -> '{new_path}'")
            print(f"  Base path: '{self.depth_file_output_node.base_path}'")
            return True
        return False
    
    def setup_depth_compositor(self, output_dir, timestamp):
        """Setup compositor nodes for depth map export"""
        # Enable compositor nodes
        self.scene.use_nodes = True
        self.scene.render.use_compositing = True
        
        # Enable Z pass
        bpy.context.view_layer.use_pass_z = True
        
        # Set up compositor settings for depth export
        self.scene.view_settings.view_transform = 'Raw'
        self.scene.view_settings.look = 'None'
        self.scene.render.film_transparent = True
        
        # Set colorspace if available
        if hasattr(self.scene, 'sequencer_colorspace_settings'):
            self.scene.sequencer_colorspace_settings.name = 'Non-Color'
        
        # Get or create compositor node tree
        if not self.scene.node_tree:
            return False
            
        tree = self.scene.node_tree
        
        # Clear existing nodes
        for node in tree.nodes:
            tree.nodes.remove(node)
        
        # Create Render Layers node
        render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')
        render_layers_node.location = (-400, 0)
        
        # Create Invert node
        invert_node = tree.nodes.new(type='CompositorNodeInvert')
        invert_node.location = (-100, -200)
        
        # Create Normalize node
        normalize_node = tree.nodes.new(type='CompositorNodeNormalize')
        normalize_node.location = (100, -200)
        
        # Create Set Alpha node
        set_alpha_node = tree.nodes.new(type='CompositorNodeSetAlpha')
        set_alpha_node.location = (300, 0)
        
        # Create File Output node for depth map
        file_output_node = tree.nodes.new(type='CompositorNodeOutputFile')
        file_output_node.label = 'Depth Output'
        file_output_node.base_path = output_dir
        file_output_node.location = (600, 0)
        
        # Set output properties
        file_output_node.format.file_format = 'PNG'
        file_output_node.format.color_mode = 'RGBA'
        file_output_node.format.color_depth = '16'
        
        # Clear default file slots and add our custom ones
        file_output_node.file_slots.clear()
        file_output_node.file_slots.new("Depth")
        file_output_node.file_slots[0].path = f"{timestamp}_depth"
        
        # Connect nodes for depth pipeline
        tree.links.new(render_layers_node.outputs["Depth"], invert_node.inputs["Color"])
        tree.links.new(render_layers_node.outputs["Alpha"], set_alpha_node.inputs["Alpha"])
        tree.links.new(invert_node.outputs["Color"], normalize_node.inputs["Value"])
        tree.links.new(normalize_node.outputs["Value"], set_alpha_node.inputs["Image"])
        tree.links.new(set_alpha_node.outputs["Image"], file_output_node.inputs["Depth"])
        
        return True
    
    def setup_render_settings(self):
        """Setup basic render settings"""
        setup_default_render_settings(self.scene)
        self.scene.render.resolution_x = 800
        self.scene.render.resolution_y = 600
        
        if self.verbose:
            print("Render settings configured")
    
    def render_cameras(self, output_dir="tmp_renders", export_depth=True, export_config=True, full_render=True):
        """Render from all cameras with optional depth map and config export"""
        try:
            # Prepare output directory (create and clear)
            abs_output_dir = self.prepare_output_directory(output_dir)
            if not abs_output_dir:
                return []
            
            # Save original settings
            self.save_render_settings()
            
            # Setup depth compositor once before the loop (if depth export is enabled)
            if export_depth:
                print("Setting up depth compositor (one-time setup)...")
                self.current_output_dir = abs_output_dir  # Store for later reference
                if not self.setup_depth_compositor_once(abs_output_dir):
                    print("Warning: Failed to setup depth compositor")
                    export_depth = False
            
            # Store original camera
            original_camera = self.scene.camera
            
            render_info = []
            
            for i, camera in enumerate(self.cameras):
                try:
                    # Generate unique timestamp for this camera
                    timestamp = self.generate_timestamp()
                    
                    # Set this camera as active
                    self.scene.camera = camera
                    
                    # Export camera configuration
                    config_path = None
                    if export_config:
                        if self.verbose:
                            print(f"Exporting camera config for sample {i+1}/{len(self.cameras)}")
                        config_path = self.export_camera_config(camera, i, abs_output_dir, timestamp)
                    
                    depth_map_path = None
                    if export_depth:
                        if self.verbose:
                            print(f"Rendering depth map for sample {i+1}/{len(self.cameras)}")
                        
                        # Update output filename only (no node recreation)
                        if self.update_depth_output_filename(timestamp):
                            # Render depth map
                            bpy.ops.render.render(write_still=True)
                            depth_map_path = os.path.join(abs_output_dir, f"{timestamp}_depth0001.png")
                            
                            # Only restore settings if we need to do regular render afterwards
                            if full_render:
                                self.restore_render_settings()
                                self.setup_render_settings()
                        else:
                            if self.verbose:
                                print(f"Warning: Could not update depth output filename for sample {i+1}")
                            # Fallback to old method
                            if self.setup_depth_compositor(abs_output_dir, timestamp):
                                bpy.ops.render.render(write_still=True)
                                depth_map_path = os.path.join(abs_output_dir, f"{timestamp}_depth0001.png")
                                if full_render:
                                    self.restore_render_settings()
                                    self.setup_render_settings()
                    
                    # Regular render (only if full_render is enabled)
                    render_path = None
                    if full_render:
                        if self.verbose:
                            print(f"Rendering regular image for sample {i+1}/{len(self.cameras)}")
                        
                        # Set output path for regular render with timestamp
                        render_output_path = os.path.join(abs_output_dir, f"{timestamp}_render")
                        self.scene.render.filepath = render_output_path
                        
                        # Render regular image
                        bpy.ops.render.render(write_still=True)
                        render_path = render_output_path + ".png"
                    
                    # Single status message per sample
                    status_parts = []
                    if export_depth and depth_map_path:
                        status_parts.append("depth")
                    if full_render and render_path:
                        status_parts.append("render")
                    if export_config and config_path:
                        status_parts.append("config")
                    
                    status = " + ".join(status_parts) if status_parts else "processed"
                    print(f"Sample {i+1:02d}/{len(self.cameras)}: {timestamp} ({status})")
                    
                    # Store info
                    render_info.append({
                        "timestamp": timestamp,
                        "camera_name": camera.name,
                        "sample_id": i,
                        "location": list(camera.location),
                        "render_path": render_path,
                        "depth_map_path": depth_map_path,
                        "config_path": config_path
                    })
                    
                except Exception as e:
                    print(f"Error rendering camera {i+1}: {e}")
                    continue
            
            # Restore original camera and settings
            self.scene.camera = original_camera
            self.restore_render_settings()
            
            # Save render information
            if render_info:
                info_path = os.path.join(abs_output_dir, "render_info.json")
                with open(info_path, 'w') as f:
                    json.dump(render_info, f, indent=2)
                
                if self.verbose:
                    print(f"Rendering complete! Check {abs_output_dir} for results.")
                    print(f"Render info saved to {info_path}")
                    if export_depth:
                        print("Depth maps exported with timestamp naming")
                    if export_config:
                        print("Camera configuration files exported with timestamp naming")
            else:
                print("No renders were completed successfully")
            
            return render_info
            
        except Exception as e:
            print(f"Error in render_cameras: {e}")
            # Ensure settings are restored even on error
            self.restore_render_settings()
            return []
    
    def run_pipeline(self, output_dir="tmp_renders", num_samples=10, export_depth=True, export_config=True, full_render=True, random_seed=None):
        """Run the complete random sampling pipeline"""
        try:
            if self.verbose:
                print("Starting Random Depth Rendering Pipeline...")
                print(f"Output directory: {output_dir}")
                print(f"Number of samples: {num_samples}")
                print(f"Full render mode: {'Enabled' if full_render else 'Disabled (depth-only)'}")
            
            # Set random seed for reproducibility
            if random_seed is not None:
                random.seed(random_seed)
                np.random.seed(random_seed)
                if self.verbose:
                    print(f"Random seed: {random_seed}")
            
            # Setup scene
            if self.verbose:
                print("Setting up scene...")
            self.clear_scene()
            self.setup_basic_scene()
            
            # Create random cameras
            if self.verbose:
                print("Creating random cameras...")
            self.create_random_cameras(num_cameras=num_samples)
            
            # Setup render settings
            if self.verbose:
                print("Configuring render settings...")
            self.setup_render_settings()
            
            # Render
            if self.verbose:
                print("Starting rendering...")
                if export_depth:
                    print("Depth map export enabled - will export depth buffers with timestamp naming")
                if export_config:
                    print("Camera config export enabled - will export JSON config files with timestamp naming")
                if not full_render:
                    print("Full render disabled - only depth maps will be generated")
            render_info = self.render_cameras(output_dir=output_dir, export_depth=export_depth, export_config=export_config, full_render=full_render)
            
            if self.verbose:
                print("Pipeline completed!")
            return render_info
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            return []

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Random Depth Rendering Worker for Blender')
    
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of random camera samples to generate (default: 20)')
    
    parser.add_argument('--output_dir', type=str, default='tmp_renders',
                        help='Output directory for renders (default: tmp_renders)')
    
    parser.add_argument('--full_render', action='store_true',
                        help='Enable full rendering (depth + regular images). If not set, only depth maps are generated.')
    
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducible results (default: 42)')
    
    parser.add_argument('--no_config', action='store_true',
                        help='Disable camera config export')
    
    parser.add_argument('--no_depth', action='store_true',
                        help='Disable depth map export')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with detailed progress messages')
    
    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        print("Random Depth Rendering Worker")
        print("=" * 50)
        print(f"Samples: {args.num_samples}")
        print(f"Output directory: {args.output_dir}")
        print(f"Full render: {'Yes' if args.full_render else 'No (depth-only)'}")
        print(f"Export depth: {'No' if args.no_depth else 'Yes'}")
        print(f"Export config: {'No' if args.no_config else 'Yes'}")
        print(f"Random seed: {args.random_seed}")
        print(f"Verbose: {'Yes' if args.verbose else 'No'}")
        print("=" * 50)
        
        worker = DepthRenderWorker(verbose=args.verbose)
        
        render_info = worker.run_pipeline(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            export_depth=not args.no_depth,
            export_config=not args.no_config,
            full_render=args.full_render,
            random_seed=args.random_seed
        )
        
        if render_info:
            print(f"Successfully rendered {len(render_info)} random views")
            if args.verbose:
                for info in render_info:
                    print(f"- Sample {info['sample_id']:02d}: {info['timestamp']} ({info['camera_name']})")
                    print(f"  Position: ({info['location'][0]:.2f}, {info['location'][1]:.2f}, {info['location'][2]:.2f})")
                    if args.full_render and info.get('render_path'):
                        print(f"  Regular render: {os.path.basename(info['render_path'])}")
                    if not args.no_depth and info.get('depth_map_path'):
                        print(f"  Depth map: {os.path.basename(info['depth_map_path'])}")
        else:
            print("No renders completed")
            
    except Exception as e:
        print(f"Fatal error: {e}") 