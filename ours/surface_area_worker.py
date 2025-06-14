import os
import json
import time
import torch
import numpy as np
from PIL import Image
import threading
import glob

class SurfaceAreaCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 max_pixel_area=None, auto_pixel_area_threshold=True, verbose=False):
        """
        Initialize surface area calculator with pixel area filtering

        Args:
            device: Computing device ('cuda' or 'cpu')
            max_pixel_area: Maximum surface area per pixel to include in calculation
            auto_pixel_area_threshold: If True, automatically calculate threshold based on scene bounding box
            verbose: Enable verbose output
        """
        self.device = device
        self.max_pixel_area = max_pixel_area
        self.auto_pixel_area_threshold = auto_pixel_area_threshold
        self.verbose = verbose
        if verbose:
            print(f"Using device: {self.device}")
            if max_pixel_area is not None and not auto_pixel_area_threshold:
                print(f"Max pixel area threshold: {self.max_pixel_area}")
            elif auto_pixel_area_threshold:
                print(f"Auto pixel area threshold enabled (will adjust per scene)")

    def calculate_smart_pixel_area_threshold(self, config, world_coords, valid_mask):
        """
        Calculate smart pixel area threshold based on scene bounding box and image resolution

        Args:
            config: Camera configuration dictionary
            world_coords: World coordinates tensor [H, W, 3]
            valid_mask: Valid pixel mask [H, W]

        Returns:
            float: Maximum pixel area threshold
        """
        try:
            if world_coords is None or valid_mask is None:
                return 1.0  # Default fallback

            # Get valid world coordinates
            valid_coords = world_coords[valid_mask]

            if len(valid_coords) == 0:
                return 1.0  # Default fallback

            # Calculate bounding box of valid points
            min_coords = torch.min(valid_coords, dim=0)[0]
            max_coords = torch.max(valid_coords, dim=0)[0]
            bbox_size = max_coords - min_coords

            # Calculate bounding box volume
            bbox_volume = torch.prod(bbox_size).item()

            # Get image resolution
            h, w = world_coords.shape[:2]
            total_pixels = h * w
            valid_pixels = torch.sum(valid_mask).item()

            # Strategy: Estimate average pixel area if surface was evenly distributed
            # Then use a multiplier to filter out pixels with excessive area
            if bbox_volume > 0 and valid_pixels > 0:
                # Estimate average surface area per pixel
                estimated_surface_area = bbox_volume ** (2/3)  # Surface area scales as volume^(2/3)
                avg_pixel_area = estimated_surface_area / valid_pixels

                # Set threshold as 5x the average pixel area
                # This filters out pixels representing too much detail compression
                threshold = avg_pixel_area * 5.0

                # Apply reasonable bounds
                threshold = max(0.001, min(threshold, 100.0))

                return threshold
            else:
                return 1.0  # Default fallback

        except Exception as e:
            print(f"Error calculating smart pixel area threshold: {e}")
            return 1.0  # Default fallback

    def load_depth_image(self, depth_path):
        """Load depth image and convert to torch tensor"""
        try:
            depth_img = Image.open(depth_path)
            depth_array = np.array(depth_img, dtype=np.float32)

            if len(depth_array.shape) == 3:
                depth_array = depth_array[:, :, 0]

            # Normalize depth values (16-bit to 0-1)
            depth_array = depth_array / 65535.0
            depth_tensor = torch.from_numpy(depth_array).to(self.device)

            return depth_tensor

        except Exception as e:
            print(f"Error loading depth image {depth_path}: {e}")
            return None

    def load_config(self, config_path):
        """Load camera configuration from JSON"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config {config_path}: {e}")
            return None

    def depth_to_world_coordinates(self, depth_tensor, config):
        """Convert depth buffer to world coordinates"""
        try:
            h, w = depth_tensor.shape

            # Extract camera parameters
            camera_pos = config['position']['absolute']
            camera_pos = torch.tensor([camera_pos['x'], camera_pos['y'], camera_pos['z']],
                                    device=self.device, dtype=torch.float32)

            # Camera properties
            lens = config['camera_properties']['lens']
            sensor_width = config['camera_properties']['sensor_width']
            clip_start = config['camera_properties']['clip_start']
            clip_end = config['camera_properties']['clip_end']

            # Calculate focal length in pixels
            focal_length_px = (lens * w) / sensor_width

            # Create pixel coordinate grids
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=self.device, dtype=torch.float32),
                torch.arange(w, device=self.device, dtype=torch.float32),
                indexing='ij'
            )

            # Convert to normalized device coordinates
            x_ndc = (x_coords - w/2) / (w/2)
            y_ndc = (y_coords - h/2) / (h/2)

            # Convert depth from normalized to actual distance
            actual_depth = depth_tensor * (clip_end - clip_start) + clip_start

            # Create mask for valid depth values
            valid_mask = (depth_tensor > 0.001) & (depth_tensor < 0.999)

            # Calculate camera rays in camera space
            ray_x = x_ndc * actual_depth / focal_length_px * sensor_width
            ray_y = -y_ndc * actual_depth / focal_length_px * sensor_width
            ray_z = -actual_depth

            # Stack into world coordinates
            camera_space_coords = torch.stack([ray_x, ray_y, ray_z], dim=-1)
            world_coords = camera_space_coords + camera_pos.unsqueeze(0).unsqueeze(0)

            return world_coords, valid_mask

        except Exception as e:
            print(f"Error in depth to world conversion: {e}")
            return None, None

    def estimate_surface_area(self, world_coords, valid_mask, config):
        """Estimate surface area using neighboring pixel derivatives with pixel area filtering"""
        try:
            if world_coords is None or valid_mask is None:
                return 0.0, {}, valid_mask

            h, w, _ = world_coords.shape

            # Pad coordinates for gradient calculation
            coords_padded = torch.nn.functional.pad(world_coords.permute(2, 0, 1),
                                                  (1, 1, 1, 1), mode='replicate').permute(1, 2, 0)

            # Calculate gradients
            grad_x = coords_padded[1:-1, 2:, :] - coords_padded[1:-1, :-2, :]
            grad_y = coords_padded[2:, 1:-1, :] - coords_padded[:-2, 1:-1, :]

            # Calculate cross product for surface normals
            surface_vectors = torch.cross(grad_x, grad_y, dim=-1)

            # Calculate magnitude for area (each pixel's surface area)
            pixel_areas = torch.norm(surface_vectors, dim=-1) * 0.25

            # Determine pixel area threshold
            if self.auto_pixel_area_threshold:
                max_pixel_area = self.calculate_smart_pixel_area_threshold(config, world_coords, valid_mask)
            else:
                max_pixel_area = self.max_pixel_area if self.max_pixel_area is not None else float('inf')

            # Create pixel area filter mask
            pixel_area_valid = pixel_areas <= max_pixel_area

            # Combine with original valid mask
            final_valid_mask = valid_mask & pixel_area_valid

            # Apply combined mask to get final valid areas
            valid_areas = pixel_areas * final_valid_mask.float()

            # Sum total area
            total_area = torch.sum(valid_areas).item()

            # Calculate filtering statistics
            total_valid_pixels = torch.sum(valid_mask).item()
            area_filtered_pixels = torch.sum(final_valid_mask).item()
            area_filter_ratio = area_filtered_pixels / max(total_valid_pixels, 1)

            filter_stats = {
                'max_pixel_area_threshold': max_pixel_area,
                'pixels_before_area_filter': int(total_valid_pixels),
                'pixels_after_area_filter': int(area_filtered_pixels),
                'area_filter_ratio': area_filter_ratio
            }

            return total_area, filter_stats, final_valid_mask

        except Exception as e:
            print(f"Error in surface area estimation: {e}")
            return 0.0, {}, valid_mask

    def estimate_surface_area_signed(self, world_coords, valid_mask, normals, config):
        """
        Estimate signed surface area using neighboring pixel derivatives with pixel area filtering.
        Positive area for front-facing surfaces, negative for back-facing surfaces.

        Args:
            world_coords: World coordinates tensor [H, W, 3]
            valid_mask: Valid pixel mask [H, W]
            normals: Surface normals tensor [H, W, 3] (world space)
            config: Camera configuration dictionary

        Returns:
            tuple: (signed_surface_area, filter_stats, final_valid_mask)
        """
        try:
            if world_coords is None or valid_mask is None or normals is None:
                return 0.0, {}, valid_mask

            h, w, _ = world_coords.shape

            # Pad coordinates for gradient calculation
            coords_padded = torch.nn.functional.pad(world_coords.permute(2, 0, 1),
                                                  (1, 1, 1, 1), mode='replicate').permute(1, 2, 0)

            # Calculate gradients
            grad_x = coords_padded[1:-1, 2:, :] - coords_padded[1:-1, :-2, :]
            grad_y = coords_padded[2:, 1:-1, :] - coords_padded[:-2, 1:-1, :]

            # Calculate cross product for surface normals from gradients
            surface_vectors = torch.cross(grad_x, grad_y, dim=-1)

            # Calculate magnitude for area (each pixel's surface area)
            pixel_areas = torch.norm(surface_vectors, dim=-1) * 0.25

            # Determine pixel area threshold
            if self.auto_pixel_area_threshold:
                max_pixel_area = self.calculate_smart_pixel_area_threshold(config, world_coords, valid_mask)
            else:
                max_pixel_area = self.max_pixel_area if self.max_pixel_area is not None else float('inf')

            # Create pixel area filter mask
            pixel_area_valid = pixel_areas <= max_pixel_area

            # Calculate camera view direction for each pixel
            camera_pos = torch.tensor(config['position'], device=world_coords.device, dtype=torch.float32)
            view_directions = world_coords - camera_pos.unsqueeze(0).unsqueeze(0)
            view_directions = view_directions / (torch.norm(view_directions, dim=-1, keepdim=True) + 1e-8)

            # Normalize surface normals
            normals_normalized = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)

            # Calculate dot product between view direction and surface normal
            # Positive dot product means back-facing (normal points toward camera)
            # Negative dot product means front-facing (normal points away from camera)
            dot_products = torch.sum(view_directions * normals_normalized, dim=-1)

            # Create sign mask: +1 for front-facing, -1 for back-facing
            sign_mask = torch.where(dot_products < 0, 1.0, -1.0)

            # Apply sign to pixel areas
            signed_pixel_areas = pixel_areas * sign_mask

            # Combine with original valid mask and pixel area filter
            final_valid_mask = valid_mask & pixel_area_valid

            # Apply combined mask to get final valid areas
            valid_signed_areas = signed_pixel_areas * final_valid_mask.float()

            # Sum total signed area
            total_signed_area = torch.sum(valid_signed_areas).item()

            # Calculate statistics
            total_valid_pixels = torch.sum(valid_mask).item()
            area_filtered_pixels = torch.sum(final_valid_mask).item()
            area_filter_ratio = area_filtered_pixels / max(total_valid_pixels, 1)

            # Calculate front-facing vs back-facing statistics
            front_facing_mask = (dot_products < 0) & final_valid_mask
            back_facing_mask = (dot_products >= 0) & final_valid_mask
            front_facing_pixels = torch.sum(front_facing_mask).item()
            back_facing_pixels = torch.sum(back_facing_mask).item()
            front_facing_ratio = front_facing_pixels / max(area_filtered_pixels, 1)

            filter_stats = {
                'max_pixel_area_threshold': max_pixel_area,
                'pixels_before_area_filter': int(total_valid_pixels),
                'pixels_after_area_filter': int(area_filtered_pixels),
                'area_filter_ratio': area_filter_ratio,
                'front_facing_pixels': int(front_facing_pixels),
                'back_facing_pixels': int(back_facing_pixels),
                'front_facing_ratio': front_facing_ratio
            }

            return total_signed_area, filter_stats, final_valid_mask

        except Exception as e:
            print(f"Error in signed surface area estimation: {e}")
            return 0.0, {}, valid_mask

    def process_depth_and_config(self, depth_path, config_path):
        """Process a single depth-config pair"""
        try:
            # Load depth image
            depth_tensor = self.load_depth_image(depth_path)
            if depth_tensor is None:
                return None

            # Load camera config
            config = self.load_config(config_path)
            if config is None:
                return None

            # Convert depth to world coordinates
            world_coords, valid_mask = self.depth_to_world_coordinates(depth_tensor, config)
            if world_coords is None:
                return None

            # Estimate surface area
            surface_area, filter_stats, final_valid_mask = self.estimate_surface_area(world_coords, valid_mask, config)

            result = {
                'depth_path': depth_path,
                'config_path': config_path,
                'surface_area': surface_area,
                'filter_stats': filter_stats,
                'sample_id': config.get('sample_id', -1),
                'timestamp': time.time()
            }

            return result

        except Exception as e:
            print(f"Error processing {depth_path} and {config_path}: {e}")
            return None


class FolderMonitor:
    def __init__(self, watch_dir="tmp_renders", max_pixel_area=None, auto_pixel_area_threshold=True, socket_file="surface_area_socket.txt", verbose=False):
        """
        Initialize folder monitor with surface area calculator

        Args:
            watch_dir: Directory to monitor for depth files
            max_pixel_area: Maximum surface area per pixel (used if auto_pixel_area_threshold=False)
            auto_pixel_area_threshold: If True, automatically calculate threshold based on scene bounding box
            socket_file: File to write results for other workers to read
            verbose: Enable verbose output
        """
        self.watch_dir = os.path.abspath(watch_dir)
        self.socket_file = os.path.join(self.watch_dir, socket_file)
        self.calculator = SurfaceAreaCalculator(
            max_pixel_area=max_pixel_area,
            auto_pixel_area_threshold=auto_pixel_area_threshold,
            verbose=verbose
        )
        self.processed_pairs = set()
        self.verbose = verbose

        if verbose:
            print(f"Monitoring directory: {self.watch_dir}")
            print(f"Socket file: {self.socket_file}")

    def scan_and_process(self):
        """Scan directory for matching depth and config file pairs"""
        try:
            if not os.path.exists(self.watch_dir):
                return

            files = os.listdir(self.watch_dir)

            # Group files by timestamp
            timestamp_groups = {}
            for file in files:
                if file.endswith('_config.json'):
                    timestamp = file.replace('_config.json', '')
                    if timestamp not in timestamp_groups:
                        timestamp_groups[timestamp] = {}
                    timestamp_groups[timestamp]['config'] = os.path.join(self.watch_dir, file)
                elif file.endswith('_depth0001.png'):
                    timestamp = file.replace('_depth0001.png', '')
                    if timestamp not in timestamp_groups:
                        timestamp_groups[timestamp] = {}
                    timestamp_groups[timestamp]['depth'] = os.path.join(self.watch_dir, file)

            # Process complete pairs
            for timestamp, files_dict in timestamp_groups.items():
                if 'config' in files_dict and 'depth' in files_dict:
                    if timestamp not in self.processed_pairs:
                        self.process_pair(files_dict['depth'], files_dict['config'], timestamp)

        except Exception as e:
            print(f"Error scanning directory: {e}")

    def process_pair(self, depth_path, config_path, timestamp):
        """Process a matching depth/config pair and write to socket file"""
        try:
            result = self.calculator.process_depth_and_config(depth_path, config_path)

            if result:
                self.processed_pairs.add(timestamp)

                # Write to socket file in the requested format
                self.write_to_socket(result)

                # Remove processed files
                try:
                    os.remove(depth_path)
                    os.remove(config_path)
                    if self.verbose:
                        print(f"  Removed processed files for timestamp {timestamp}")
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Could not remove files: {e}")

        except Exception as e:
            print(f"Error processing pair {timestamp}: {e}")

    def write_to_socket(self, result):
        """Write result to socket file in the format: camera_pos_x,y,z camera_view_dir_x,y,z surface_area"""
        try:
            camera_pos = result['camera_position']
            view_dir = result['view_direction']
            surface_area = result['surface_area']

            # Format: pos_x pos_y pos_z view_x view_y view_z surface_area
            line = f"{camera_pos['x']} {camera_pos['y']} {camera_pos['z']} {view_dir['x']} {view_dir['y']} {view_dir['z']} {surface_area}\n"

            # Append to socket file
            with open(self.socket_file, 'a') as f:
                f.write(line)
                f.flush()  # Ensure data is written immediately

            if self.verbose:
                print(f"  Written to socket: pos({camera_pos['x']:.3f}, {camera_pos['y']:.3f}, {camera_pos['z']:.3f}) "
                      f"view({view_dir['x']:.3f}, {view_dir['y']:.3f}, {view_dir['z']:.3f}) area({surface_area:.6f})")

        except Exception as e:
            print(f"Error writing to socket file: {e}")


def parse_arguments():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Surface Area Worker for processing depth buffers')

    parser.add_argument('--watch_dir', type=str, default='tmp_renders',
                        help='Directory to monitor for depth files (default: tmp_renders)')

    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with detailed progress messages')

    return parser.parse_args()

def main():
    """Main worker function"""
    args = parse_arguments()

    print("Surface Area Worker Started")
    print("="*50)
    print(f"Monitoring: {os.path.abspath(args.watch_dir)}")
    print(f"Verbose: {'Yes' if args.verbose else 'No'}")
    print("Waiting for depth buffers and config files...")
    print("Press Ctrl+C to stop")
    print("="*50)

    monitor = FolderMonitor(args.watch_dir, verbose=args.verbose)

    try:
        while True:
            monitor.scan_and_process()
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down worker...")

    # Print final summary
    print(f"\nProcessed {len(monitor.processed_pairs)} depth buffer pairs")
    print(f"Socket file location: {monitor.socket_file}")
    if os.path.exists(monitor.socket_file):
        print("Results written to socket file for other workers to read")
    else:
        print("No results were processed")


if __name__ == "__main__":
    main()
