# Basic Falcor Rasterizer

A simple Python-based rasterization renderer built on top of Falcor that uniformly samples camera positions and directions within a scene's bounding box.

## Features

- **Uniform Camera Sampling**: Randomly samples camera positions within the scene bounding box and directions on the unit sphere
- **Rasterization-based Rendering**: Uses Falcor's GBufferRaster pass for fast rasterization (not ray tracing)
- **Multiple Output Buffers**: Captures diffuse color, depth, and world-space normals
- **Performance Tracking**: Measures and reports rendering performance statistics
- **Automatic Output Management**: Clears and recreates output directories automatically
- **Flexible Configuration**: Supports various scene formats and customizable render settings

## Requirements

- Falcor with Python bindings
- Required Python packages:
  - `numpy`
  - `torch`
  - `pyexr`
  - `Pillow (PIL)`
  - `PyYAML`

## Installation

1. Make sure Falcor is built with Python bindings enabled
2. Install required Python packages:
   ```bash
   pip install numpy torch pyexr Pillow PyYAML
   ```
3. Update the `config.yaml` file with your model paths

## Usage

### Basic Command Line Usage

```bash
# Render a scene with default settings (10 samples, 512x512 resolution)
python basic_rasterizer.py scene_file.pyscene

# Custom settings
python basic_rasterizer.py scene_file.pyscene --num_samples 20 --width 1024 --height 768 --output_dir my_renders

# With random seed for reproducibility
python basic_rasterizer.py scene_file.pyscene --seed 42 --verbose
```

### Command Line Arguments

- `scene_file`: Path to the scene file (.pyscene, .fbx, .gltf, etc.)
- `--num_samples`: Number of random camera samples to render (default: 10)
- `--output_dir`: Output directory name (default: "tmp_render")
- `--width`: Render width in pixels (default: 512)
- `--height`: Render height in pixels (default: 512)
- `--seed`: Random seed for reproducible results
- `--verbose`: Enable verbose output

### Python API Usage

```python
from basic_rasterizer import BasicRasterizer

# Create renderer
renderer = BasicRasterizer(
    render_width=1024,
    render_height=768,
    verbose=True
)

# Setup and render
renderer.setup_falcor()
renderer.load_scene("path/to/scene.pyscene")
renderer.setup_render_graph()
renderer.render_samples(num_samples=20, output_dir="my_output")
```

### Custom Camera Sampling

```python
# Manual camera control
output_dir = renderer.prepare_output_directory("custom_renders")

for i in range(10):
    # Sample camera position and direction
    camera_pos = renderer.sample_camera_position(margin_factor=0.3)
    camera_dir = renderer.sample_camera_direction()

    # Set camera
    renderer.set_camera(camera_pos, camera_dir)

    # Render frame
    outputs, render_time = renderer.render_frame()

    # Save with custom info
    camera_info = {
        "sample_id": i,
        "position": camera_pos.tolist(),
        "direction": camera_dir.tolist(),
        "render_time": render_time
    }
    renderer.save_outputs(outputs, camera_info, output_dir, i)
```

## Output Files

The renderer generates the following files for each sample:

- `sample_XXXX_TIMESTAMP_diffuse.png`: Diffuse color image (8-bit PNG)
- `sample_XXXX_TIMESTAMP_depth.exr`: Depth buffer (32-bit EXR)
- `sample_XXXX_TIMESTAMP_normals.exr`: World-space normals (32-bit EXR)
- `sample_XXXX_TIMESTAMP_camera.json`: Camera configuration and metadata

### Camera JSON Format

```json
{
  "sample_id": 0,
  "position": [x, y, z],
  "direction": [dx, dy, dz],
  "target": [tx, ty, tz],
  "render_time_seconds": 0.123,
  "scene_bounds": {
    "min": [minx, miny, minz],
    "max": [maxx, maxy, maxz]
  }
}
```

## Example Scenes

The renderer supports various scene formats:

- **Falcor Python Scenes** (`.pyscene`): Native Falcor scene format
- **glTF** (`.gltf`, `.glb`): Standard 3D format
- **FBX** (`.fbx`): Autodesk format
- **OBJ** (`.obj`): Wavefront format

## Performance Notes

- **Rasterization vs Ray Tracing**: This renderer uses rasterization for speed, not ray tracing
- **GPU Memory**: Large scenes may require significant GPU memory
- **Resolution Impact**: Higher resolutions significantly impact performance
- **Scene Complexity**: Complex scenes with many materials/textures will render slower

## Performance Statistics

The renderer automatically tracks and reports:

- Total render time
- Average time per sample
- Min/max render times
- Samples per second

Example output:
```
Performance Statistics:
  Total samples: 10
  Total time: 12.345s
  Average time per sample: 1.235s
  Min time: 1.123s
  Max time: 1.456s
  Samples per second: 0.81
```

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure Falcor Python bindings are installed
2. **Scene Loading Failed**: Check scene file path and format
3. **GPU Memory Error**: Reduce render resolution or use simpler scenes
4. **No Output Generated**: Check file permissions and output directory

### Debug Tips

- Use `--verbose` flag for detailed output
- Check the console for error messages
- Verify scene file exists and is readable
- Make sure GPU drivers are updated

## Configuration

Edit `config.yaml` to customize default settings:

```yaml
ExperimentParams:
  MODELS_PATH: "path/to/your/models/"

BasicRasterizer:
  render_width: 512
  render_height: 512
  default_samples: 10
  output_dir: "tmp_render"
  margin_factor: 0.2

Performance:
  verbose: true
  track_timing: true
```

## Examples

See `example_usage.py` for more detailed usage examples:

```bash
# Run examples with a specific scene
python example_usage.py path/to/scene.pyscene

# Run batch rendering examples
python example_usage.py
```

## License

This project follows the same license as Falcor.
