# New Features Summary

## Overview
The basic rasterizer and API server have been updated with several new features to support more flexible scene management, sample rendering with albedo output, and customizable training parameters.

## New Features

### 1. Scene Management

#### BasicRasterizer Methods:
- `get_available_scenes()` - Returns dictionary of available scenes with existence status
- `set_current_scene(scene_name)` - Sets the current scene by name

#### API Endpoints:
- `GET /scenes` - List available scenes with existence status
- `POST /setup_renderer` - Setup renderer instance (replaces setup_scene)
- `POST /set_scene` - Set current scene for a renderer session

#### Example Usage:
```python
# Get available scenes
scenes = renderer.get_available_scenes()
print(f"Available scenes: {list(scenes.keys())}")

# Set current scene
success = renderer.set_current_scene("cornell_box")
```

### 2. Sample Rendering with Albedo

#### BasicRasterizer Method:
- `sample_render_positions(num_samples, nn_sampling=False, return_albedo=True)`
  - Returns list of render results with camera positions (absolute & normalized)
  - Includes albedo render data as uint8 arrays
  - Supports both random and neural network sampling

#### API Endpoint:
- `POST /sample_render` - Sample render positions with albedo returns

#### Response Format:
```json
{
  "sample_id": 0,
  "camera_position_absolute": [x, y, z],
  "camera_direction_absolute": [dx, dy, dz],
  "camera_position_normalized": [nx, ny, nz],
  "camera_direction_normalized": [theta, phi],
  "surface_area": 0.123456,
  "render_time": 0.045,
  "sampling_method": "random|neural_network",
  "albedo_render": [...], // uint8 array data
  "albedo_shape": [height, width, channels]
}
```

### 3. Custom Training Parameters

#### BasicRasterizer Method:
- `train_nf_sampler_steps(num_steps, **training_params)`
  - Supports custom learning_rate, epochs_per_fit, batch_size, hidden_units, hidden_layers
  - Returns detailed training statistics

#### API Endpoint:
- `POST /train` - Train NFSampler with custom parameters

#### Supported Parameters:
- `num_steps` - Number of training steps
- `learning_rate` - Learning rate for optimizer
- `epochs_per_fit` - Epochs per training fit
- `batch_size` - Training batch size
- `hidden_units` - Number of hidden units in network
- `hidden_layers` - Number of hidden layers

### 4. Neural Network Sampling Toggle

Both the BasicRasterizer and API now support toggling between random and neural network sampling:
- `nn_sampling=False` - Use random sampling
- `nn_sampling=True` - Use trained neural network sampling (if available)

## API Workflow

### New Recommended Workflow:
1. `GET /scenes` - See available scenes
2. `POST /setup_renderer` - Setup renderer instance
3. `POST /set_scene` - Set current scene
4. `POST /sample_render` - Sample positions with albedo (random sampling)
5. `POST /train` - Train NFSampler with custom parameters
6. `POST /sample_render` - Sample positions with NN sampling

### Example API Usage:
```python
# Setup renderer
response = requests.post("/setup_renderer", json={
    "render_width": 512,
    "render_height": 512,
    "use_nf_sampler": True
})
session_id = response.json()['session_id']

# Set scene
requests.post("/set_scene", json={
    "session_id": session_id,
    "scene_name": "cornell_box"
})

# Sample render with albedo
requests.post("/sample_render", json={
    "session_id": session_id,
    "num_samples": 10,
    "nn_sampling": False,
    "return_albedo": True
})

# Train with custom parameters
requests.post("/train", json={
    "session_id": session_id,
    "num_steps": 100,
    "learning_rate": 0.001,
    "batch_size": 32
})

# Sample with NN sampling
requests.post("/sample_render", json={
    "session_id": session_id,
    "num_samples": 10,
    "nn_sampling": True,
    "return_albedo": True
})
```

## Testing

### Test Scripts:
- `test_new_features.py` - Tests all new features locally
- `api_client_example.py` - Updated with new API features

### Running Tests:
```bash
# Test locally
python ours/test_new_features.py

# Test API (start server first)
python ours/api_server.py
python ours/api_client_example.py
```

## Backward Compatibility

The API maintains backward compatibility:
- Legacy `/render` endpoint still works
- Old `/setup_scene` functionality moved to `/setup_renderer` + `/set_scene`
- All existing parameters and responses are preserved

## Performance Notes

- Albedo renders are returned as uint8 arrays to reduce JSON size
- GPU memory is managed carefully during rendering
- Training parameters can be tuned for performance vs. quality trade-offs
- Neural network sampling provides better surface area coverage after training
