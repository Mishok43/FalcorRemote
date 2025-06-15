# Image Inference Database Script

This script processes images from a specified folder, runs scene description inferences using the Mistral Vision API, extracts objects from the descriptions, and saves the results to an SQLite database or JSON file.

## Features

- **Scene Description**: Generates detailed descriptions of what's visible in each image
- **Object Extraction**: Automatically extracts objects from scene descriptions (more efficient than checking predefined lists)
- **Multiple Output Formats**: Save results to SQLite database or JSON file
- **Batch Processing**: Processes multiple images efficiently
- **Error Handling**: Gracefully handles API errors and missing files

## Output Formats

### SQLite Database
Creates a table with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `scene_path` | TEXT | Path to the scene/object file |
| `image_path` | TEXT | Path to the image file |
| `scene_description` | TEXT | Detailed description of the scene |
| `objects_present` | TEXT | JSON array of objects extracted from the description |
| `inference_timestamp` | DATETIME | When the inference was performed |

### JSON File
Creates a structured JSON file with metadata and results:

```json
{
  "metadata": {
    "total_results": 2,
    "created_at": "2024-01-15T10:30:00",
    "version": "1.0"
  },
  "results": [
    {
      "scene_path": "scene.obj",
      "image_path": "images/depth_2.0.png",
      "scene_description": "A dark room with depth map showing...",
      "objects_present": ["wall", "floor", "light", "room"],
      "inference_timestamp": "2024-01-15T10:30:00"
    }
  ]
}
```

## Prerequisites

1. **Mistral API Key**: Set the `MISTRAL_API_KEY` environment variable or pass it to the script
2. **Python Dependencies**: The script uses the existing `mistral_vision_inference.py` module
3. **Images**: Place your images in a folder (default: `images/`)

## Usage

### Basic Usage

```bash
# Process images and save to SQLite database (default)
python create_inference_db.py

# Process images and save to JSON file
python create_inference_db.py --output-format json

# Process images in a specific folder
python create_inference_db.py --images-dir /path/to/images

# Specify a scene path (useful for 3D object files)
python create_inference_db.py --scene-path "scene.obj" --images-dir images
```

### Advanced Options

```bash
# Use custom file paths
python create_inference_db.py \
    --output-format json \
    --json-path my_results.json \
    --images-dir /path/to/images

# Use custom database path
python create_inference_db.py \
    --output-format sql \
    --db-path my_results.db \
    --images-dir /path/to/images

# Use a custom configuration file
python create_inference_db.py --config-file my_config.json

# Show results table after processing
python create_inference_db.py --show-results

# Combine multiple options
python create_inference_db.py \
    --images-dir /path/to/images \
    --scene-path "my_scene.obj" \
    --output-format json \
    --json-path results.json \
    --show-results
```

### Command Line Arguments

- `--images-dir, -i`: Path to directory containing images (default: "images")
- `--scene-path, -s`: Path to the scene/object file (default: "unknown")
- `--output-format, -f`: Output format: sql or json (default: sql)
- `--db-path, -d`: Path to SQLite database file (default: "inference_results.db")
- `--json-path, -j`: Path to JSON results file (default: "inference_results.json")
- `--config-file, -c`: Path to configuration file (default: "config.json")
- `--show-results, -r`: Show results table after processing

## Configuration

The `config.json` file contains API settings:

```json
{
    "model": "mistral-large-latest",
    "max_tokens": 1000,
    "temperature": 0.7
}
```

## Example Output

```
Looking for images in: images
Found 2 image files:
  images/depth_2.0.png
  images/scene_1.jpg

Initializing JSON output: inference_results.json
Initializing image processor...

Processing 2 images...

Processing image 1/2
Processing: images/depth_2.0.png
  Getting scene description...
  Extracting objects from description...

Processing image 2/2
Processing: images/scene_1.jpg
  Getting scene description...
  Extracting objects from description...

Saving results...
Saved 2 results to inference_results.json

Processing completed!
Total time: 15.23 seconds
Average time per image: 7.62 seconds
Results saved to: inference_results.json

========================================================================================================================
INFERENCE RESULTS (JSON)
========================================================================================================================
Scene Path                    | Image Path                     | Scene Description                    | Objects Present
------------------------------------------------------------------------------------------------------------------------
scene.obj                     | images/depth_2.0.png           | A dark room with depth map showing... | wall, floor, light, room
scene.obj                     | images/scene_1.jpg             | A modern office space with a desk... | desk, chair, computer, office
------------------------------------------------------------------------------------------------------------------------
Total records: 2
```

## Database Queries (SQLite)

You can query the SQLite database directly:

```bash
# View all results
sqlite3 inference_results.db "SELECT * FROM scene_inferences;"

# Find images containing specific objects
sqlite3 inference_results.db "SELECT image_path FROM scene_inferences WHERE objects_present LIKE '%chair%';"

# Get results for a specific scene
sqlite3 inference_results.db "SELECT * FROM scene_inferences WHERE scene_path = 'scene.obj';"
```

## JSON File Processing

You can process the JSON results with any programming language:

```python
import json

# Load results
with open('inference_results.json', 'r') as f:
    data = json.load(f)

# Access metadata
print(f"Total results: {data['metadata']['total_results']}")

# Process results
for result in data['results']:
    print(f"Image: {result['image_path']}")
    print(f"Objects: {result['objects_present']}")
```

## Error Handling

The script handles various error conditions:

- Missing images directory
- API errors during inference
- Invalid image files
- Database connection issues (SQLite)
- JSON file corruption

Errors are logged and the script continues processing other images.

## Performance

- **Efficient Processing**: Only 2 API calls per image (1 for description + 1 for object extraction)
- **Comprehensive Object Detection**: Discovers objects automatically rather than checking a predefined list
- **Faster Execution**: Significantly faster than checking 27+ predefined objects individually
- **Better Coverage**: Can identify objects that weren't in the original predefined list

## How It Works

1. **Scene Description**: The model generates a detailed description of what's visible in the image
2. **Object Extraction**: A second prompt asks the model to list all objects as a comma-separated list
3. **Parsing**: The response is parsed to extract individual object names
4. **Filtering**: Common words and articles are filtered out to keep only actual objects
5. **Storage**: Results are stored in the chosen format (SQLite or JSON) with timestamps

## Choosing Between SQLite and JSON

**Use SQLite when:**
- You need complex queries and filtering
- You want to integrate with existing database workflows
- You need ACID compliance and transaction support
- You're working with large datasets

**Use JSON when:**
- You want human-readable output
- You need to share results with other applications
- You prefer simple file-based storage
- You want to process results in other programming languages
- You need to version control your results 