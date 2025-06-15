# Async Boolean Check System for Object Detection

This module provides async functionality to check for object presence in images using the existing MistralVisionAPI from `mistral_vision_inference.py`. It generates presence prompts based on object names and performs efficient batch processing with async capabilities.

## Features

- **Async Processing**: Efficient concurrent API calls using `asyncio` and thread pools
- **Existing API Integration**: Uses the existing `MistralVisionAPI` class functions
- **Object Name Integration**: Automatically generates presence prompts using object names as arguments
- **Batch Processing**: Check multiple images and objects simultaneously
- **Custom Prompts**: Support for custom prompts or automatic prompt generation
- **Confidence Scoring**: Intelligent parsing of API responses with confidence levels
- **Error Handling**: Robust error handling and logging
- **Rate Limiting**: Configurable concurrent request limits
- **Pattern File Support**: Works with existing pattern JSON files

## Installation

1. Make sure you have the existing `mistral_vision_inference.py` file in the same directory
2. Install the required dependencies:
```bash
pip install -r requirements_async.txt
```

3. Set your Mistral API key:
```bash
export MISTRAL_API_KEY="your_api_key_here"
```

## Quick Start

### Basic Usage

```python
import asyncio
from async_check_boolean import AsyncMistralVisionAPI

async def check_for_cat():
    api = AsyncMistralVisionAPI()
    
    # Check for a cat in an image
    result = await api.check_object_presence_async("path/to/image.jpg", "cat")
    
    print(f"Cat present: {result.is_present}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Response: {result.response_text}")

asyncio.run(check_for_cat())
```

### Using Pattern Files

```python
async def check_with_patterns():
    # Use a specific patterns file
    api = AsyncMistralVisionAPI(patterns_file="search_patterns/bug.json")
    
    result = await api.check_object_presence_async("path/to/image.jpg", "bug")
    print(f"Bug detected: {result.is_present}")

asyncio.run(check_with_patterns())
```

### Multiple Objects in One Image

```python
async def check_multiple_objects():
    api = AsyncMistralVisionAPI()
    
    object_names = ["cat", "dog", "bird"]
    results = await api.check_multiple_objects_async("path/to/image.jpg", object_names)
    
    for result in results:
        status = "Present" if result.is_present else "Absent"
        print(f"{result.object_name}: {status} (confidence: {result.confidence:.2f})")
```

### Batch Processing

```python
async def batch_check():
    api = AsyncMistralVisionAPI()
    
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    batch_result = await api.batch_check_object_presence(image_paths, "bug")
    
    print(f"Bug found in {batch_result.present_count}/{batch_result.total_images} images")
    print(f"Average confidence: {batch_result.average_confidence:.2f}")
```

## API Reference

### AsyncMistralVisionAPI Class

#### Constructor
```python
AsyncMistralVisionAPI(api_key=None, config_path="config.json", patterns_file=None)
```

- `api_key`: Mistral API key (optional, can use MISTRAL_API_KEY env var)
- `config_path`: Path to configuration JSON file
- `patterns_file`: Path to JSON file containing object patterns

#### Methods

##### `generate_presence_prompt(object_name, pattern_index=0)`
Generate a presence prompt for a specific object using existing patterns.

**Parameters:**
- `object_name`: Name of the object to check for
- `pattern_index`: Index of the pattern to use

**Returns:** Formatted presence prompt string

##### `generate_presence_prompts(object_name, num_prompts=5)`
Generate multiple presence prompts for an object.

**Parameters:**
- `object_name`: Name of the object to check for
- `num_prompts`: Number of prompts to generate

**Returns:** List of presence prompts

##### `check_object_presence_async(image_path, object_name, prompt=None)`
Check for object presence in a single image asynchronously.

**Parameters:**
- `image_path`: Path to the image file
- `object_name`: Name of the object to check for
- `prompt`: Custom prompt (optional, will generate if None)

**Returns:** `ObjectCheckResult` object

##### `check_multiple_objects_async(image_path, object_names, prompts=None)`
Check for multiple objects in a single image asynchronously.

**Parameters:**
- `image_path`: Path to the image file
- `object_names`: List of object names to check for
- `prompts`: Dictionary mapping object names to custom prompts

**Returns:** List of `ObjectCheckResult` objects

##### `batch_check_object_presence(image_paths, object_name, prompt=None, max_concurrent=5)`
Check for object presence across multiple images asynchronously.

**Parameters:**
- `image_paths`: List of image file paths
- `object_name`: Name of the object to check for
- `prompt`: Custom prompt (optional)
- `max_concurrent`: Maximum number of concurrent requests

**Returns:** `BatchCheckResult` object

##### `batch_check_multiple_objects(image_paths, object_names, prompts=None, max_concurrent=5)`
Check for multiple objects across multiple images asynchronously.

**Parameters:**
- `image_paths`: List of image file paths
- `object_names`: List of object names to check for
- `prompts`: Dictionary mapping object names to custom prompts
- `max_concurrent`: Maximum number of concurrent requests per object

**Returns:** Dictionary mapping object names to `BatchCheckResult` objects

##### `get_object_name()`
Get the object name from the existing API.

**Returns:** Object name string

##### `get_patterns()`
Get the patterns from the existing API.

**Returns:** List of pattern strings

### Data Classes

#### ObjectCheckResult
```python
@dataclass
class ObjectCheckResult:
    object_name: str
    image_path: str
    is_present: bool
    confidence: float
    response_text: str
    processing_time: float
    prompt_used: str
```

#### BatchCheckResult
```python
@dataclass
class BatchCheckResult:
    object_name: str
    results: List[ObjectCheckResult]
    total_images: int
    present_count: int
    absent_count: int
    average_confidence: float
    total_processing_time: float
```

## Integration with Existing Code

The async system is built on top of the existing `MistralVisionAPI` class, so it:

- Uses the same configuration files (`config.json`)
- Supports the same pattern files (JSON files in `search_patterns/`)
- Maintains compatibility with existing code
- Adds async capabilities without breaking existing functionality

### Using Existing Pattern Files

```python
# Use existing bug patterns
api = AsyncMistralVisionAPI(patterns_file="search_patterns/bug.json")

# Use existing needle patterns  
api = AsyncMistralVisionAPI(patterns_file="search_patterns/needle.json")

# Use default patterns
api = AsyncMistralVisionAPI()
```

## Examples

### Example 1: Single Object Detection
```python
import asyncio
from async_check_boolean import AsyncMistralVisionAPI

async def detect_cat():
    api = AsyncMistralVisionAPI()
    result = await api.check_object_presence_async("cat_photo.jpg", "cat")
    
    if result.is_present:
        print(f"Cat detected with {result.confidence:.2f} confidence!")
    else:
        print("No cat found in the image.")

asyncio.run(detect_cat())
```

### Example 2: Multiple Objects with Custom Prompts
```python
async def detect_animals():
    api = AsyncMistralVisionAPI()
    
    custom_prompts = {
        "cat": "Is there a domestic cat or feline visible?",
        "dog": "Can you see any dog or canine animal?",
        "bird": "Are there any birds or flying animals?"
    }
    
    results = await api.check_multiple_objects_async(
        "park_photo.jpg", 
        ["cat", "dog", "bird"], 
        custom_prompts
    )
    
    for result in results:
        print(f"{result.object_name}: {'✓' if result.is_present else '✗'}")

asyncio.run(detect_animals())
```

### Example 3: Batch Processing with Rate Limiting
```python
async def batch_detect_bugs():
    api = AsyncMistralVisionAPI(patterns_file="search_patterns/bug.json")
    
    image_paths = [f"sample_{i}.jpg" for i in range(1, 11)]
    batch_result = await api.batch_check_object_presence(
        image_paths, 
        "bug", 
        max_concurrent=3
    )
    
    print(f"Bugs found in {batch_result.present_count} out of {batch_result.total_images} images")
    print(f"Average confidence: {batch_result.average_confidence:.2f}")

asyncio.run(batch_detect_bugs())
```

## Error Handling

The system includes comprehensive error handling:

- **API Errors**: Network issues and API failures are caught and logged
- **File Errors**: Missing or invalid image files are handled gracefully
- **Rate Limiting**: Automatic retry logic for rate-limited requests
- **Timeout Handling**: Configurable timeouts for API requests

## Performance Tips

1. **Concurrent Requests**: Adjust `max_concurrent` based on your API rate limits
2. **Batch Size**: Process images in batches for better performance
3. **Image Size**: Optimize image sizes before processing
4. **Caching**: Consider caching results for repeated checks

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure `MISTRAL_API_KEY` is set correctly
2. **Image Not Found**: Verify image paths are correct and files exist
3. **Rate Limiting**: Reduce `max_concurrent` if you hit rate limits
4. **Memory Issues**: Process images in smaller batches for large datasets
5. **Missing mistral_vision_inference.py**: Ensure the file is in the same directory

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Running the Examples

To run the provided examples:

```bash
python example_async_usage.py
```

Make sure to:
1. Set your `MISTRAL_API_KEY` environment variable
2. Provide valid image paths in the examples
3. Have the `mistral_vision_inference.py` file in the same directory

## Migration from Existing Code

If you're already using the `MistralVisionAPI` class, you can easily add async capabilities:

```python
# Existing synchronous code
from mistral_vision_inference import MistralVisionAPI

api = MistralVisionAPI()
result = api.check_object_presence("image.jpg", "Is there a cat?")

# New async code
from async_check_boolean import AsyncMistralVisionAPI

async def check_async():
    api = AsyncMistralVisionAPI()
    result = await api.check_object_presence_async("image.jpg", "cat")
    return result

# Use in synchronous code
import asyncio
result = asyncio.run(check_async())
```

## Key Differences from Original Implementation

1. **Async Wrapper**: This is an async wrapper around the existing `MistralVisionAPI`
2. **Thread Pool**: Uses `asyncio.run_in_executor()` to make synchronous calls async
3. **Enhanced Results**: Adds confidence scoring and detailed result objects
4. **Pattern Integration**: Leverages existing pattern files and generation methods
5. **Compatibility**: Maintains full compatibility with existing code 