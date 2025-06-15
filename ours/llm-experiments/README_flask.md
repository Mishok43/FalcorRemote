# Flask Image Processing Service

A Flask web service that provides HTTP endpoints for processing images with Mistral Vision API and extracting objects. This service wraps the functionality of the `create_inference_db.py` script into a web API.

## Features

- **HTTP Endpoints**: RESTful API for image processing
- **File Upload**: Upload single or multiple images
- **Async Processing**: Concurrent image processing with configurable limits
- **Multiple Output Formats**: Save results to JSON or SQLite
- **Health Monitoring**: Service health check endpoint
- **File Download**: Download results as files
- **Error Handling**: Comprehensive error handling and validation
- **Threaded**: Multi-threaded for handling multiple requests

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_flask.txt
```

### 2. Set Environment Variables

```bash
export MISTRAL_API_KEY="your_mistral_api_key_here"
```

### 3. Run the Service

```bash
python flask_service.py
```

The service will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check
- **URL**: `GET /health`
- **Description**: Check service health and configuration
- **Response**: Service status, API key configuration, processor status

### 2. Process Single Image
- **URL**: `POST /process-image`
- **Description**: Process a single image and extract objects
- **Form Data**:
  - `file`: Image file (required)
  - `scene_path`: Scene/object file path (optional, default: "unknown")
  - `output_format`: "json" or "sql" (optional, default: "json")
  - `db_path`: SQLite database path (optional, default: "inference_results.db")
  - `json_path`: JSON results file path (optional, default: "inference_results.json")

### 3. Process Multiple Images
- **URL**: `POST /process-batch`
- **Description**: Process multiple images concurrently
- **Form Data**:
  - `files`: Multiple image files (required)
  - `scene_path`: Scene/object file path (optional, default: "unknown")
  - `output_format`: "json" or "sql" (optional, default: "json")
  - `db_path`: SQLite database path (optional, default: "inference_results.db")
  - `json_path`: JSON results file path (optional, default: "inference_results.json")
  - `max_concurrent`: Maximum concurrent requests (optional, default: 3)

### 4. Get Results
- **URL**: `GET /results`
- **Description**: Retrieve all processing results
- **Query Parameters**:
  - `output_format`: "json" or "sql" (optional, default: "json")
  - `db_path`: SQLite database path (optional, default: "inference_results.db")
  - `json_path`: JSON results file path (optional, default: "inference_results.json")

### 5. Download Results
- **URL**: `GET /download-results`
- **Description**: Download results as a file
- **Query Parameters**:
  - `output_format`: "json" or "sql" (optional, default: "json")
  - `db_path`: SQLite database path (optional, default: "inference_results.db")
  - `json_path`: JSON results file path (optional, default: "inference_results.json")

### 6. API Information
- **URL**: `GET /`
- **Description**: Get API information and usage examples

## Usage Examples

### Using curl

#### Process Single Image
```bash
curl -X POST "http://localhost:5000/process-image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg" \
  -F "scene_path=scene.obj" \
  -F "output_format=json"
```

#### Process Multiple Images
```bash
curl -X POST "http://localhost:5000/process-batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/image1.jpg" \
  -F "files=@/path/to/image2.jpg" \
  -F "scene_path=scene.obj" \
  -F "max_concurrent=3"
```

#### Get Results
```bash
curl -X GET "http://localhost:5000/results?output_format=json"
```

#### Download Results
```bash
curl -X GET "http://localhost:5000/download-results?output_format=json" \
  -o results.json
```

### Using Python requests

```python
import requests

# Process single image
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'scene_path': 'scene.obj',
        'output_format': 'json'
    }
    response = requests.post('http://localhost:5000/process-image', 
                           files=files, data=data)
    result = response.json()
    print(result)

# Process multiple images
with open('image1.jpg', 'rb') as f1, open('image2.jpg', 'rb') as f2:
    files = [
        ('files', f1),
        ('files', f2)
    ]
    data = {
        'scene_path': 'scene.obj',
        'max_concurrent': 3
    }
    response = requests.post('http://localhost:5000/process-batch', 
                           files=files, data=data)
    result = response.json()
    print(result)

# Download results
response = requests.get('http://localhost:5000/download-results?output_format=json')
with open('downloaded_results.json', 'wb') as f:
    f.write(response.content)
```

### Using JavaScript/Fetch

```javascript
// Process single image
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('scene_path', 'scene.obj');
formData.append('output_format', 'json');

fetch('http://localhost:5000/process-image', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// Process multiple images
const formData = new FormData();
for (let file of fileInput.files) {
    formData.append('files', file);
}
formData.append('scene_path', 'scene.obj');
formData.append('max_concurrent', 3);

fetch('http://localhost:5000/process-batch', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// Download results
fetch('http://localhost:5000/download-results?output_format=json')
.then(response => response.blob())
.then(blob => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'results.json';
    a.click();
});
```

## Response Formats

### Process Image Response
```json
{
  "success": true,
  "scene_path": "scene.obj",
  "image_path": "image.jpg",
  "scene_description": "A modern office with a desk and computer...",
  "objects_present": ["desk", "computer", "chair", "office"],
  "processing_time": 2.5,
  "message": "Image processed successfully"
}
```

### Batch Process Response
```json
{
  "success": true,
  "total_images": 3,
  "processed_images": 3,
  "failed_images": 0,
  "total_time": 7.2,
  "average_time_per_image": 2.4,
  "results": [
    {
      "success": true,
      "scene_path": "scene.obj",
      "image_path": "image1.jpg",
      "scene_description": "...",
      "objects_present": ["desk", "computer"],
      "processing_time": 2.4,
      "message": "Processed successfully"
    }
  ],
  "message": "Processed 3 images successfully, 0 failed"
}
```

### Health Check Response
```json
{
  "status": "healthy",
  "api_key_configured": true,
  "config_loaded": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Error Response
```json
{
  "error": "Error message description"
}
```

## Configuration

### Environment Variables
- `MISTRAL_API_KEY`: Your Mistral API key (required)

### Configuration File
The service uses the same `config.json` file as the command-line script:

```json
{
    "model": "mistral-large-latest",
    "max_tokens": 1000,
    "temperature": 0.7
}
```

### Flask Configuration
- **Max file size**: 16MB (configurable)
- **Allowed extensions**: png, jpg, jpeg, gif, bmp, tiff, webp
- **Threading**: Enabled for concurrent requests
- **Debug mode**: Enabled in development

## Performance

### Concurrency Control
- Default: 3 concurrent requests
- Configurable via `max_concurrent` parameter
- Prevents API rate limiting

### Processing Time
- Single image: ~2-5 seconds
- Multiple images: Concurrent processing
- Example: 10 images with 3 concurrent = ~15-25 seconds

### Flask Threading
- Multi-threaded server for handling multiple requests
- Each request runs in its own thread
- Good for handling multiple simultaneous users

## Error Handling

The service provides comprehensive error handling:

- **400 Bad Request**: Invalid file type, missing files, invalid parameters
- **413 Payload Too Large**: File too large (over 16MB)
- **404 Not Found**: Endpoint not found
- **500 Internal Server Error**: Processing failures, API errors
- **Graceful Degradation**: Continues processing even if some images fail

## File Handling

### Upload Limits
- **Maximum file size**: 16MB per file
- **Allowed formats**: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP
- **Temporary storage**: Files are saved temporarily and cleaned up automatically

### Download Features
- **JSON results**: Download as JSON file with timestamp
- **SQLite database**: Download complete database file
- **Automatic naming**: Files include timestamp for uniqueness

## Development

### Running in Development Mode
```bash
python flask_service.py
```

### Running with Flask CLI
```bash
export FLASK_APP=flask_service.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000
```

### Production Deployment
```bash
# Using gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 flask_service:app

# Using waitress (Windows)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 flask_service:app
```

## Security Considerations

- **File Upload Validation**: File type and size validation
- **Temporary Files**: Automatic cleanup of uploaded files
- **Input Validation**: All inputs are validated
- **Error Messages**: Generic error messages to avoid information leakage
- **File Size Limits**: Prevents large file uploads

## Monitoring

### Health Check
Monitor service health with:
```bash
curl http://localhost:5000/health
```

### Logs
Flask provides detailed logging for:
- Request processing
- Error handling
- File uploads
- Processing times

### Error Tracking
- All errors are logged with stack traces
- HTTP status codes for different error types
- Detailed error messages for debugging

## Integration

This Flask service can be easily integrated with:
- **Web Applications**: Frontend frameworks (React, Vue, Angular)
- **Mobile Apps**: iOS/Android applications
- **Other Services**: Microservices architecture
- **Workflow Tools**: Apache Airflow, Prefect
- **Monitoring**: Prometheus, Grafana

## Comparison with FastAPI

| Feature | Flask | FastAPI |
|---------|-------|---------|
| **Async Support** | Limited (uses asyncio.run) | Native async/await |
| **Performance** | Good | Excellent |
| **Documentation** | Manual | Automatic (Swagger) |
| **Type Validation** | Manual | Automatic (Pydantic) |
| **Learning Curve** | Simple | Moderate |
| **Dependencies** | Minimal | More dependencies |
| **Production Ready** | Yes | Yes |

## Troubleshooting

### Common Issues

1. **"Image processor not initialized"**
   - Check if `MISTRAL_API_KEY` is set
   - Verify `config.json` exists and is valid

2. **"File too large"**
   - Reduce file size or increase `MAX_CONTENT_LENGTH`

3. **"Invalid file type"**
   - Check file extension is in allowed list
   - Ensure file is actually an image

4. **"Processing failed"**
   - Check Mistral API key is valid
   - Verify network connectivity
   - Check API rate limits

### Debug Mode
Enable debug mode for detailed error messages:
```python
app.run(debug=True)
``` 