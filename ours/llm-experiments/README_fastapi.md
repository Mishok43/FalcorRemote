# FastAPI Image Processing Service

A FastAPI web service that provides HTTP endpoints for processing images with Mistral Vision API and extracting objects. This service wraps the functionality of the `create_inference_db.py` script into a web API.

## Features

- **HTTP Endpoints**: RESTful API for image processing
- **File Upload**: Upload single or multiple images
- **Async Processing**: Concurrent image processing with configurable limits
- **Multiple Output Formats**: Save results to JSON or SQLite
- **Health Monitoring**: Service health check endpoint
- **Automatic Documentation**: Interactive API docs with Swagger UI
- **CORS Support**: Cross-origin resource sharing enabled

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_fastapi.txt
```

### 2. Set Environment Variables

```bash
export MISTRAL_API_KEY="your_mistral_api_key_here"
```

### 3. Run the Service

```bash
python fastapi_service.py
```

The service will start on `http://localhost:8000`

## API Endpoints

### 1. Health Check
- **URL**: `GET /health`
- **Description**: Check service health and configuration
- **Response**: Service status, API key configuration, processor status

### 2. Process Single Image
- **URL**: `POST /process-image`
- **Description**: Process a single image and extract objects
- **Parameters**:
  - `file`: Image file (multipart/form-data)
  - `scene_path`: Scene/object file path (optional, default: "unknown")
  - `output_format`: "json" or "sql" (optional, default: "json")
  - `db_path`: SQLite database path (optional, default: "inference_results.db")
  - `json_path`: JSON results file path (optional, default: "inference_results.json")

### 3. Process Multiple Images
- **URL**: `POST /process-batch`
- **Description**: Process multiple images concurrently
- **Parameters**:
  - `files`: List of image files (multipart/form-data)
  - `scene_path`: Scene/object file path (optional, default: "unknown")
  - `output_format`: "json" or "sql" (optional, default: "json")
  - `db_path`: SQLite database path (optional, default: "inference_results.db")
  - `json_path`: JSON results file path (optional, default: "inference_results.json")
  - `max_concurrent`: Maximum concurrent requests (optional, default: 3)

### 4. Get Results
- **URL**: `GET /results`
- **Description**: Retrieve all processing results
- **Parameters**:
  - `output_format`: "json" or "sql" (optional, default: "json")
  - `db_path`: SQLite database path (optional, default: "inference_results.db")
  - `json_path`: JSON results file path (optional, default: "inference_results.json")

### 5. API Documentation
- **URL**: `GET /docs`
- **Description**: Interactive API documentation (Swagger UI)

## Usage Examples

### Using curl

#### Process Single Image
```bash
curl -X POST "http://localhost:8000/process-image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg" \
  -F "scene_path=scene.obj" \
  -F "output_format=json"
```

#### Process Multiple Images
```bash
curl -X POST "http://localhost:8000/process-batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/image1.jpg" \
  -F "files=@/path/to/image2.jpg" \
  -F "scene_path=scene.obj" \
  -F "max_concurrent=3"
```

#### Get Results
```bash
curl -X GET "http://localhost:8000/results?output_format=json"
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
    response = requests.post('http://localhost:8000/process-image', 
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
    response = requests.post('http://localhost:8000/process-batch', 
                           files=files, data=data)
    result = response.json()
    print(result)
```

### Using JavaScript/Fetch

```javascript
// Process single image
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('scene_path', 'scene.obj');
formData.append('output_format', 'json');

fetch('http://localhost:8000/process-image', {
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

fetch('http://localhost:8000/process-batch', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
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
  "config_loaded": true
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

## Performance

### Concurrency Control
- Default: 3 concurrent requests
- Configurable via `max_concurrent` parameter
- Prevents API rate limiting

### Processing Time
- Single image: ~2-5 seconds
- Multiple images: Concurrent processing
- Example: 10 images with 3 concurrent = ~15-25 seconds

## Error Handling

The service provides comprehensive error handling:

- **400 Bad Request**: Invalid file type, missing files
- **500 Internal Server Error**: Processing failures, API errors
- **Graceful Degradation**: Continues processing even if some images fail

## Development

### Running in Development Mode
```bash
python fastapi_service.py
```

### Running with uvicorn directly
```bash
uvicorn fastapi_service:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment
```bash
uvicorn fastapi_service:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the service is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These provide interactive documentation where you can:
- See all available endpoints
- Test the API directly
- View request/response schemas
- Understand parameter requirements

## Security Considerations

- **File Upload Limits**: Consider adding file size limits
- **Authentication**: Add authentication for production use
- **Rate Limiting**: Implement rate limiting for API endpoints
- **Input Validation**: All inputs are validated automatically by FastAPI

## Monitoring

### Health Check
Monitor service health with:
```bash
curl http://localhost:8000/health
```

### Logs
The service provides detailed logging for:
- Request processing
- Error handling
- Performance metrics
- API responses

## Integration

This FastAPI service can be easily integrated with:
- **Web Applications**: Frontend frameworks (React, Vue, Angular)
- **Mobile Apps**: iOS/Android applications
- **Other Services**: Microservices architecture
- **Workflow Tools**: Apache Airflow, Prefect
- **Monitoring**: Prometheus, Grafana 