#!/usr/bin/env python3
"""
Flask service for image processing with Mistral Vision API.
Provides HTTP endpoints for processing images and saving results to database/JSON.
"""

import os
import json
import time
import base64
from typing import List, Dict, Optional
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import requests

# Import our existing classes
from create_inference_db import AsyncImageProcessor, InferenceDatabase, JSONResultsManager

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
processor = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_processor():
    """Initialize the image processor."""
    global processor
    try:
        processor = AsyncImageProcessor(config_path="config.json")
        print("✅ Image processor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize image processor: {e}")
        processor = None
        return False

# Initialize processor on startup
initialize_processor()

@app.route('/health', methods=['GET'])
def health_check():
    """Check the health of the service."""
    api_key_configured = bool(os.getenv("MISTRAL_API_KEY"))
    config_loaded = processor is not None
    
    status = "healthy" if api_key_configured and config_loaded else "unhealthy"
    
    return jsonify({
        "status": status,
        "api_key_configured": api_key_configured,
        "config_loaded": config_loaded,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/process-image', methods=['POST'])
def process_single_image():
    """
    Process a single image and extract objects.
    
    Expected form data:
    - file: Image file
    - scene_path: Scene/object file path (optional)
    - output_format: "json" or "sql" (optional)
    - db_path: SQLite database path (optional)
    - json_path: JSON results file path (optional)
    """
    if not processor:
        return jsonify({"error": "Image processor not initialized"}), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: " + ", ".join(ALLOWED_EXTENSIONS)}), 400
    
    # Get parameters
    scene_path = request.form.get('scene_path', 'unknown')
    output_format = request.form.get('output_format', 'json')
    db_path = request.form.get('db_path', 'inference_results.db')
    json_path = request.form.get('json_path', 'inference_results.json')
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        # Create temporary file
        suffix = Path(file.filename).suffix if file.filename else '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        # Initialize output manager
        if output_format == 'sql':
            output_manager = InferenceDatabase(db_path)
        else:
            output_manager = JSONResultsManager(json_path)
        
        # Process the image
        start_time = time.time()
        
        # Use asyncio to run the async processor
        import asyncio
        import aiohttp
        
        async def process_image():
            async with aiohttp.ClientSession() as session:
                return await processor.process_single_image_async(
                    session, temp_path, scene_path
                )
        
        # Run the async function
        result = asyncio.run(process_image())
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Save result
        if output_format == 'sql':
            output_manager.insert_inference_result(
                result['scene_path'],
                result['image_path'],
                result['scene_description'],
                result['objects_present']
            )
        else:
            output_manager.add_inference_result(
                result['scene_path'],
                result['image_path'],
                result['scene_description'],
                result['objects_present']
            )
            output_manager.save_results()
        
        return jsonify({
            "success": True,
            "scene_path": result['scene_path'],
            "image_path": file.filename,
            "scene_description": result['scene_description'],
            "objects_present": result['objects_present'],
            "processing_time": processing_time,
            "message": "Image processed successfully"
        })
        
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.route('/process-batch', methods=['POST'])
def process_multiple_images():
    """
    Process multiple images concurrently.
    
    Expected form data:
    - files: Multiple image files
    - scene_path: Scene/object file path (optional)
    - output_format: "json" or "sql" (optional)
    - db_path: SQLite database path (optional)
    - json_path: JSON results file path (optional)
    - max_concurrent: Maximum concurrent requests (optional)
    """
    if not processor:
        return jsonify({"error": "Image processor not initialized"}), 500
    
    # Check if files were uploaded
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No files selected"}), 400
    
    # Validate file types
    for file in files:
        if file.filename and not allowed_file(file.filename):
            return jsonify({"error": f"Invalid file type for {file.filename}"}), 400
    
    # Get parameters
    scene_path = request.form.get('scene_path', 'unknown')
    output_format = request.form.get('output_format', 'json')
    db_path = request.form.get('db_path', 'inference_results.db')
    json_path = request.form.get('json_path', 'inference_results.json')
    max_concurrent = int(request.form.get('max_concurrent', 3))
    
    # Save uploaded files temporarily
    temp_files = []
    try:
        # Save all files temporarily
        for file in files:
            if file.filename:
                suffix = Path(file.filename).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    file.save(temp_file.name)
                    temp_files.append((temp_file.name, file.filename))
        
        # Initialize output manager
        if output_format == 'sql':
            output_manager = InferenceDatabase(db_path)
        else:
            output_manager = JSONResultsManager(json_path)
        
        # Process images
        start_time = time.time()
        
        # Use asyncio to run the async processor
        import asyncio
        import aiohttp
        
        async def process_images():
            async with aiohttp.ClientSession() as session:
                # Process all images concurrently
                tasks = []
                for temp_path, filename in temp_files:
                    task = processor.process_single_image_async(session, temp_path, scene_path)
                    tasks.append((task, filename))
                
                # Execute all tasks
                results = []
                for task, filename in tasks:
                    try:
                        result = await task
                        result['image_path'] = filename  # Use original filename
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'scene_path': scene_path,
                            'image_path': filename,
                            'scene_description': f"Error: {str(e)}",
                            'objects_present': []
                        })
                
                return results
        
        # Run the async function
        results = asyncio.run(process_images())
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Save all results
        for result in results:
            if output_format == 'sql':
                output_manager.insert_inference_result(
                    result['scene_path'],
                    result['image_path'],
                    result['scene_description'],
                    result['objects_present']
                )
            else:
                output_manager.add_inference_result(
                    result['scene_path'],
                    result['image_path'],
                    result['scene_description'],
                    result['objects_present']
                )
        
        if output_format == 'json':
            output_manager.save_results()
        
        # Count successes and failures
        successful_results = [r for r in results if not r['scene_description'].startswith('Error')]
        failed_results = [r for r in results if r['scene_description'].startswith('Error')]
        
        # Convert to response format
        response_results = []
        for result in results:
            response_results.append({
                "success": not result['scene_description'].startswith('Error'),
                "scene_path": result['scene_path'],
                "image_path": result['image_path'],
                "scene_description": result['scene_description'],
                "objects_present": result['objects_present'],
                "processing_time": total_time / len(results),
                "message": "Processed successfully" if not result['scene_description'].startswith('Error') else "Processing failed"
            })
        
        return jsonify({
            "success": True,
            "total_images": len(files),
            "processed_images": len(successful_results),
            "failed_images": len(failed_results),
            "total_time": total_time,
            "average_time_per_image": total_time / len(files),
            "results": response_results,
            "message": f"Processed {len(successful_results)} images successfully, {len(failed_results)} failed"
        })
        
    except Exception as e:
        return jsonify({"error": f"Batch processing failed: {str(e)}"}), 500
    
    finally:
        # Clean up temporary files
        for temp_path, _ in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

@app.route('/results', methods=['GET'])
def get_results():
    """
    Get all processing results.
    
    Query parameters:
    - output_format: "json" or "sql" (optional)
    - db_path: SQLite database path (optional)
    - json_path: JSON results file path (optional)
    """
    try:
        output_format = request.args.get('output_format', 'json')
        db_path = request.args.get('db_path', 'inference_results.db')
        json_path = request.args.get('json_path', 'inference_results.json')
        
        if output_format == 'sql':
            output_manager = InferenceDatabase(db_path)
            results = output_manager.get_all_results()
            
            # Convert to JSON format
            formatted_results = []
            for scene_path, image_path, scene_desc, objects_json, timestamp in results:
                try:
                    objects = json.loads(objects_json) if objects_json else []
                except:
                    objects = []
                
                formatted_results.append({
                    'scene_path': scene_path,
                    'image_path': image_path,
                    'scene_description': scene_desc,
                    'objects_present': objects,
                    'inference_timestamp': timestamp
                })
            
            return jsonify({
                'format': 'sql',
                'total_results': len(formatted_results),
                'results': formatted_results
            })
        
        else:
            output_manager = JSONResultsManager(json_path)
            return jsonify({
                'format': 'json',
                'total_results': len(output_manager.results),
                'results': output_manager.results
            })
    
    except Exception as e:
        return jsonify({"error": f"Failed to get results: {str(e)}"}), 500

@app.route('/download-results', methods=['GET'])
def download_results():
    """
    Download results as a file.
    
    Query parameters:
    - output_format: "json" or "sql" (optional)
    - db_path: SQLite database path (optional)
    - json_path: JSON results file path (optional)
    """
    try:
        output_format = request.args.get('output_format', 'json')
        db_path = request.args.get('db_path', 'inference_results.db')
        json_path = request.args.get('json_path', 'inference_results.json')
        
        if output_format == 'sql':
            if not os.path.exists(db_path):
                return jsonify({"error": "Database file not found"}), 404
            return send_file(db_path, as_attachment=True, download_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        
        else:
            if not os.path.exists(json_path):
                return jsonify({"error": "JSON file not found"}), 404
            return send_file(json_path, as_attachment=True, download_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    except Exception as e:
        return jsonify({"error": f"Failed to download results: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        "message": "Image Processing API with Mistral Vision",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "process_single": "/process-image",
            "process_batch": "/process-batch",
            "get_results": "/results",
            "download_results": "/download-results"
        },
        "usage": {
            "single_image": "POST /process-image with 'file' in form data",
            "multiple_images": "POST /process-batch with 'files' in form data",
            "get_results": "GET /results?output_format=json",
            "health_check": "GET /health"
        }
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    ) 