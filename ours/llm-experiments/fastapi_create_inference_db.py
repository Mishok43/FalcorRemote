#!/usr/bin/env python3
"""
FastAPI service for image processing with Mistral Vision API.
Provides HTTP endpoints for processing images and saving results to database/JSON.
"""

import os
import json
import asyncio
import base64
from typing import List, Dict, Optional
from pathlib import Path
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our existing classes
from create_inference_db import AsyncImageProcessor, InferenceDatabase, JSONResultsManager

# Initialize FastAPI app
app = FastAPI(
    title="Image Processing API",
    description="API for processing images with Mistral Vision and extracting objects",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
processor = None
output_manager = None

# Pydantic models for request/response
class ProcessImageRequest(BaseModel):
    scene_path: str = "unknown"
    output_format: str = "json"  # "json" or "sql"
    db_path: str = "inference_results.db"
    json_path: str = "inference_results.json"

class ProcessImageResponse(BaseModel):
    success: bool
    scene_path: str
    image_path: str
    scene_description: str
    objects_present: List[str]
    processing_time: float
    message: str = ""

class ProcessBatchRequest(BaseModel):
    scene_path: str = "unknown"
    output_format: str = "json"
    db_path: str = "inference_results.db"
    json_path: str = "inference_results.json"
    max_concurrent: int = 3

class ProcessBatchResponse(BaseModel):
    success: bool
    total_images: int
    processed_images: int
    failed_images: int
    total_time: float
    average_time_per_image: float
    results: List[ProcessImageResponse]
    message: str = ""

class HealthResponse(BaseModel):
    status: str
    api_key_configured: bool
    config_loaded: bool

# Initialize the processor on startup
@app.on_event("startup")
async def startup_event():
    global processor
    try:
        processor = AsyncImageProcessor(config_path="config.json")
        print("✅ Image processor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize image processor: {e}")
        processor = None

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the service."""
    api_key_configured = bool(os.getenv("MISTRAL_API_KEY"))
    config_loaded = processor is not None
    
    status = "healthy" if api_key_configured and config_loaded else "unhealthy"
    
    return HealthResponse(
        status=status,
        api_key_configured=api_key_configured,
        config_loaded=config_loaded
    )

# Process single image endpoint
@app.post("/process-image", response_model=ProcessImageResponse)
async def process_single_image(
    file: UploadFile = File(...),
    scene_path: str = "unknown",
    output_format: str = "json",
    db_path: str = "inference_results.db",
    json_path: str = "inference_results.json"
):
    """
    Process a single image and extract objects.
    
    Args:
        file: Image file to process
        scene_path: Path to the scene/object file
        output_format: Output format ("json" or "sql")
        db_path: SQLite database path
        json_path: JSON results file path
    
    Returns:
        Processing results with scene description and objects
    """
    if not processor:
        raise HTTPException(status_code=500, detail="Image processor not initialized")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        # Create temporary file
        suffix = Path(file.filename).suffix if file.filename else '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Initialize output manager
        if output_format == 'sql':
            output_manager = InferenceDatabase(db_path)
        else:
            output_manager = JSONResultsManager(json_path)
        
        # Process the image
        start_time = asyncio.get_event_loop().time()
        
        # Create aiohttp session for processing
        import aiohttp
        async with aiohttp.ClientSession() as session:
            result = await processor.process_single_image_async(
                session, temp_path, scene_path
            )
        
        end_time = asyncio.get_event_loop().time()
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
        
        return ProcessImageResponse(
            success=True,
            scene_path=result['scene_path'],
            image_path=file.filename or "uploaded_image",
            scene_description=result['scene_description'],
            objects_present=result['objects_present'],
            processing_time=processing_time,
            message="Image processed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_path):
            os.unlink(temp_path)

# Process multiple images endpoint
@app.post("/process-batch", response_model=ProcessBatchResponse)
async def process_multiple_images(
    files: List[UploadFile] = File(...),
    scene_path: str = "unknown",
    output_format: str = "json",
    db_path: str = "inference_results.db",
    json_path: str = "inference_results.json",
    max_concurrent: int = 3
):
    """
    Process multiple images concurrently.
    
    Args:
        files: List of image files to process
        scene_path: Path to the scene/object file
        output_format: Output format ("json" or "sql")
        db_path: SQLite database path
        json_path: JSON results file path
        max_concurrent: Maximum concurrent requests
    
    Returns:
        Batch processing results
    """
    if not processor:
        raise HTTPException(status_code=500, detail="Image processor not initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} must be an image"
            )
    
    # Save uploaded files temporarily
    temp_files = []
    try:
        # Save all files temporarily
        for file in files:
            suffix = Path(file.filename).suffix if file.filename else '.jpg'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_files.append((temp_file.name, file.filename or f"image_{len(temp_files)}"))
        
        # Initialize output manager
        if output_format == 'sql':
            output_manager = InferenceDatabase(db_path)
        else:
            output_manager = JSONResultsManager(json_path)
        
        # Process images
        start_time = asyncio.get_event_loop().time()
        
        # Create aiohttp session for processing
        import aiohttp
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
        
        end_time = asyncio.get_event_loop().time()
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
            response_results.append(ProcessImageResponse(
                success=not result['scene_description'].startswith('Error'),
                scene_path=result['scene_path'],
                image_path=result['image_path'],
                scene_description=result['scene_description'],
                objects_present=result['objects_present'],
                processing_time=total_time / len(results),
                message="Processed successfully" if not result['scene_description'].startswith('Error') else "Processing failed"
            ))
        
        return ProcessBatchResponse(
            success=True,
            total_images=len(files),
            processed_images=len(successful_results),
            failed_images=len(failed_results),
            total_time=total_time,
            average_time_per_image=total_time / len(files),
            results=response_results,
            message=f"Processed {len(successful_results)} images successfully, {len(failed_results)} failed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for temp_path, _ in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# Get results endpoint
@app.get("/results")
async def get_results(
    output_format: str = "json",
    db_path: str = "inference_results.db",
    json_path: str = "inference_results.json"
):
    """
    Get all processing results.
    
    Args:
        output_format: Output format ("json" or "sql")
        db_path: SQLite database path
        json_path: JSON results file path
    
    Returns:
        All processing results
    """
    try:
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
            
            return JSONResponse({
                'format': 'sql',
                'total_results': len(formatted_results),
                'results': formatted_results
            })
        
        else:
            output_manager = JSONResultsManager(json_path)
            return JSONResponse({
                'format': 'json',
                'total_results': len(output_manager.results),
                'results': output_manager.results
            })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Image Processing API with Mistral Vision",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "process_single": "/process-image",
            "process_batch": "/process-batch",
            "get_results": "/results"
        },
        "docs": "/docs"
    }

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "fastapi_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 