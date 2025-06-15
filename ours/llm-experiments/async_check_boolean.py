#!/usr/bin/env python3
"""
Async boolean checks for objects on images using Mistral Vision API.
This module provides async functionality to check for object presence in images
and generates presence prompts based on object names using the existing MistralVisionAPI.
"""

import os
import asyncio
import time
from typing import Optional, Union, List, Dict, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import argparse
import sys
# Import the existing MistralVisionAPI
from mistral_vision_inference import MistralVisionAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ObjectCheckResult:
    """Result of an object presence check."""
    object_name: str
    image_path: str
    is_present: bool
    confidence: float
    response_text: str
    processing_time: float
    prompt_used: str

@dataclass
class BatchCheckResult:
    """Result of a batch object presence check."""
    object_name: str
    results: List[ObjectCheckResult]
    total_images: int
    present_count: int
    absent_count: int
    average_confidence: float
    total_processing_time: float

class AsyncMistralVisionAPI:
    """Async wrapper for MistralVisionAPI with object presence checking capabilities."""
    
    def __init__(self, api_key: Optional[str] = None, config_path: str = "config.json", patterns_file: Optional[str] = None):
        """
        Initialize the async Mistral Vision API client.
        
        Args:
            api_key: Mistral API key. If None, will try to get from MISTRAL_API_KEY env var.
            config_path: Path to the configuration JSON file
            patterns_file: Path to the JSON file containing object patterns
        """
        # Initialize the existing MistralVisionAPI
        self.vision_api = MistralVisionAPI(api_key, config_path, patterns_file)
        
        # Get the object name from the existing API
        self.object_name = self.vision_api.get_object_name()
        
        # Get the patterns from the existing API
        self.patterns = self.vision_api.patterns
    
    def generate_presence_prompt(self, object_name: str, pattern_index: int = 0) -> str:
        """
        Generate a presence prompt for a specific object using existing patterns.
        
        Args:
            object_name: Name of the object to check for
            pattern_index: Index of the pattern to use
            
        Returns:
            Formatted presence prompt
        """
        if pattern_index >= len(self.patterns):
            pattern_index = 0
        
        return self.patterns[pattern_index].format(object=object_name)
    
    def generate_presence_prompts(self, object_name: str, num_prompts: int = 5) -> List[str]:
        """
        Generate multiple presence prompts for an object.
        
        Args:
            object_name: Name of the object to check for
            num_prompts: Number of prompts to generate
            
        Returns:
            List of presence prompts
        """
        prompts = []
        for i in range(min(num_prompts, len(self.patterns))):
            prompts.append(self.generate_presence_prompt(object_name, i))
        return prompts
    
    def _parse_boolean_response_with_confidence(self, response: str) -> Tuple[bool, float]:
        """
        Parse the API response to extract boolean result and confidence.
        Uses the existing _parse_boolean_response method and adds confidence calculation.
        
        Args:
            response: Raw response text from the API
            
        Returns:
            Tuple of (is_present, confidence)
        """
        # Use the existing boolean parsing method
        print(response)
        is_present = self.vision_api._parse_boolean_response(response)
        print(is_present)
        
        response_lower = response.lower().strip()
        
        # Calculate confidence based on response clarity
        positive_indicators = [
            "yes", "true", "present", "visible", "can see", "do see", 
            "there is", "contains", "shows", "detectable", "apparent",
            "found", "identified", "spotted", "there are", "i can see",
            "i see", "visible", "clear", "obvious", "definitely"
        ]
        
        negative_indicators = [
            "no", "false", "not present", "not visible", "cannot see", 
            "don't see", "not there", "doesn't contain", "does not contain",
            "not shown", "not detectable", "not apparent", "not found",
            "not identified", "not spotted", "i cannot see", "i don't see",
            "not visible", "not clear", "not obvious", "definitely not"
        ]
        
        # Count indicators
        positive_count = sum(1 for indicator in positive_indicators if indicator in response_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in response_lower)
        
        # Calculate confidence
        if is_present:
            confidence = min(0.95, 0.5 + (positive_count * 0.1))
        else:
            confidence = min(0.95, 0.5 + (negative_count * 0.1))
        
        return is_present, confidence
    
    async def check_object_presence_async(self, 
                                        image_path: str, 
                                        object_name: str,
                                        prompt: Optional[str] = None) -> ObjectCheckResult:
        """
        Check for object presence in an image asynchronously using the existing API.
        
        Args:
            image_path: Path to the image file
            object_name: Name of the object to check for
            prompt: Custom prompt (if None, will generate one)
            
        Returns:
            ObjectCheckResult with the check results
        """
        start_time = time.time()
        
        # Generate prompt if not provided
        if prompt is None:
            
            prompt = self.generate_presence_prompt(object_name)
        
        try:
            # Use the existing check_object_presence_timed method
            # We'll run this in a thread pool to make it async
            loop = asyncio.get_event_loop()
            
            def run_check():
                return self.vision_api.check_object_presence_timed(image_path, prompt)
            
            # Run the synchronous method in a thread pool
            result, processing_time = await loop.run_in_executor(None, run_check)
            
            # Get the full response for confidence calculation
            full_response = await loop.run_in_executor(
                None, 
                lambda: self.vision_api.run_vision_inference(image_path, prompt)
            )
            
            # Extract response text
            response_text = full_response["choices"][0]["message"]["content"]
            
            # Parse boolean result and confidence
            is_present, confidence = self._parse_boolean_response_with_confidence(response_text)
            
            return ObjectCheckResult(
                object_name=object_name,
                image_path=image_path,
                is_present=is_present,
                confidence=confidence,
                response_text=response_text,
                processing_time=processing_time,
                prompt_used=prompt
            )
            
        except Exception as e:
            logger.error(f"Error checking object presence for {object_name} in {image_path}: {e}")
            processing_time = time.time() - start_time
            
            return ObjectCheckResult(
                object_name=object_name,
                image_path=image_path,
                is_present=False,
                confidence=0.0,
                response_text=f"Error: {str(e)}",
                processing_time=processing_time,
                prompt_used=prompt
            )
    
    async def check_multiple_objects_async(self, 
                                         image_path: str, 
                                         object_names: List[str],
                                         prompts: Optional[Dict[str, str]] = None) -> List[ObjectCheckResult]:
        """
        Check for multiple objects in a single image asynchronously.
        
        Args:
            image_path: Path to the image file
            object_names: List of object names to check for
            prompts: Dictionary mapping object names to custom prompts
            
        Returns:
            List of ObjectCheckResult for each object
        """
        tasks = []
        for object_name in object_names:
            prompt = prompts.get(object_name) if prompts else None
            task = self.check_object_presence_async(image_path, object_name, prompt)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def batch_check_object_presence(self, 
                                        image_paths: List[str], 
                                        object_name: str,
                                        prompt: Optional[str] = None,
                                        max_concurrent: int = 5) -> BatchCheckResult:
        """
        Check for object presence across multiple images asynchronously.
        
        Args:
            image_paths: List of image file paths
            object_name: Name of the object to check for
            prompt: Custom prompt (if None, will generate one)
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            BatchCheckResult with aggregated results
        """
        start_time = time.time()
        
        # Generate prompt if not provided
        if prompt is None:
            prompt = self.generate_presence_prompt(object_name)
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def check_with_semaphore(image_path):
            async with semaphore:
                return await self.check_object_presence_async(image_path, object_name, prompt)
        
        # Create tasks
        tasks = [check_with_semaphore(image_path) for image_path in image_paths]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to ObjectCheckResult
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
            else:
                valid_results.append(result)
        
        # Calculate statistics
        present_count = sum(1 for r in valid_results if r.is_present)
        absent_count = len(valid_results) - present_count
        avg_confidence = sum(r.confidence for r in valid_results) / len(valid_results) if valid_results else 0
        total_processing_time = time.time() - start_time
        
        return BatchCheckResult(
            object_name=object_name,
            results=valid_results,
            total_images=len(image_paths),
            present_count=present_count,
            absent_count=absent_count,
            average_confidence=avg_confidence,
            total_processing_time=total_processing_time
        )
    
    async def batch_check_multiple_objects(self, 
                                         image_paths: List[str], 
                                         object_names: List[str],
                                         prompts: Optional[Dict[str, str]] = None,
                                         max_concurrent: int = 5) -> Dict[str, BatchCheckResult]:
        """
        Check for multiple objects across multiple images asynchronously.
        
        Args:
            image_paths: List of image file paths
            object_names: List of object names to check for
            prompts: Dictionary mapping object names to custom prompts
            max_concurrent: Maximum number of concurrent requests per object
            
        Returns:
            Dictionary mapping object names to BatchCheckResult
        """
        tasks = {}
        for object_name in object_names:
            prompt = prompts.get(object_name) if prompts else None
            task = self.batch_check_object_presence(
                image_paths, object_name, prompt, max_concurrent
            )
            tasks[object_name] = task
        
        # Execute all batch checks concurrently
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Map results back to object names
        batch_results = {}
        for object_name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Batch check failed for {object_name}: {result}")
            else:
                batch_results[object_name] = result
        
        return batch_results
    
    def get_object_name(self) -> str:
        """Get the object name from the existing API."""
        return self.object_name
    
    def get_patterns(self) -> List[str]:
        """Get the patterns from the existing API."""
        return self.patterns

# Example usage and demonstration functions
async def demonstrate_single_object_check(object_name: str, image_path: str, api: AsyncMistralVisionAPI):
    """Demonstrate checking for a single object in an image."""
    
    if not os.path.exists(image_path):
        logger.warning(f"Image {image_path} not found. Skipping demonstration.")
        return
    
    # Check for the specified object in the image
    prompt = f"Is there a {object_name} in the image? Answer yes or no."
    result = await api.check_object_presence_async(image_path, object_name, prompt)
    
    print(f"Object: {result.object_name}")
    print(f"Image: {result.image_path}")
    print(f"Present: {result.is_present}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Response: {result.response_text}")
    print(f"Processing time: {result.processing_time:.2f}s")

async def demonstrate_batch_check(object_name: str, images_dir: str, api: AsyncMistralVisionAPI):
    """Demonstrate batch checking for an object across multiple images."""
    
    # Get all image files from the directory
    if not os.path.exists(images_dir):
        logger.warning(f"Images directory {images_dir} not found. Skipping batch demonstration.")
        return
    
    # Get all image files from the directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    image_paths = []
    
    for filename in os.listdir(images_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(images_dir, filename))
    
    if not image_paths:
        logger.warning(f"No image files found in {images_dir}. Skipping batch demonstration.")
        return
    
    # Batch check for the specified object
    prompt = f"Is there a {object_name} in the image? Answer only yes or no."
    batch_result = await api.batch_check_object_presence(image_paths, object_name, prompt)
    
    print(f"Batch Check Results for '{batch_result.object_name}':")
    print(f"Total images: {batch_result.total_images}")
    print(f"Present: {batch_result.present_count}")
    print(f"Absent: {batch_result.absent_count}")
    print(f"Average confidence: {batch_result.average_confidence:.2f}")
    print(f"Total processing time: {batch_result.total_processing_time:.2f}s")
    
    # Show individual results
    for result in batch_result.results:
        print(f"  {result.image_path}: {'Present' if result.is_present else 'Absent'} "
              f"(confidence: {result.confidence:.2f})")

async def demonstrate_multiple_objects(image_path: str, object_names: List[str], api: AsyncMistralVisionAPI):
    """Demonstrate checking for multiple objects in a single image."""
    
    if not os.path.exists(image_path):
        logger.warning(f"Image {image_path} not found. Skipping demonstration.")
        return
    
    # Check for multiple objects
    results = await api.check_multiple_objects_async(image_path, object_names)
    
    print(f"Multiple Object Check Results for {image_path}:")
    for result in results:
        print(f"  {result.object_name}: {'Present' if result.is_present else 'Absent'} "
              f"(confidence: {result.confidence:.2f})")

async def main(args):
    """Main function to run demonstrations."""
    print("Async Mistral Vision API Object Detection Demo")
    print("=" * 50)

    # Initialize the API
    if args.patterns_file:
        api = AsyncMistralVisionAPI(patterns_file=args.patterns_file, config_path=args.config_file)
    else:
        api = AsyncMistralVisionAPI(config_path=args.config_file)
    
    try:
        # Run demonstrations based on provided arguments
        if args.image_path:
            print(f"\n1. Single Object Check for '{args.object_name}' in '{args.image_path}':")
            await demonstrate_single_object_check(args.object_name, args.image_path, api)
        
        if args.images_dir:
            print(f"\n2. Batch Object Check for '{args.object_name}' in directory '{args.images_dir}':")
            await demonstrate_batch_check(args.object_name, args.images_dir, api)
        
        if args.image_path and args.object_names:
            print(f"\n3. Multiple Objects Check in '{args.image_path}':")
            await demonstrate_multiple_objects(args.image_path, args.object_names, api)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mistral Vision API Object Detection Demo")
    parser.add_argument("--object_name", "-o", type=str, default="bug", help="Object name to check for")
    parser.add_argument("--image_path", "-img", type=str, help="Path to a single image file")
    parser.add_argument("--images_dir", "-i", type=str, default="images", help="Path to the directory containing images")
    parser.add_argument("--object_names", "-objs", type=str, nargs='+', help="List of object names to check for multiple objects")
    parser.add_argument("--config_file", "-c", type=str, default="config.json", help="Path to the configuration JSON file")
    parser.add_argument("--describe", "-d", type=bool, default=False, help="Demonstrate all, single, batch, or multiple objects")
    parser.add_argument("--patterns_file", "-p", type=str, default=None, help="Path to the JSON file containing object patterns")
    args = parser.parse_args()

    asyncio.run(main(args))

