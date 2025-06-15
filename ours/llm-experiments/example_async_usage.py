#!/usr/bin/env python3
"""
Example usage of the async boolean check system for objects on images.
This script demonstrates how to use the AsyncMistralVisionAPI class to check
for object presence in images using the existing MistralVisionAPI functions.
"""

import asyncio
import os
from async_check_boolean import AsyncMistralVisionAPI, ObjectCheckResult, BatchCheckResult

async def example_single_object_check():
    """Example: Check for a single object in one image."""
    print("=== Single Object Check Example ===")
    
    # Initialize the API client (uses existing MistralVisionAPI internally)
    api = AsyncMistralVisionAPI()
    
    # Example: Check for a cat in an image
    image_path = "images/example.jpg"  # Replace with your image path
    object_name = "cat"
    
    if os.path.exists(image_path):
        result = await api.check_object_presence_async(image_path, object_name)
        
        print(f"Checking for '{object_name}' in '{image_path}'")
        print(f"Present: {result.is_present}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Response: {result.response_text}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Prompt used: {result.prompt_used}")
    else:
        print(f"Image {image_path} not found. Please provide a valid image path.")

async def example_multiple_objects_single_image():
    """Example: Check for multiple objects in one image."""
    print("\n=== Multiple Objects in Single Image Example ===")
    
    api = AsyncMistralVisionAPI()
    
    image_path = "images/example.jpg"  # Replace with your image path
    object_names = ["cat", "dog", "bird", "car", "tree"]
    
    if os.path.exists(image_path):
        results = await api.check_multiple_objects_async(image_path, object_names)
        
        print(f"Checking for multiple objects in '{image_path}':")
        for result in results:
            status = "Present" if result.is_present else "Absent"
            print(f"  {result.object_name}: {status} (confidence: {result.confidence:.2f})")
    else:
        print(f"Image {image_path} not found. Please provide a valid image path.")

async def example_batch_check():
    """Example: Check for one object across multiple images."""
    print("\n=== Batch Check Example ===")
    
    api = AsyncMistralVisionAPI()
    
    # List of image paths (replace with your actual image paths)
    image_paths = [
        "images/image1.jpg",
        "images/image2.jpg", 
        "images/image3.jpg",
        "images/image4.jpg",
        "images/image5.jpg"
    ]
    
    # Filter to only existing images
    existing_images = [path for path in image_paths if os.path.exists(path)]
    
    if existing_images:
        object_name = "bug"
        batch_result = await api.batch_check_object_presence(existing_images, object_name)
        
        print(f"Batch check results for '{object_name}':")
        print(f"Total images processed: {batch_result.total_images}")
        print(f"Present in: {batch_result.present_count} images")
        print(f"Absent in: {batch_result.absent_count} images")
        print(f"Average confidence: {batch_result.average_confidence:.2f}")
        print(f"Total processing time: {batch_result.total_processing_time:.2f}s")
        
        print("\nIndividual results:")
        for result in batch_result.results:
            status = "Present" if result.is_present else "Absent"
            print(f"  {result.image_path}: {status} (confidence: {result.confidence:.2f})")
    else:
        print("No images found. Please provide valid image paths.")

async def example_custom_prompts():
    """Example: Using custom prompts for object detection."""
    print("\n=== Custom Prompts Example ===")
    
    api = AsyncMistralVisionAPI()
    
    image_path = "images/example.jpg"  # Replace with your image path
    object_name = "needle"
    
    # Custom prompt for finding a needle in a haystack scenario
    custom_prompt = "Is there a needle visible in this image? Look carefully for any small, thin, metallic objects."
    
    if os.path.exists(image_path):
        result = await api.check_object_presence_async(image_path, object_name, custom_prompt)
        
        print(f"Custom prompt check for '{object_name}':")
        print(f"Custom prompt: {custom_prompt}")
        print(f"Present: {result.is_present}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Response: {result.response_text}")
    else:
        print(f"Image {image_path} not found. Please provide a valid image path.")

async def example_multiple_objects_batch():
    """Example: Check for multiple objects across multiple images."""
    print("\n=== Multiple Objects Batch Check Example ===")
    
    api = AsyncMistralVisionAPI()
    
    image_paths = [
        "images/image1.jpg",
        "images/image2.jpg",
        "images/image3.jpg"
    ]
    
    object_names = ["cat", "dog", "bird"]
    
    # Custom prompts for specific objects
    custom_prompts = {
        "cat": "Is there a cat or feline animal visible in this image?",
        "dog": "Can you see any dog or canine animal in this image?",
        "bird": "Are there any birds, flying animals, or avian creatures in this image?"
    }
    
    existing_images = [path for path in image_paths if os.path.exists(path)]
    
    if existing_images:
        batch_results = await api.batch_check_multiple_objects(
            existing_images, object_names, custom_prompts, max_concurrent=3
        )
        
        print("Multiple objects batch check results:")
        for object_name, batch_result in batch_results.items():
            print(f"\n{object_name.upper()}:")
            print(f"  Present in: {batch_result.present_count}/{batch_result.total_images} images")
            print(f"  Average confidence: {batch_result.average_confidence:.2f}")
            print(f"  Processing time: {batch_result.total_processing_time:.2f}s")
    else:
        print("No images found. Please provide valid image paths.")

async def example_prompt_generation():
    """Example: Generate different presence prompts for an object."""
    print("\n=== Prompt Generation Example ===")
    
    api = AsyncMistralVisionAPI()
    
    object_name = "car"
    
    # Generate different prompts using existing patterns
    single_prompt = api.generate_presence_prompt(object_name, pattern_index=0)
    multiple_prompts = api.generate_presence_prompts(object_name, num_prompts=5)
    
    print(f"Single prompt for '{object_name}': {single_prompt}")
    print(f"\nMultiple prompts for '{object_name}':")
    for i, prompt in enumerate(multiple_prompts):
        print(f"  {i+1}. {prompt}")

async def example_with_patterns_file():
    """Example: Using a patterns file for specific object detection."""
    print("\n=== Patterns File Example ===")
    
    # Initialize with a specific patterns file
    patterns_file = "search_patterns/bug.json"
    
    if os.path.exists(patterns_file):
        api = AsyncMistralVisionAPI(patterns_file=patterns_file)
        
        print(f"Using patterns file: {patterns_file}")
        print(f"Object name: {api.get_object_name()}")
        print(f"Number of patterns: {len(api.get_patterns())}")
        
        # Show some generated prompts
        prompts = api.generate_presence_prompts(api.get_object_name(), 3)
        print("\nGenerated prompts:")
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}. {prompt}")
    else:
        print(f"Patterns file {patterns_file} not found. Using default patterns.")
        api = AsyncMistralVisionAPI()
        print(f"Default object name: {api.get_object_name()}")

async def main():
    """Run all examples."""
    print("Async Mistral Vision API - Object Detection Examples")
    print("(Using existing MistralVisionAPI functions)")
    print("=" * 60)
    
    try:
        # Run all examples
        await example_prompt_generation()
        await example_with_patterns_file()
        await example_single_object_check()
        await example_multiple_objects_single_image()
        await example_custom_prompts()
        await example_batch_check()
        await example_multiple_objects_batch()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have:")
        print("1. Set the MISTRAL_API_KEY environment variable")
        print("2. Provided valid image paths in the examples")
        print("3. The mistral_vision_inference.py file is in the same directory")

if __name__ == "__main__":
    asyncio.run(main()) 