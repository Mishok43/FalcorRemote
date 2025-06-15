#!/usr/bin/env python3
"""
Minimal async scene description using AsyncMistralVisionAPI from async_boolean_check.py
Optimized for maximum speed with connection pooling and aggressive concurrency.
"""

import asyncio
from typing import Optional, List
from dataclasses import dataclass
from async_check_boolean import AsyncMistralVisionAPI
import os
import argparse
from pathlib import Path
import time
import aiohttp
from concurrent.futures import ThreadPoolExecutor

@dataclass
class SceneDescriptionResult:
    """Result of a scene description generation."""
    image_path: str
    description: str
    processing_time: float
    sample_id: Optional[int] = None

class AsyncSceneDescriptionAPI:
    """Optimized wrapper around AsyncMistralVisionAPI for maximum speed."""
    
    def __init__(self, **kwargs):
        """Initialize using AsyncMistralVisionAPI with speed optimizations."""
        # Override config for maximum speed
        kwargs.setdefault('config_path', 'config.json')
        self.mistral_api = AsyncMistralVisionAPI(**kwargs)
        
        # Ultra-optimized config for maximum speed
        self.mistral_api.config['max_tokens'] = 50   # Minimal tokens for speed
        self.mistral_api.config['temperature'] = 0.0  # Deterministic for speed
        
        # Create connection pool for HTTP requests
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=20)  # More workers for I/O

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=100,  # High connection limit
                limit_per_host=50,  # High per-host limit
                ttl_dns_cache=300,  # DNS caching
                use_dns_cache=True,
            ),
            timeout=aiohttp.ClientTimeout(total=30)  # Reasonable timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=True)

    async def generate_scene_description_async(self, image_path: str, sample_id: int = None) -> SceneDescriptionResult:
        """Generate scene description for an image with maximum speed."""
        # Ultra-minimal prompt for speed
        prompt = "List all objects you see in the image and their locations."
        
        # Use the existing async inference method
        result = await self.mistral_api.run_vision_inference(image_path, prompt, sample_id)
        
        if "choices" in result and len(result["choices"]) > 0:
            description = result["choices"][0]["message"]["content"]
        else:
            description = "No description generated"
            
        return SceneDescriptionResult(
            image_path=image_path,
            description=description,
            processing_time=0.0,
            sample_id=sample_id
        )

    async def generate_scene_description_from_tensor_async(self, tensor, sample_id: int = None) -> SceneDescriptionResult:
        """Generate scene description from tensor directly with maximum speed."""
        # Ultra-minimal prompt for speed
        prompt = "Describe scene briefly."
        
        # Use the existing tensor inference method
        result = await self.mistral_api.run_vision_inference_from_tensor(tensor, prompt, sample_id)
        
        if "choices" in result and len(result["choices"]) > 0:
            description = result["choices"][0]["message"]["content"]
        else:
            description = "No description generated"
            
        return SceneDescriptionResult(
            image_path="<tensor>",
            description=description,
            processing_time=0.0,
            sample_id=sample_id
        )

    async def batch_generate_scene_descriptions_from_tensors(self, tensors: List, max_concurrent: int = 20) -> List[SceneDescriptionResult]:
        """Generate scene descriptions for multiple tensors with maximum concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_tensor(tensor, index: int):
            async with semaphore:
                return await self.generate_scene_description_from_tensor_async(tensor, sample_id=index)
        
        tasks = [process_single_tensor(tensor, i) for i, tensor in enumerate(tensors)]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def process_directory_async(self, directory_path: str, max_concurrent: int = 20, output_file: Optional[str] = None) -> List[SceneDescriptionResult]:
        """Process all images in a directory with maximum speed."""
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Find all image files in directory
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory {directory_path} does not exist")
        
        image_files = []
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
        
        if not image_files:
            print(f"No image files found in {directory_path}")
            return []
        
        print(f"Found {len(image_files)} images in {directory_path}")
        print(f"Processing with max {max_concurrent} concurrent requests for maximum speed...")
        
        # Process images in parallel with maximum concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_image(image_path: str, index: int):
            async with semaphore:
                try:
                    result = await self.generate_scene_description_async(image_path, sample_id=index)
                    print(f"✓ {index+1}/{len(image_files)}: {Path(image_path).name}")
                    return result
                except Exception as e:
                    print(f"✗ {Path(image_path).name}: {e}")
                    return SceneDescriptionResult(
                        image_path=image_path,
                        description=f"Error: {str(e)}",
                        processing_time=0.0,
                        sample_id=index
                    )
        
        tasks = [process_single_image(path, i) for i, path in enumerate(image_files)]
        results = await asyncio.gather(*tasks)
        
        # Save results to file if specified
        if output_file:
            self._save_results_to_file(results, output_file)
        
        return results

    def _save_results_to_file(self, results: List[SceneDescriptionResult], output_file: str):
        """Save results to a text file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Scene Descriptions Results\n")
                f.write("=" * 50 + "\n\n")
                
                for result in results:
                    f.write(f"Image: {Path(result.image_path).name}\n")
                    f.write(f"Sample ID: {result.sample_id}\n")
                    f.write(f"Description:\n{result.description}\n")
                    f.write("-" * 30 + "\n\n")
            
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results to file: {e}")


# Simple test
async def test_scene_description(image_path: str):
    """Test the scene description functionality."""
    async with AsyncSceneDescriptionAPI() as api:
        result = await api.generate_scene_description_async(image_path)
        print(f"Scene description: {result.description}")

async def test_directory_processing(directory_path: str, output_file: Optional[str] = None):
    """Test directory processing functionality with maximum speed."""
    async with AsyncSceneDescriptionAPI() as api:
        print(f"Processing directory: {directory_path}")
        print("Using optimized settings for maximum speed...")
        
        # analyze time taken to process the directory
        start_time = time.time()
        results = await api.process_directory_async(directory_path, max_concurrent=20, output_file=output_file)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / len(results) if results else 0
        
        print(f"\n⚡ Performance Results:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per image: {avg_time:.2f} seconds")
        print(f"Images per second: {len(results) / total_time:.2f}")
        
        print(f"\nProcessed {len(results)} images:")
        for result in results:
            print(f"\n{Path(result.image_path).name}:")
            print(f"  {result.description[:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Mistral Vision API Scene Understanding Demo")
    parser.add_argument("--image_path", "-i", type=str, help="Path to the image file")
    parser.add_argument("--directory", "-d", type=str, help="Path to directory containing images")
    parser.add_argument("--output", "-o", type=str, help="Output file to save results (for directory processing)")
    parser.add_argument("--max_concurrent", "-m", type=int, default=20, help="Maximum concurrent requests (default: 20)")
    args = parser.parse_args()

    if args.directory:
        # Process directory
        asyncio.run(test_directory_processing(args.directory, args.output))
    elif args.image_path:
        # Process single image
        asyncio.run(test_scene_description(args.image_path))
    else:
        print("Please provide either --image_path or --directory argument")
        parser.print_help()
