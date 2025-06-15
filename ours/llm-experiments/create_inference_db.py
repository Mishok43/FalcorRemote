#!/usr/bin/env python3
"""
Script to process images, run boolean and scene description inferences,
and save results to an SQL database or JSON file.
"""

import os
import sqlite3
import json
import argparse
import time
import asyncio
import aiohttp
import base64
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys
from datetime import datetime

# Add the current directory to the path to import mistral_vision_inference
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mistral_vision_inference import MistralVisionAPI


class InferenceDatabase:
    def __init__(self, db_path: str = "inference_results.db"):
        """
        Initialize the SQLite database for storing inference results.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with the required table structure."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create the main table for inference results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scene_inferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scene_path TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    scene_description TEXT,
                    objects_present TEXT,
                    inference_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(scene_path, image_path)
                )
            ''')
            
            # Create an index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_scene_path 
                ON scene_inferences(scene_path)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_image_path 
                ON scene_inferences(image_path)
            ''')
            
            conn.commit()
    
    def insert_inference_result(self, scene_path: str, image_path: str, 
                              scene_description: str, objects_present: List[str]):
        """
        Insert inference results into the database.
        
        Args:
            scene_path: Path to the scene/object file
            image_path: Path to the image file
            scene_description: Description of the scene
            objects_present: List of objects present in the image
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert objects list to JSON string for storage
            objects_json = json.dumps(objects_present)
            
            cursor.execute('''
                INSERT OR REPLACE INTO scene_inferences 
                (scene_path, image_path, scene_description, objects_present)
                VALUES (?, ?, ?, ?)
            ''', (scene_path, image_path, scene_description, objects_json))
            
            conn.commit()
    
    def get_all_results(self) -> List[Tuple]:
        """Get all inference results from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT scene_path, image_path, scene_description, objects_present, inference_timestamp
                FROM scene_inferences
                ORDER BY scene_path, image_path
            ''')
            return cursor.fetchall()
    
    def get_results_by_scene(self, scene_path: str) -> List[Tuple]:
        """Get all results for a specific scene."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT scene_path, image_path, scene_description, objects_present, inference_timestamp
                FROM scene_inferences
                WHERE scene_path = ?
                ORDER BY image_path
            ''', (scene_path,))
            return cursor.fetchall()
    
    def print_results_table(self):
        """Print all results in a formatted table."""
        results = self.get_all_results()
        
        if not results:
            print("No results found in database.")
            return
        
        print("\n" + "="*120)
        print("INFERENCE RESULTS DATABASE")
        print("="*120)
        print(f"{'Scene Path':<30} | {'Image Path':<30} | {'Scene Description':<40} | {'Objects Present'}")
        print("-"*120)
        
        for scene_path, image_path, scene_desc, objects_json, timestamp in results:
            # Truncate long paths for display
            scene_display = scene_path[-29:] if len(scene_path) > 30 else scene_path
            image_display = image_path[-29:] if len(image_path) > 30 else image_path
            
            # Parse objects from JSON
            try:
                objects = json.loads(objects_json) if objects_json else []
                objects_display = ", ".join(objects[:3])  # Show first 3 objects
                if len(objects) > 3:
                    objects_display += f" (+{len(objects)-3} more)"
            except:
                objects_display = "Error parsing"
            
            # Truncate description for display
            desc_display = scene_desc[:37] + "..." if len(scene_desc) > 40 else scene_desc
            
            print(f"{scene_display:<30} | {image_display:<30} | {desc_display:<40} | {objects_display}")
        
        print("-"*120)
        print(f"Total records: {len(results)}")


class JSONResultsManager:
    def __init__(self, json_path: str = "inference_results.json"):
        """
        Initialize the JSON results manager.
        
        Args:
            json_path: Path to the JSON results file
        """
        self.json_path = json_path
        self.results = []
        self.load_existing_results()
    
    def load_existing_results(self):
        """Load existing results from JSON file if it exists."""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                    self.results = data.get('results', [])
                print(f"Loaded {len(self.results)} existing results from {self.json_path}")
            except Exception as e:
                print(f"Warning: Could not load existing results from {self.json_path}: {e}")
                self.results = []
        else:
            self.results = []
    
    def add_inference_result(self, scene_path: str, image_path: str, 
                           scene_description: str, objects_present: List[str]):
        """
        Add inference results to the JSON file.
        
        Args:
            scene_path: Path to the scene/object file
            image_path: Path to the image file
            scene_description: Description of the scene
            objects_present: List of objects present in the image
        """
        # Check if this result already exists
        for i, result in enumerate(self.results):
            if result['scene_path'] == scene_path and result['image_path'] == image_path:
                # Update existing result
                self.results[i] = {
                    'scene_path': scene_path,
                    'image_path': image_path,
                    'scene_description': scene_description,
                    'objects_present': objects_present,
                    'inference_timestamp': datetime.now().isoformat()
                }
                return
        
        # Add new result
        self.results.append({
            'scene_path': scene_path,
            'image_path': image_path,
            'scene_description': scene_description,
            'objects_present': objects_present,
            'inference_timestamp': datetime.now().isoformat()
        })
    
    def save_results(self):
        """Save all results to JSON file."""
        data = {
            'metadata': {
                'total_results': len(self.results),
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            'results': self.results
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.results)} results to {self.json_path}")
    
    def print_results_table(self):
        """Print all results in a formatted table."""
        if not self.results:
            print("No results found in JSON file.")
            return
        
        print("\n" + "="*120)
        print("INFERENCE RESULTS (JSON)")
        print("="*120)
        print(f"{'Scene Path':<30} | {'Image Path':<30} | {'Scene Description':<40} | {'Objects Present'}")
        print("-"*120)
        
        for result in self.results:
            scene_path = result['scene_path']
            image_path = result['image_path']
            scene_desc = result['scene_description']
            objects = result['objects_present']
            
            # Truncate long paths for display
            scene_display = scene_path[-29:] if len(scene_path) > 30 else scene_path
            image_display = image_path[-29:] if len(image_path) > 30 else image_path
            
            # Format objects for display
            objects_display = ", ".join(objects[:3])  # Show first 3 objects
            if len(objects) > 3:
                objects_display += f" (+{len(objects)-3} more)"
            
            # Truncate description for display
            desc_display = scene_desc[:37] + "..." if len(scene_desc) > 40 else scene_desc
            
            print(f"{scene_display:<30} | {image_display:<30} | {desc_display:<40} | {objects_display}")
        
        print("-"*120)
        print(f"Total records: {len(self.results)}")


class AsyncImageProcessor:
    def __init__(self, api_key: Optional[str] = None, config_path: str = "config.json"):
        """
        Initialize the async image processor with Mistral Vision API.
        
        Args:
            api_key: Mistral API key
            config_path: Path to the configuration file
        """
        self.vision_api = MistralVisionAPI(api_key=api_key, config_path=config_path)
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.base_url = "https://api.mistral.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Load configuration
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            default_config = {
                "model": "mistral-large-latest",
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            
            return config
        except FileNotFoundError:
            return {
                "model": "mistral-large-latest",
                "max_tokens": 1000,
                "temperature": 0.7
            }
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode an image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    async def run_vision_inference_async(self, session: aiohttp.ClientSession, image_path: str, prompt: str) -> dict:
        """
        Run vision inference asynchronously.
        
        Args:
            session: aiohttp session
            image_path: Path to the image file
            prompt: Text prompt
            
        Returns:
            API response dictionary
        """
        try:
            # Encode image to base64
            image_base64 = self.encode_image_to_base64(image_path)
            
            # Prepare request payload
            payload = {
                "model": self.config["model"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                "max_tokens": self.config["max_tokens"],
                "temperature": self.config["temperature"]
            }
            
            # Make async request
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {"error": f"API request failed: {response.status} - {error_text}"}
                    
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    async def get_scene_description_async(self, session: aiohttp.ClientSession, image_path: str) -> str:
        """
        Get a detailed scene description for an image asynchronously.
        
        Args:
            session: aiohttp session
            image_path: Path to the image file
            
        Returns:
            Scene description string
        """
        prompt = "Describe what you see in this image. Focus on the main objects, and their arrangement, and the overall scene."
        
        try:
            result = await self.run_vision_inference_async(session, image_path, prompt)
            
            if 'choices' in result:
                return result.get('choices')[0].get('message', {}).get('content', 'No description available')
            else:
                return 'No description available'
        except Exception as e:
            return f"Error getting description: {str(e)}"
    
    async def get_comprehensive_objects_list_async(self, session: aiohttp.ClientSession, image_path: str) -> List[str]:
        """
        Get a comprehensive list of all objects in the image asynchronously.
        
        Args:
            session: aiohttp session
            image_path: Path to the image file
            
        Returns:
            List of objects with synonyms and similar terms
        """
        prompt = """List ALL possible objects you can see in this image. Include:
1. Every object visible in the image
2. Synonyms and alternative names for each object
3. Similar or related objects that could be used to describe what you see
4. Both specific and general terms
5. Include objects that might be part of larger objects

Return ONLY a comma-separated list of object names, nothing else. Be comprehensive and thorough. Include as many relevant terms as possible."""
        
        try:
            result = await self.run_vision_inference_async(session, image_path, prompt)
            
            if 'choices' in result:
                content = result.get('choices')[0].get('message', {}).get('content', '')
            else:
                content = ''
            
            # Parse the comma-separated list
            if content and len(content) > 0:
                # Clean up the response and split by commas
                objects = [obj.strip().lower() for obj in content.split(',')]
                
                # Remove empty strings and common non-object words
                filtered_objects = []
                skip_words = {
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                    'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those', 'there', 'here', 'can',
                    'see', 'you', 'image', 'shows', 'showing', 'visible', 'appears', 'looks', 'seems',
                    'main', 'object', 'objects', 'scene', 'arrangement', 'overall', 'focus', 'what',
                    'where', 'when', 'how', 'why', 'which', 'who', 'whom', 'whose', 'include', 'possible',
                    'every', 'each', 'all', 'visible', 'comprehensive', 'thorough', 'relevant', 'terms',
                    'synonyms', 'alternative', 'names', 'similar', 'related', 'specific', 'general',
                    'part', 'larger', 'return', 'only', 'comma', 'separated', 'list', 'nothing', 'else',
                    'be', 'as', 'many', 'possible', 'list', 'all', 'objects', 'can', 'see', 'image'
                }
                
                for obj in objects:
                    obj = obj.strip()
                    if obj and obj not in skip_words and len(obj) > 1:
                        filtered_objects.append(obj)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_objects = []
                for obj in filtered_objects:
                    if obj not in seen:
                        seen.add(obj)
                        unique_objects.append(obj)
                
                return unique_objects[:20]  # Limit to top 20 objects to avoid too much noise
            else:
                return []
                
        except Exception as e:
            print(f"Error getting comprehensive objects list: {str(e)}")
            return []
    
    async def process_single_image_async(self, session: aiohttp.ClientSession, image_path: str, scene_path: str = "unknown") -> Dict:
        """
        Process a single image asynchronously to get scene description and comprehensive objects list.
        
        Args:
            session: aiohttp session
            image_path: Path to the image file
            scene_path: Path to the scene/object file (default: "unknown")
            
        Returns:
            Dictionary containing processing results
        """
        print(f"Processing: {image_path}")
        
        # Get scene description
        print("  Getting scene description...")
        scene_description = await self.get_scene_description_async(session, image_path)
        
        # Get comprehensive objects list directly
        print("  Getting comprehensive objects list...")
        objects_present = await self.get_comprehensive_objects_list_async(session, image_path)
        
        return {
            'scene_path': scene_path,
            'image_path': image_path,
            'scene_description': scene_description,
            'objects_present': objects_present
        }
    
    async def process_images_batch_async(self, image_paths: List[str], scene_path: str = "unknown", max_concurrent: int = 3) -> List[Dict]:
        """
        Process multiple images in batch asynchronously with controlled concurrency.
        
        Args:
            image_paths: List of image file paths
            scene_path: Path to the scene/object file (default: "unknown")
            max_concurrent: Maximum number of concurrent requests (default: 3)
            
        Returns:
            List of processing results
        """
        results = []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(image_path: str) -> Dict:
            async with semaphore:
                try:
                    return await self.process_single_image_async(session, image_path, scene_path)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    return {
                        'scene_path': scene_path,
                        'image_path': image_path,
                        'scene_description': f"Error: {str(e)}",
                        'objects_present': []
                    }
        
        # Create aiohttp session
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create tasks for all images
            tasks = [process_with_semaphore(image_path) for image_path in image_paths]
            
            # Process all tasks concurrently
            print(f"Processing {len(image_paths)} images with max {max_concurrent} concurrent requests...")
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"Error processing image {i+1}: {result}")
                    results.append({
                        'scene_path': scene_path,
                        'image_path': image_paths[i],
                        'scene_description': f"Error: {str(result)}",
                        'objects_present': []
                    })
                else:
                    results.append(result)
        
        return results


def find_image_files(images_dir: str) -> List[str]:
    """
    Find all image files in the specified directory.
    
    Args:
        images_dir: Path to the images directory
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    if not os.path.exists(images_dir):
        print(f"Images directory '{images_dir}' does not exist.")
        return []
    
    image_paths = []
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    
    return sorted(image_paths)


def main():
    """Main function to process images and save results to database or JSON."""
    parser = argparse.ArgumentParser(description='Process images and save inference results to database or JSON')
    parser.add_argument('--images-dir', '-i', 
                       type=str, 
                       default="images",
                       help='Path to directory containing images (default: images)')
    parser.add_argument('--scene-path', '-s',
                       type=str,
                       default="unknown",
                       help='Path to the scene/object file (default: unknown)')
    parser.add_argument('--output-format', '-f',
                       type=str,
                       choices=['sql', 'json'],
                       default='json',
                       help='Output format: sql or json (default: json)')
    parser.add_argument('--db-path', '-d',
                       type=str,
                       default="inference_results.db",
                       help='Path to SQLite database file (default: inference_results.db)')
    parser.add_argument('--json-path', '-j',
                       type=str,
                       default="inference_results.json",
                       help='Path to JSON results file (default: inference_results.json)')
    parser.add_argument('--config-file', '-c',
                       type=str,
                       default="config.json",
                       help='Path to configuration file (default: config.json)')
    parser.add_argument('--show-results', '-r',
                       action='store_true',
                       help='Show results table after processing')
    parser.add_argument('--max-concurrent', '-m',
                       type=int,
                       default=3,
                       help='Maximum number of concurrent API requests (default: 3)')
    
    args = parser.parse_args()
    
    # Find image files
    print(f"Looking for images in: {args.images_dir}")
    image_paths = find_image_files(args.images_dir)
    
    if not image_paths:
        print("No image files found. Please check the images directory.")
        return
    
    print(f"Found {len(image_paths)} image files:")
    for path in image_paths:
        print(f"  {path}")
    
    # Initialize output manager based on format
    if args.output_format == 'sql':
        print(f"\nInitializing SQLite database: {args.db_path}")
        output_manager = InferenceDatabase(args.db_path)
    else:  # json
        print(f"\nInitializing JSON output: {args.json_path}")
        output_manager = JSONResultsManager(args.json_path)
    
    # Initialize image processor
    print("Initializing image processor...")
    processor = AsyncImageProcessor(config_path=args.config_file)
    
    # Process images asynchronously
    print(f"\nProcessing {len(image_paths)} images with max {args.max_concurrent} concurrent requests...")
    start_time = time.time()
    
    # Run async processing
    results = asyncio.run(processor.process_images_batch_async(
        image_paths, 
        args.scene_path, 
        max_concurrent=args.max_concurrent
    ))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save results
    print(f"\nSaving results...")
    for result in results:
        if args.output_format == 'sql':
            output_manager.insert_inference_result(
                result['scene_path'],
                result['image_path'],
                result['scene_description'],
                result['objects_present']
            )
        else:  # json
            output_manager.add_inference_result(
                result['scene_path'],
                result['image_path'],
                result['scene_description'],
                result['objects_present']
            )
    
    # Finalize JSON output if needed
    if args.output_format == 'json':
        output_manager.save_results()
    
    # Print summary
    print(f"\nProcessing completed!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/len(image_paths):.2f} seconds")
    
    if args.output_format == 'sql':
        print(f"Results saved to: {args.db_path}")
    else:
        print(f"Results saved to: {args.json_path}")
    
    # Show results if requested
    if args.show_results:
        output_manager.print_results_table()


if __name__ == "__main__":
    main()
