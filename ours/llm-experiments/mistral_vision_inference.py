#!/usr/bin/env python3
"""
Minimal example for using Mistral's Vision Language Model API
to run inference on an image.
"""

import os
import base64
import requests
import json
import re
import time
import argparse
from typing import Optional, Union, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

class MistralVisionAPI:
    def __init__(self, api_key: Optional[str] = None, config_path: str = "config.json", patterns_file: Optional[str] = None):
        """
        Initialize the Mistral Vision API client.
        
        Args:
            api_key: Mistral API key. If None, will try to get from MISTRAL_API_KEY env var.
            config_path: Path to the configuration JSON file
            patterns_file: Path to the JSON file containing object patterns
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable or pass it to the constructor.")
        
        # Load configuration from config.json
        self.config = self._load_config(config_path)
        
        # Load patterns from JSON file if provided
        self.patterns_data = self._load_patterns_from_file(patterns_file) if patterns_file else None
        self.patterns = self._get_patterns_from_data()
        
        self.base_url = "https://api.mistral.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Set default values if not present in config
            default_config = {
                "model": "mistral-large-latest",
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            # Update defaults with config values
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            
            print(f"Loaded configuration from {config_path}:")
            print(f"  Model: {config['model']}")
            print(f"  Max tokens: {config['max_tokens']}")
            print(f"  Temperature: {config['temperature']}")
            
            return config
            
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default values.")
            return {
                "model": "mistral-large-latest",
                "max_tokens": 1000,
                "temperature": 0.7
            }
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {config_path}: {e}. Using default values.")
            return {
                "model": "mistral-large-latest",
                "max_tokens": 1000,
                "temperature": 0.7
            }
    
    def _load_patterns_from_file(self, patterns_file: str) -> dict:
        """
        Load patterns from a JSON file.
        
        Args:
            patterns_file: Path to the JSON file containing object patterns
            
        Returns:
            Dictionary containing patterns data
        """
        try:
            with open(patterns_file, 'r') as f:
                patterns_data = json.load(f)
            
            # Validate the structure
            if "object_name" not in patterns_data:
                raise ValueError("JSON file must contain 'object_name' field")
            if "base_patterns" not in patterns_data:
                raise ValueError("JSON file must contain 'base_patterns' field")
            
            print(f"Loaded patterns for object '{patterns_data['object_name']}' from {patterns_file}")
            print(f"  Total patterns: {len(patterns_data['base_patterns'])}")
            
            return patterns_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Patterns file {patterns_file} not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {patterns_file}: {e}")
    
    def _get_patterns_from_data(self) -> List[str]:
        """
        Get patterns from loaded data or return default patterns.
        
        Returns:
            List of patterns
        """
        if self.patterns_data:
            return self.patterns_data["base_patterns"]
        else:
            return self._get_default_patterns()
    
    def _get_default_patterns(self) -> List[str]:
        """
        Get default search patterns if no JSON file is provided.
        
        Returns:
            List of default search patterns
        """
        return [
            "Is {object} present in the image?",
            "Is there {object} in the image?",
            "Can you see {object} in the image?",
            "Do you see {object} in the image?",
            "Is {object} visible in the image?",
            "Does this image contain {object}?",
            "Are there {object} in the image?",
            "Is {object} in the image?",
            "Does the image show {object}?",
            "Is {object} shown in the image?",
            "Can you find {object} in the image?",
            "Is {object} there in the image?",
            "Does this picture have {object}?",
            "Is {object} present?",
            "Can you spot {object}?",
            "Is {object} detectable?",
            "Does the photo contain {object}?",
            "Are any {object} visible?",
            "Is {object} apparent in the image?",
            "Can you identify {object}?"
        ]
    
    def get_object_name(self) -> str:
        """
        Get the object name from loaded patterns data.
        
        Returns:
            Object name or "object" as default
        """
        if self.patterns_data and "object_name" in self.patterns_data:
            return self.patterns_data["object_name"]
        return "object"
    
    def generate_presence_prompts(self) -> List[str]:
        """
        Generate presence prompts using the loaded patterns and object name.
        
        Returns:
            List of presence prompts
        """
        object_name = self.get_object_name()
        return [pattern.format(object=object_name) for pattern in self.patterns]
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode an image file to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _is_object_presence_question(self, prompt: str) -> bool:
        """
        Check if the prompt is asking about object presence using loaded patterns.
        
        Args:
            prompt: The user's prompt
            
        Returns:
            True if the prompt is asking about object presence, False otherwise
        """
        # Convert to lowercase for easier matching
        prompt_lower = prompt.lower().strip()
        
        # Check against loaded patterns
        presence_prompts = self.generate_presence_prompts()
        
        for presence_prompt in presence_prompts:
            # Convert pattern to lowercase for comparison
            pattern_lower = presence_prompt.lower()
            
            # Check if the prompt matches the pattern structure
            if pattern_lower in prompt_lower:
                return True
        
        # Fallback to regex patterns for edge cases
        presence_patterns = [
            r'is\s+\w+\s+present',
            r'is\s+there\s+\w+',
            r'can\s+you\s+see\s+\w+',
            r'do\s+you\s+see\s+\w+',
            r'is\s+\w+\s+visible',
            r'does\s+this\s+contain\s+\w+',
            r'are\s+there\s+\w+',
            r'is\s+\w+\s+in\s+the\s+image',
            r'does\s+the\s+image\s+show\s+\w+',
            r'is\s+\w+\s+shown',
            r'can\s+you\s+find\s+\w+',
            r'is\s+\w+\s+there'
        ]
        
        for pattern in presence_patterns:
            if re.search(pattern, prompt_lower):
                return True
        
        return False
    
    def _extract_object_name(self, prompt: str) -> str:
        """
        Extract the object name from a presence question or use loaded object name.
        
        Args:
            prompt: The user's prompt
            
        Returns:
            The object name being asked about
        """
        # If we have loaded patterns data, use the object name from there
        if self.patterns_data and "object_name" in self.patterns_data:
            return self.patterns_data["object_name"]
        
        # Otherwise, try to extract from the prompt using regex patterns
        prompt_lower = prompt.lower().strip()
        
        # Try to extract object name using regex patterns
        patterns = [
            r'is\s+(\w+(?:\s+\w+)*)\s+present',
            r'is\s+there\s+(\w+(?:\s+\w+)*)',
            r'can\s+you\s+see\s+(\w+(?:\s+\w+)*)',
            r'do\s+you\s+see\s+(\w+(?:\s+\w+)*)',
            r'is\s+(\w+(?:\s+\w+)*)\s+visible',
            r'does\s+this\s+contain\s+(\w+(?:\s+\w+)*)',
            r'are\s+there\s+(\w+(?:\s+\w+)*)',
            r'is\s+(\w+(?:\s+\w+)*)\s+in\s+the\s+image',
            r'does\s+the\s+image\s+show\s+(\w+(?:\s+\w+)*)',
            r'is\s+(\w+(?:\s+\w+)*)\s+shown',
            r'can\s+you\s+find\s+(\w+(?:\s+\w+)*)',
            r'is\s+(\w+(?:\s+\w+)*)\s+there'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                return match.group(1).strip()
        
        # Fallback: try to extract any noun phrase after common question words
        fallback_patterns = [
            r'(?:is|are|can|do|does)\s+(\w+(?:\s+\w+)*)',
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                return match.group(1).strip()
        
        return "object"  # Default fallback
    
    def _parse_boolean_response(self, response: str) -> bool:
        """
        Parse the LLM response to extract a boolean value.
        
        Args:
            response: The LLM response text
            
        Returns:
            True if the response indicates presence, False otherwise
        """
        response_lower = response.lower().strip()
        
        # Positive indicators
        positive_indicators = [
            'yes', 'true', 'present', 'visible', 'there', 'found', 'can see',
            'do see', 'is visible', 'is present', 'is there', 'exists',
            'appears', 'shows', 'contains', 'includes', 'has'
        ]
        
        # Negative indicators
        negative_indicators = [
            'no', 'false', 'not present', 'not visible', 'not there', 'not found',
            'cannot see', 'do not see', 'is not visible', 'is not present',
            'is not there', 'does not exist', 'does not appear', 'does not show',
            'does not contain', 'does not include', 'does not have'
        ]
        
        # Check for positive indicators
        for indicator in positive_indicators:
            if indicator in response_lower:
                print(indicator)
                return True
        
        # Check for negative indicators
        for indicator in negative_indicators:
            if indicator in response_lower:
                return False
        
        # If no clear indicators found, try to extract yes/no from the beginning
        if response_lower.startswith('yes'):
            return True
        elif response_lower.startswith('no'):
            return False
        
        # Default to False if unclear
        return False
    
    def run_vision_inference(self, 
                           image_path: str, 
                           prompt: str,
                           model: Optional[str] = None,
                           max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None) -> dict:
        """
        Run vision inference on an image using Mistral's API.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt describing what to analyze in the image
            model: Model to use (default: from config.json)
            max_tokens: Maximum tokens in response (default: from config.json)
            temperature: Temperature for response generation (default: from config.json)
            
        Returns:
            API response as dictionary
        """
        # Use provided parameters or fall back to config values
        model = model or self.config["model"]
        max_tokens = max_tokens or self.config["max_tokens"]
        temperature = temperature or self.config["temperature"]
        
        # Encode image to base64
        image_base64 = self.encode_image_to_base64(image_path)
        
        # Prepare the request payload
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Make the API request
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    def run_vision_inference_timed(self, 
                                 image_path: str, 
                                 prompt: str,
                                 model: Optional[str] = None,
                                 max_tokens: Optional[int] = None,
                                 temperature: Optional[float] = None) -> Tuple[dict, float]:
        """
        Run vision inference on an image and return the result with timing information.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt describing what to analyze in the image
            model: Model to use (default: from config.json)
            max_tokens: Maximum tokens in response (default: from config.json)
            temperature: Temperature for response generation (default: from config.json)
            
        Returns:
            Tuple of (API response as dictionary, inference time in seconds)
        """
        start_time = time.time()
        
        try:
            result = self.run_vision_inference(image_path, prompt, model, max_tokens, temperature)
            end_time = time.time()
            inference_time = end_time - start_time
            
            return result, inference_time
            
        except Exception as e:
            end_time = time.time()
            inference_time = end_time - start_time
            raise Exception(f"Inference failed after {inference_time:.2f}s: {str(e)}")
    
    def check_object_presence(self, 
                            image_path: str, 
                            prompt: str,
                            model: Optional[str] = None,
                            max_tokens: Optional[int] = None,
                            temperature: Optional[float] = None) -> Union[bool, dict]:
        """
        Check if an object is present in an image and return a boolean response.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt asking about object presence
            model: Model to use (default: from config.json)
            max_tokens: Maximum tokens in response (default: from config.json)
            temperature: Temperature for response generation (default: from config.json)
            
        Returns:
            Boolean indicating presence if it's a presence question, otherwise the full API response
        """
        # Check if this is an object presence question
        if not self._is_object_presence_question(prompt):
            # If not a presence question, return the full response
            return self.run_vision_inference(image_path, prompt, model, max_tokens, temperature)
        
        # Extract the object name for better prompting
        object_name = self._extract_object_name(prompt)
        
        # Create a more specific prompt for binary response
        binary_prompt = f"""Look at this image and answer with only 'yes' or 'no': Is {object_name} present in this image?

Please respond with only 'yes' or 'no'."""
        
        # Use lower temperature for more consistent binary responses
        binary_temperature = min(temperature or self.config["temperature"], 0.3)
        binary_max_tokens = min(max_tokens or self.config["max_tokens"], 50)
        
        # Run the inference
        result = self.run_vision_inference(
            image_path, 
            binary_prompt, 
            model, 
            binary_max_tokens, 
            binary_temperature
        )
        
        # Extract the response text
        if "choices" in result and len(result["choices"]) > 0:
            response_text = result["choices"][0]["message"]["content"]
            
            # Parse the response to get boolean
            is_present = self._parse_boolean_response(response_text)
            
            return is_present
        else:
            # If we can't parse the response, return the full result
            return False
    
    def check_object_presence_timed(self, 
                                  image_path: str, 
                                  prompt: str,
                                  model: Optional[str] = None,
                                  max_tokens: Optional[int] = None,
                                  temperature: Optional[float] = None) -> Tuple[Union[bool, dict], float]:
        """
        Check if an object is present in an image and return a boolean response with timing.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt asking about object presence
            model: Model to use (default: from config.json)
            max_tokens: Maximum tokens in response (default: from config.json)
            temperature: Temperature for response generation (default: from config.json)
            
        Returns:
            Tuple of (result, inference time in seconds)
        """
        start_time = time.time()
        
        try:
            result = self.check_object_presence(image_path, prompt, model, max_tokens, temperature)
            end_time = time.time()
            inference_time = end_time - start_time
            
            return result, inference_time
            
        except Exception as e:
            end_time = time.time()
            inference_time = end_time - start_time
            raise Exception(f"Object presence check failed after {inference_time:.2f}s: {str(e)}")
    
    def run_batch_inference(self, 
                          image_paths: List[str], 
                          prompt: str,
                          model: Optional[str] = None,
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          max_workers: int = 5) -> List[Tuple[str, dict, float]]:
        """
        Run vision inference on multiple images in parallel.
        
        Args:
            image_paths: List of paths to image files
            prompt: Text prompt describing what to analyze in the images
            model: Model to use (default: from config.json)
            max_tokens: Maximum tokens in response (default: from config.json)
            temperature: Temperature for response generation (default: from config.json)
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of tuples (image_path, result, inference_time)
        """
        results = []
        
        def process_single_image(image_path):
            try:
                result, inference_time = self.run_vision_inference_timed(
                    image_path, prompt, model, max_tokens, temperature
                )
                return image_path, result, inference_time
            except Exception as e:
                return image_path, {"error": str(e)}, 0.0
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(process_single_image, image_path): image_path 
                for image_path in image_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                image_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append((image_path, {"error": str(e)}, 0.0))
        
        # Sort results by original order
        path_to_result = {path: result for path, result, _ in results}
        sorted_results = []
        for image_path in image_paths:
            if image_path in path_to_result:
                for path, result, time_taken in results:
                    if path == image_path:
                        sorted_results.append((path, result, time_taken))
                        break
        
        return sorted_results
    
    def run_batch_object_presence(self, 
                                image_paths: List[str], 
                                prompt: str,
                                model: Optional[str] = None,
                                max_tokens: Optional[int] = None,
                                temperature: Optional[float] = None,
                                max_workers: int = 5) -> List[Tuple[str, Union[bool, dict], float]]:
        """
        Check object presence in multiple images in parallel.
        
        Args:
            image_paths: List of paths to image files
            prompt: Text prompt asking about object presence
            model: Model to use (default: from config.json)
            max_tokens: Maximum tokens in response (default: from config.json)
            temperature: Temperature for response generation (default: from config.json)
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of tuples (image_path, result, inference_time)
        """
        results = []
        
        def process_single_image(image_path):
            try:
                result, inference_time = self.check_object_presence_timed(
                    image_path, prompt, model, max_tokens, temperature
                )
                return image_path, result, inference_time
            except Exception as e:
                return image_path, {"error": str(e)}, 0.0
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(process_single_image, image_path): image_path 
                for image_path in image_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                image_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append((image_path, {"error": str(e)}, 0.0))
        
        # Sort results by original order
        path_to_result = {path: result for path, result, _ in results}
        sorted_results = []
        for image_path in image_paths:
            if image_path in path_to_result:
                for path, result, time_taken in results:
                    if path == image_path:
                        sorted_results.append((path, result, time_taken))
                        break
        
        return sorted_results
    
    def demonstrate_patterns(self, object_name: str = "cat") -> None:
        """
        Demonstrate how patterns are generated for a specific object.
        
        Args:
            object_name: Name of the object to demonstrate patterns for
        """
        print(f"\nDemonstrating patterns for '{object_name}':")
        print("-" * 50)
        
        generated_patterns = self.generate_patterns_for_object(object_name)
        for i, pattern in enumerate(generated_patterns, 1):
            print(f"{i:2d}. {pattern}")
        
        print(f"\nTotal patterns: {len(generated_patterns)}")
    
    def demonstrate_loaded_patterns(self) -> None:
        """
        Demonstrate the loaded patterns and object name.
        """
        print(f"\nLoaded Object: '{self.get_object_name()}'")
        print(f"Total Patterns: {len(self.patterns)}")
        print("-" * 50)
        
        presence_prompts = self.generate_presence_prompts()
        for i, prompt in enumerate(presence_prompts, 1):
            print(f"{i:2d}. {prompt}")
        
        print(f"\nObject Name: {self.get_object_name()}")
        print(f"Patterns loaded from: {'JSON file' if self.patterns_data else 'default patterns'}")


def main():
    """
    Example usage of the Mistral Vision API with timing and batch processing.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Mistral Vision API with custom patterns')
    parser.add_argument('--image-dir', '-i', 
                       type=str, 
                       default="images",
                       help='Path to directory containing images (default: images)')
    parser.add_argument('--object-name', '-o', 
                       type=str, 
                       default="object",)
    parser.add_argument('--patterns-file', '-p', 
                       type=str, 
                       default="search_patterns/object.json",
                       help='Path to JSON file containing object patterns (default: search_patterns/object.json)')
    parser.add_argument('--config-file', '-c',
                       type=str,
                       default="config.json",
                       help='Path to configuration file (default: config.json)')
    
    args = parser.parse_args()
    
    # Example usage
    try:
        # Check if the patterns file exists
        if os.path.exists(args.patterns_file):
            print(f"Using patterns file: {args.patterns_file}")
            vision_api = MistralVisionAPI(patterns_file=args.patterns_file, config_path=args.config_file)
        else:
            print(f"Patterns file {args.patterns_file} not found. Using default patterns.")
            vision_api = MistralVisionAPI(config_path=args.config_file)
        
        # Demonstrate the loaded patterns
        # print("=" * 60)
        # print("LOADED SEARCH PATTERNS")
        # print("=" * 60)
        # vision_api.demonstrate_loaded_patterns()
        
        # Example image paths (you'll need to provide your own images)
        
        images_dir = 'images'
        image_paths = [
            os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        # Filter to only existing images
        existing_images = [path for path in image_paths if os.path.exists(path)]
        
        if not existing_images:
            print("No image files found. Please provide valid image paths.")
            return
        
        # Test single image with timing
        print("=" * 60)
        print("SINGLE IMAGE INFERENCE WITH TIMING")
        print("=" * 60)
        
        image_path = existing_images[0]
        presence_prompt = f"Is {args.object_name} present in the image?"
        
        print(f"Processing: {image_path}")
        print(f"Prompt: {presence_prompt}")
        
        result, inference_time = vision_api.check_object_presence_timed(
            image_path, presence_prompt
        )
        
        print(f"Result: {result}")
        print(f"Inference time: {inference_time:.2f} seconds")
        
        # Test batch processing
        if len(existing_images) > 1:
            print("\n" + "=" * 60)
            print("BATCH INFERENCE WITH TIMING")
            print("=" * 60)
            
            batch_prompt = presence_prompt
            
            print(f"Processing {len(existing_images)} images in parallel...")
            print(f"Prompt: {batch_prompt}")
            
            batch_start_time = time.time()
            
            batch_results = vision_api.run_batch_object_presence(
                existing_images, 
                batch_prompt,
                max_workers=3  # Limit concurrent requests
            )
            
            batch_end_time = time.time()
            total_batch_time = batch_end_time - batch_start_time
            
            print(f"\nBatch Results:")
            print("-" * 40)
            
            total_inference_time = 0
            for image_path, result, inference_time in batch_results:
                print(f"{image_path}: {result} ({inference_time:.2f}s)")
                total_inference_time += inference_time
            
            print(f"\nBatch Statistics:")
            print(f"  Total batch time: {total_batch_time:.2f}s")
            print(f"  Total inference time: {total_inference_time:.2f}s")
            print(f"  Average per image: {total_inference_time/len(existing_images):.2f}s")
            print(f"  Parallel speedup: {total_inference_time/total_batch_time:.2f}x")
        
        # Test regular inference with timing
        print("\n" + "=" * 60)
        print("REGULAR INFERENCE WITH TIMING")
        print("=" * 60)
        
        # description_prompt = "Describe what you see in this image in detail."
        
        # print(f"Processing: {image_path}")
        # print(f"Prompt: {description_prompt}")
        
        # result, inference_time = vision_api.run_vision_inference_timed(
        #     image_path, description_prompt
        # )
        
        
        
        print(f"Inference time: {inference_time:.2f} seconds")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()