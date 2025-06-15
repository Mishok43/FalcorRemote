#!/usr/bin/env python3
"""
Simplified Mistral Vision API for yes/no object detection.
"""

import os
import base64
import requests
import json
import time
from typing import Optional, Tuple

class MistralVisionAPI:
    def __init__(self, api_key: Optional[str] = "WPu0KpNOAUFviCzNgnbWQuRbz7Zpwyde", config_path: str = "config.json", patterns_file: Optional[str] = None):
        """
        Initialize the Mistral Vision API client.

        Args:
            api_key: Mistral API key. If None, will try to get from MISTRAL_API_KEY env var.
            config_path: Path to the configuration JSON file
            patterns_file: Ignored - kept for compatibility
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable or pass it to the constructor.")

        # Load basic configuration
        self.config = self._load_config(config_path)

        self.base_url = "https://api.mistral.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Set defaults for yes/no questions
            default_config = {
                "model": "pixtral-12b-2409",
                "max_tokens": 50,
                "temperature": 0.1
            }

            for key, value in default_config.items():
                if key not in config:
                    config[key] = value

            return config

        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "model": "pixtral-12b-2409",
                "max_tokens": 50,
                "temperature": 0.1
            }

    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode an image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _parse_boolean_response(self, response: str) -> bool:
        """Parse the LLM response to extract yes/no."""
        response_lower = response.lower().strip()

        # Simple yes/no detection
        if any(word in response_lower for word in ['yes', 'true', 'present', 'visible', 'there', 'found']):
            return True
        elif any(word in response_lower for word in ['no', 'false', 'not', 'absent', 'cannot', 'don\'t']):
            return False

        # Default to False if unclear
        return False

    def run_vision_inference(self, image_path: str, prompt: str) -> dict:
        """Run vision inference on an image."""
        # Encode image to base64
        image_base64 = self.encode_image_to_base64(image_path)

        # Prepare the request payload
        payload = {
            "model": self.config["model"],
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
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"]
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

    def check_object_presence_timed(self, image_path: str, prompt: str) -> Tuple[bool, float]:
        """
        Check if an object is present in an image and return yes/no with timing.

        Args:
            image_path: Path to the image file
            prompt: Text prompt asking about object presence

        Returns:
            Tuple of (is_present: bool, inference_time: float)
        """
        start_time = time.time()

        try:
            # Create a simple yes/no prompt
            simple_prompt = f"Look at this image and answer only 'yes' or 'no': {prompt}"

            # Run the inference
            result = self.run_vision_inference(image_path, simple_prompt)

            # Extract the response text
            if "choices" in result and len(result["choices"]) > 0:
                response_text = result["choices"][0]["message"]["content"]
                is_present = self._parse_boolean_response(response_text)
            else:
                is_present = False

            end_time = time.time()
            inference_time = end_time - start_time

            return is_present, inference_time

        except Exception as e:
            end_time = time.time()
            inference_time = end_time - start_time
            raise Exception(f"Object presence check failed after {inference_time:.2f}s: {str(e)}")

    # Compatibility methods for async wrapper
    def get_object_name(self) -> str:
        return "object"

    @property
    def patterns(self):
        return ["Is there a {object} in the image?"]


def main():
    """Simple test of the API."""
    import argparse

    parser = argparse.ArgumentParser(description='Test Mistral Vision API')
    parser.add_argument('--image', '-i', type=str, required=True, help='Path to image file')
    parser.add_argument('--object', '-o', type=str, default='car', help='Object to look for')
    args = parser.parse_args()

    try:
        api = MistralVisionAPI()
        prompt = f"Is there a {args.object} in the image?"

        print(f"Checking for '{args.object}' in {args.image}")
        is_present, time_taken = api.check_object_presence_timed(args.image, prompt)

        print(f"Result: {'YES' if is_present else 'NO'}")
        print(f"Time: {time_taken:.2f}s")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
