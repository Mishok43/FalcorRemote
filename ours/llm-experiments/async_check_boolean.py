#!/usr/bin/env python3
"""
Simple async yes/no object detection using Mistral Vision API.
"""

import os
import base64
import requests
import json
import time
import asyncio
from typing import Optional, Tuple
from dataclasses import dataclass

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

class AsyncMistralVisionAPI:
    """Simple async Mistral Vision API for yes/no object detection."""

    def __init__(self, api_key: Optional[str] = "WPu0KpNOAUFviCzNgnbWQuRbz7Zpwyde", config_path: str = "config.json", patterns_file: Optional[str] = None):
        """Initialize the async Mistral Vision API client."""
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required.")

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

    def tensor_to_base64(self, tensor) -> str:
        """Convert tensor directly to base64 JPEG without disk I/O."""
        import io
        from PIL import Image

        # Ensure tensor is on CPU
        if hasattr(tensor, 'is_cuda') and tensor.is_cuda:
            tensor = tensor.cpu()

        # Convert to numpy and ensure correct range [0, 1] -> [0, 255]
        if hasattr(tensor, 'dtype') and (tensor.dtype == torch.float32 or tensor.dtype == torch.float64):
            import torch
            np_array = (torch.clamp(tensor, 0, 1) * 255).byte().numpy()
        else:
            np_array = tensor.numpy() if hasattr(tensor, 'numpy') else tensor

        # Ensure correct shape (H, W, C)
        if len(np_array.shape) == 3 and np_array.shape[2] >= 3:
            np_array = np_array[:, :, :3]  # Take RGB only

        # Convert to PIL Image
        pil_image = Image.fromarray(np_array, mode='RGB')

        # Convert to JPEG bytes in memory
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _parse_boolean_response(self, response: str) -> Tuple[bool, float]:
        """Parse the LLM response to extract yes/no and confidence."""
        response_lower = response.lower().strip()

        # Count positive and negative indicators
        positive_words = ['yes', 'true', 'present', 'visible', 'there', 'found', 'can see', 'i see']
        negative_words = ['no', 'false', 'not', 'absent', 'cannot', 'don\'t', 'not visible', 'not present']

        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)

        # Determine result
        if positive_count > negative_count:
            is_present = True
            confidence = min(0.95, 0.5 + (positive_count * 0.1))
        else:
            is_present = False
            confidence = min(0.95, 0.5 + (negative_count * 0.1))

        return is_present, confidence

    async def run_vision_inference(self, image_path: str, prompt: str) -> dict:
        """Run vision inference on an image asynchronously."""
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

        # Make async API request
        loop = asyncio.get_event_loop()

        def make_request():
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

        return await loop.run_in_executor(None, make_request)

    async def run_vision_inference_from_tensor(self, tensor, prompt: str) -> dict:
        """Run vision inference on a tensor directly without disk I/O."""
        # Convert tensor directly to base64
        image_base64 = self.tensor_to_base64(tensor)

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

        # Make async API request
        loop = asyncio.get_event_loop()

        def make_request():
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

        return await loop.run_in_executor(None, make_request)

    async def check_object_presence_async(self, image_path: str, object_name: str, prompt: Optional[str] = None) -> ObjectCheckResult:
        """
        Check for object presence in an image asynchronously.

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
            prompt = f"Is there a clearly recognizable {object_name} in the image?"

        # Create a simple yes/no prompt with high confidence requirement
        simple_prompt = f"Look at this image and answer only 'yes' or 'no': {prompt}. Only claim it if there's the object if you're 99% confident."

        try:
            # Run the inference
            result = await self.run_vision_inference(image_path, simple_prompt)

            # Extract the response text
            if "choices" in result and len(result["choices"]) > 0:
                response_text = result["choices"][0]["message"]["content"]
                is_present, confidence = self._parse_boolean_response(response_text)
            else:
                response_text = "No response"
                is_present = False
                confidence = 0.0

            processing_time = time.time() - start_time

            return ObjectCheckResult(
                object_name=object_name,
                image_path=image_path,
                is_present=is_present,
                confidence=confidence,
                response_text=response_text,
                processing_time=processing_time,
                prompt_used=simple_prompt
            )

        except Exception as e:
            processing_time = time.time() - start_time

            return ObjectCheckResult(
                object_name=object_name,
                image_path=image_path,
                is_present=False,
                confidence=0.0,
                response_text=f"Error: {str(e)}",
                processing_time=processing_time,
                prompt_used=simple_prompt
            )

    async def check_object_presence_from_tensor_async(self, tensor, object_name: str, prompt: Optional[str] = None) -> ObjectCheckResult:
        """
        Check for object presence in a tensor directly without disk I/O.

        Args:
            tensor: Image tensor (torch.Tensor or numpy array)
            object_name: Name of the object to check for
            prompt: Custom prompt (if None, will generate one)

        Returns:
            ObjectCheckResult with the check results
        """
        start_time = time.time()

        # Generate prompt if not provided
        if prompt is None:
            prompt = f"Is there a clearly recognizable {object_name} in the image?"

        # Create a simple yes/no prompt with high confidence requirement
        simple_prompt = f"Look at this image and answer only 'yes' or 'no': {prompt}. Only claim it if there's the object if you're 99% confident."

        try:
            # Run the inference directly from tensor
            result = await self.run_vision_inference_from_tensor(tensor, simple_prompt)

            # Extract the response text
            if "choices" in result and len(result["choices"]) > 0:
                response_text = result["choices"][0]["message"]["content"]
                is_present, confidence = self._parse_boolean_response(response_text)
            else:
                response_text = "No response"
                is_present = False
                confidence = 0.0

            processing_time = time.time() - start_time

            return ObjectCheckResult(
                object_name=object_name,
                image_path="<tensor>",  # No file path since we're using tensor
                is_present=is_present,
                confidence=confidence,
                response_text=response_text,
                processing_time=processing_time,
                prompt_used=simple_prompt
            )

        except Exception as e:
            processing_time = time.time() - start_time

            return ObjectCheckResult(
                object_name=object_name,
                image_path="<tensor>",
                is_present=False,
                confidence=0.0,
                response_text=f"Error: {str(e)}",
                processing_time=processing_time,
                prompt_used=simple_prompt
            )

    # Compatibility methods
    def get_object_name(self) -> str:
        return "object"

    @property
    def patterns(self):
        return ["Is there a {object} in the image?"]


# Simple test function
async def test_object_detection(image_path: str, object_name: str):
    """Test the object detection functionality."""
    api = AsyncMistralVisionAPI()

    print(f"Checking for '{object_name}' in {image_path}")
    result = await api.check_object_presence_async(image_path, object_name)

    print(f"Object: {result.object_name}")
    print(f"Present: {'YES' if result.is_present else 'NO'}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Response: {result.response_text}")
    print(f"Time: {result.processing_time:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple async object detection")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to image file")
    parser.add_argument("--object", "-o", type=str, default="car", help="Object to look for")
    args = parser.parse_args()

    asyncio.run(test_object_detection(args.image, args.object))

