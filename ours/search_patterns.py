#!/usr/bin/env python3
"""
Search Pattern Generator for Object Detection

This module generates various search patterns for different objects
to be used with vision language models for object presence detection.
"""

import json
import os
import re
from typing import List, Dict, Optional, Union
from pathlib import Path


class SearchPatternGenerator:
    """
    Generates search patterns for object detection queries.
    
    This class creates various ways to ask about object presence
    in images, including different question formats, synonyms,
    and variations that can be used with vision language models.
    """
    
    def __init__(self, output_dir: str = "search_patterns"):
        """
        Initialize the SearchPatternGenerator.
        
        Args:
            output_dir: Directory to save generated JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Base patterns for object presence questions
        self.base_patterns = [
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
        
        # Alternative question formats
        self.alternative_formats = [
            "Look for {object} in this image",
            "Search for {object}",
            "Find {object} if present",
            "Check if {object} exists",
            "Determine if {object} is present",
            "Look at this image and tell me if {object} is there",
            "Examine the image for {object}",
            "Scan the image for {object}",
            "Inspect the image for {object}",
            "Analyze the image for {object} presence"
        ]
        
    
    
    
    def generate_patterns(self, object_name: str, include_contexts: bool = True) -> Dict[str, List[str]]:
        """
        Generate all search patterns for a given object.
        
        Args:
            object_name: The object to generate patterns for
            include_contexts: Whether to include context-specific patterns
            
        Returns:
            Dictionary containing different types of patterns
        """
        
        patterns = {
            "object_name": object_name,
            "base_patterns": [],
        }
        
        # Generate base patterns for each object variation
        # Base patterns
        for pattern in self.base_patterns:
            patterns["base_patterns"].append(pattern.format(object=object_name))
        
        
        return patterns
    
    def save_patterns(self, object_name: str, patterns: Dict[str, List[str]], 
                     filename: Optional[str] = None) -> str:
        """
        Save generated patterns to a JSON file.
        
        Args:
            object_name: The object name
            patterns: The generated patterns dictionary
            filename: Optional custom filename
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"{object_name.lower().replace(' ', '_')}_search.json"
        
        filepath = self.output_dir / filename
        
        # Add metadata
        patterns_with_metadata = {
            "metadata": {
                "object_name": object_name,
                "total_patterns": sum(len(v) for v in patterns.values() if isinstance(v, list)),
            },
            "patterns": patterns
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(patterns_with_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Patterns saved to: {filepath}")
        return str(filepath)
    
    def generate_and_save(self, object_name: str, include_contexts: bool = True, 
                         filename: Optional[str] = None) -> str:
        """
        Generate patterns for an object and save them to JSON.
        
        Args:
            object_name: The object to generate patterns for
            include_contexts: Whether to include context-specific patterns
            filename: Optional custom filename
            
        Returns:
            Path to the saved file
        """
        patterns = self.generate_patterns(object_name, include_contexts)
        return self.save_patterns(object_name, patterns, filename)
    
    def load_patterns(self, filename: str) -> Dict[str, Union[str, List[str], Dict]]:
        """
        Load patterns from a JSON file.
        
        Args:
            filename: Name of the JSON file to load
            
        Returns:
            Dictionary containing the loaded patterns
        """
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Pattern file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data


def main():
    """
    Example usage of the SearchPatternGenerator.
    """
    # Initialize the generator
    generator = SearchPatternGenerator()
    
    # Example objects to generate patterns for
    example_objects = [
        "bug",
        "needle", 
        # "cat",
        # "car",
        # "book",
        # "helmet"
    ]
    
    print("Generating search patterns for example objects...")
    print("=" * 60)
    
    for obj in example_objects:
        print(f"\nGenerating patterns for: {obj}")
        print("-" * 40)
        
        # Generate and save patterns
        filepath = generator.generate_and_save(obj)
        
        # Load and display some statistics
        data = generator.load_patterns(os.path.basename(filepath))
        metadata = data["metadata"]
        patterns = data["patterns"]
        
        print(f"Total patterns generated: {metadata['total_patterns']}")
        print(f"Base patterns: {len(patterns['base_patterns'])}")
        
        # Show a few example patterns
        print("\nExample patterns:")
        for i, pattern in enumerate(patterns['base_patterns'][:3]):
            print(f"  {i+1}. {pattern}")
    
    print("\n" + "=" * 60)
    print("Pattern generation complete!")
    print(f"Files saved in: {generator.output_dir}")
    


if __name__ == "__main__":
    main()