from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from typing_extensions import override

import numpy as np
import copy


class RandomSample:
    """
    Encapsulates a sampled value along with its probability density.
    
    Usage:
        sample = sampler.sample()  # Returns RandomSample in [0,1]
        print(f"Value: {sample.value}, PDF: {sample.pdf}")
    """
    
    def __init__(self, value: Any, pdf: float):
        self.value = value
        self.pdf = pdf
        
    def unpack(self) -> tuple[Any, float]:
        """Return (value, pdf) tuple for backward compatibility"""
        return self.value, self.pdf
        
    def __repr__(self):
        return f"RandomSample(value={self.value}, pdf={self.pdf})"


class PSSampler(ABC):
    """
    Primary Space Sampler - samples in [0,1] uniform space.
    
    New Architecture:
    1. Distributions register their dimension requirements via register_dimension()
    2. RenderSpaceState calls sample() to get total PDF
    3. Distributions call get() to retrieve individual dimension values
    4. Bounds checking prevents exceeding registered dimensions
    """
    
    def __init__(self, name: Optional[str] = None, rng: Optional[Any] = None):
        # Don't store module references - create RNG generator instead
        if rng is not None:
            self.rng = rng
        elif name is not None:
            # Create a proper Generator object, not a module reference
            self.rng = np.random.default_rng(list(map(ord, name)))
        else:
            # Create a proper Generator object, not a module reference  
            self.rng = np.random.default_rng()
            
        # Dimension tracking
        self.total_dimensions = 0
        self.current_dimension = 0
        self.sampled_values: List[float] = []

    def register_dimension(self) -> int:
        """
        Register a dimension requirement and return its index.
        Called by distributions during initialization.
        
        Returns:
            Index of the registered dimension
        """
        dim_index = self.total_dimensions
        self.total_dimensions += 1
        return dim_index
    

    def reinitialize(self):
        pass 

        

    @abstractmethod
    def sample(self) -> float:
        """
        Sample all registered dimensions at once and return total PDF.
        This should populate internal storage for get() calls.
        
        Returns:
            Total PDF for the multi-dimensional sample
        """
        pass

    def get(self) -> float:
        """
        Get the next dimension value. Called sequentially by distributions.
        
        Returns:
            Value in [0,1] for the current dimension
        """
        if self.current_dimension >= self.total_dimensions:
            raise Exception(f"Sampler get() called beyond registered dimensions: {self.current_dimension} >= {self.total_dimensions}")
            
        if self.current_dimension >= len(self.sampled_values):
            raise Exception(f"No sampled values available for dimension {self.current_dimension}")
            
        value = self.sampled_values[self.current_dimension]
        self.current_dimension += 1
        return value

    def reset_counter(self):
        """Reset the dimension counter for a new sampling round"""
        self.current_dimension = 0

    @abstractmethod 
    def sample_primary(self, num_samples: int = 1) -> List[RandomSample]:
        """
        Legacy method for backward compatibility.
        New code should use sample()/get() pattern.
        
        Args:
            num_samples: Number of samples to generate (default: 1)
            
        Returns:
            List[RandomSample] - always returns a list, even for single samples
        """
        pass

    def num_dimensions(self) -> int:
        """
        Return the number of dimensions registered with this sampler.
        """
        return self.total_dimensions

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize sampler state for distributed computing"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PSSampler":
        """Reconstruct sampler from serialized state"""
        pass


SAMPLER_REGISTRY = {}  # str -> Type[PSSampler]

def register_sampler(cls):
    """
    Decorator that adds a PSSampler subclass to the registry.
    """
    SAMPLER_REGISTRY[cls.__name__] = cls
    return cls


@register_sampler
class DummySampler(PSSampler):
    """
    Random Number Generator sampler - uniform random sampling in [0,1].
    
    Usage:
        # New architecture:
        sampler = RNGSampler()
        dist1.register_with_sampler(sampler)  # registers dimensions
        dist2.register_with_sampler(sampler)
        total_pdf = sampler.sample()  # samples all dimensions
        val1 = sampler.get()  # gets first dimension
        val2 = sampler.get()  # gets second dimension
        
        # Legacy usage:
        sampler = RNGSampler(name="reproducible_seed")
        sample = sampler.sample_primary()  # RandomSample with pdf=1.0
    """
    
    @override
    def sample(self) -> float:
        """Sample all registered dimensions and return total PDF"""
        if self.total_dimensions == 0:
            raise Exception("No dimensions registered with RNGSampler")
            
        # Generate all dimension values at once
        self.sampled_values = [float(self.rng.uniform(0, 1)) for _ in range(self.total_dimensions)]
        self.reset_counter()

        # For uniform distribution, total PDF is 1.0 (product of individual PDFs of 1.0)
        return 1.0
    
    @override
    def sample_primary(self, num_samples: int = 1) -> List[RandomSample]:
        """Legacy method - for uniform distribution in [0,1], pdf = 1.0"""
        samples = []
        for _ in range(num_samples):
            samples.append(RandomSample(float(self.rng.uniform(0, 1)), 1.0))
        
        return samples

    @override
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "RNGSampler",
            "total_dimensions": self.total_dimensions,
            "rng_state": self.rng.bit_generator.state if hasattr(self.rng, 'bit_generator') else None
        }

    @classmethod
    @override
    def from_dict(cls, data: Dict[str, Any]) -> "RNGSampler":
        sampler = cls()
        sampler.total_dimensions = data.get("total_dimensions", 0)
        # Note: RNG state restoration is complex, may need specific handling
        return sampler


