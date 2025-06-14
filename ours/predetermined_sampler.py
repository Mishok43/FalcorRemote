from typing import List, Dict, Any, Optional
import numpy as np

class RandomSample:
    """Simple container for sample values and PDF"""
    def __init__(self, value, pdf):
        self.value = value
        self.pdf = pdf

class PSSampler:
    """Base sampler class"""
    def __init__(self, name: Optional[str] = None, rng: Optional[Any] = None):
        self.name = name
        self.rng = rng
        self.total_dimensions = 0
        self.sampled_values = []
        self.current_dimension_idx = 0

    def register_dimension(self):
        """Register a dimension with the sampler"""
        self.total_dimensions += 1

    def reset_counter(self):
        """Reset the dimension counter for get() calls"""
        self.current_dimension_idx = 0

    def get(self) -> float:
        """Get the next dimension value from current sample"""
        if self.current_dimension_idx >= len(self.sampled_values):
            raise Exception("No more dimensions available")
        value = self.sampled_values[self.current_dimension_idx]
        self.current_dimension_idx += 1
        return value

    def sample(self) -> float:
        """Sample and return PDF"""
        raise NotImplementedError()

    def sample_primary(self, num_samples: int = 1) -> List[RandomSample]:
        """Sample and return RandomSample objects"""
        raise NotImplementedError()

class PredeterminedSampler(PSSampler):
    """
    Precomputed samples for efficient batch rendering.

    Usage:
        # Generate batch from NFSampler
        nf_samples = nf_sampler.sample_primary(num_samples=250)
        presampled = PredeterminedSampler.from_samples(nf_samples)

        # Use for rendering
        for i in range(250):
            pdf = presampled.sample()  # Get PDF for current sample
            val1 = presampled.get()    # Fast lookup of first dimension
            val2 = presampled.get()    # Fast lookup of second dimension
            # ... etc for all 5 dimensions
    """

    def __init__(self, samples: List[List[float]], pdfs: Optional[List[float]] = None,
                 name: Optional[str] = None, rng: Optional[Any] = None):
        super().__init__(name, rng)

        # Store multi-dimensional samples
        self.samples = samples  # List of [List[float]] - each inner list is one multi-dimensional sample

        if pdfs is None:
            self.pdfs = [1.0] * len(samples)
        else:
            self.pdfs = pdfs

        self.next_sample_idx = 0

        # Infer dimensions from first sample
        if samples and len(samples) > 0:
            self.total_dimensions = len(samples[0])

    @classmethod
    def from_samples(cls, random_samples: List[RandomSample]) -> "PredeterminedSampler":
        """Create from list of RandomSample objects"""
        samples = []
        pdfs = []

        for sample in random_samples:
            # Handle both single values and lists
            if isinstance(sample.value, list):
                samples.append(sample.value)
            else:
                samples.append([sample.value])  # Wrap single value in list
            pdfs.append(sample.pdf)

        return cls(samples, pdfs)

    @classmethod
    def from_base_sampler(cls, base_sampler: PSSampler, num_samples: int) -> "PredeterminedSampler":
        """Generate precomputed multi-dimensional samples from another sampler"""
        samples_primary = base_sampler.sample_primary(num_samples)
        return cls.from_samples(samples_primary)

    def sample(self) -> float:
        """Use precomputed multi-dimensional sample and return its PDF"""
        if self.next_sample_idx >= len(self.samples):
            raise Exception("PredeterminedSampler exhausted")

        # Get the current multi-dimensional sample
        current_sample = self.samples[self.next_sample_idx]
        current_pdf = self.pdfs[self.next_sample_idx]

        # Store values for get() calls
        self.sampled_values = current_sample[:]
        self.reset_counter()

        self.next_sample_idx += 1
        return current_pdf

    def sample_primary(self, num_samples: int = 1) -> List[RandomSample]:
        """Return precomputed samples, supports multiple samples"""
        # Check if we have enough samples
        if self.next_sample_idx + num_samples > len(self.samples):
            raise Exception(f"PredeterminedSampler exhausted: need {num_samples} samples but only {len(self.samples) - self.next_sample_idx} remaining")

        # Generate all requested samples
        results = []
        for i in range(num_samples):
            sample_values = self.samples[self.next_sample_idx + i]
            pdf = self.pdfs[self.next_sample_idx + i]
            results.append(RandomSample(sample_values, pdf))

        self.next_sample_idx += num_samples
        return results

    def reset(self):
        """Reset to beginning of samples"""
        self.next_sample_idx = 0

    def remaining_samples(self) -> int:
        """Get number of remaining samples"""
        return len(self.samples) - self.next_sample_idx

    def total_samples(self) -> int:
        """Get total number of samples"""
        return len(self.samples)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "PredeterminedSampler",
            "samples": self.samples,
            "pdfs": self.pdfs,
            "next_sample_idx": self.next_sample_idx,
            "total_dimensions": self.total_dimensions
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredeterminedSampler":
        sampler = cls(data["samples"], data["pdfs"])
        sampler.next_sample_idx = data.get("next_sample_idx", 0)
        sampler.total_dimensions = data.get("total_dimensions", 0)
        return sampler
