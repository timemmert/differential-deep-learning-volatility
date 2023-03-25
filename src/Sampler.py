from typing import List, Sequence
from abc import ABC, abstractmethod

import numpy as np

class Sampler(ABC):
    """Abstract class for sampling simulation parameters."""

    @abstractmethod
    def sample(self, n_paths: int) -> List[List[float]]:
        """Samples n_paths simulation parameters.

        Args:
            n_paths: Number of paths, i.e. simulation parameters configurations.

        Returns:
            List containing simulation parameters configurations in the list format
        """
        pass


class UniformSampler(Sampler):
    """Class for sampling simulation parameters using uniform random distribution."""
    def __init__(self, low: float | Sequence[float], high: float | Sequence[float]):
        """Constructor.

        Args:
            low: The minimum parameters value, can be a list for multidimensional configurations or a single scalar
            high: The maximum parameters value, can be a list for multidimensional configurations or a single scalar
            """
        self.low = low
        self.high = high

    def sample(self, n_paths: int):
        return [np.random.uniform(low=self.low, high=self.high) for _ in range(n_paths)]