from src import Path

from abc import ABC, abstractmethod
from typing import List

import numpy as np

class Model(ABC):
    """Abstract class for sumulation models."""

    @abstractmethod
    def simulate(self, n_steps: int, params: List[float]) -> Path:
        """Generate inputs and simulate current path.

        Args:
            n_steps: Number of steps of the simulation
            params: Simulation parameters

        Return:
            Path containing inputs, outputs, derivatives, and simulation parameters
        """
        pass


class PolynomialAndTrigonometricModel(Model):
    """Example simulation model, mix of polynomial function of 3rd grades and sin function."""

    @staticmethod
    def __generate_inputs(n_steps: int):
        """Randomly uniformly sample inputs between -10 and 10"""
        return np.random.uniform(low=-10, high=10, size=n_steps)

    def simulate(self, n_steps: int, params: List[float]) -> Path:
        a, b, c, d = params

        x = self.__generate_inputs(n_steps)
        y = a * (x ** 3) + b * (x ** 2) + c * np.sin(3 * x) + d
        dydx = a * 3 * (x ** 2) + b * x + c * np.cos(3 * x)

        path = Path(inputs=x.tolist(), outputs=y.tolist(), derivatives=dydx.tolist(), params=params)
        return path
