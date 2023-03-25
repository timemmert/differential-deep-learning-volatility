from abc import ABC
from typing import List

from src import Path, Model, Sampler

class Pricer:
    """Class for generation of training data using model, parameters samples, and option."""

    def __init__(self, model: Model, parameters_sampler: Sampler, option=None):
        """Constructor.

        Args:
            model: Simulation model
            parameters_sampler: Sampler used to sample simulation parameters
            option: Stock option
        """
        self.model = model
        self.parameters_sampler = parameters_sampler
        self.option = option

    def generate_training_data(self, n_paths: int, n_steps: int) -> List[Path]:
        """Generates training data by performing simulations using the model.

        Args:
            n_paths: Number of paths, i.e. parameters configurations
            n_steps: Number of steps for every simulation
        """
        params = self.parameters_sampler.sample(n_paths)
        return [self.model.simulate(n_steps, params_list) for params_list in params]

