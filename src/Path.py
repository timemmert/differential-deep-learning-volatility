class Path:
    """A single path, containing the inputs, outputs, derivations, and parameters of the simulation."""

    def __init__(self, inputs, outputs, derivatives, params):
        """Constructor.

        Args:
            inputs: Inputs of the simulation, often referred as x or u
            outputs: Outputs of the simulation, often referred as y
            derivatives: Derivatives of outputs in respect to inputs, often referred as dy/dx
            params: Simulation parameters
        """
        self.inputs = inputs
        self.outputs = outputs
        self.derivatives = derivatives
        self.params = params

    def __repr__(self):
        return repr({"inputs": self.inputs, "outputs": self.outputs,
                     "derivatives": self.derivatives, "params": self.params})