import typing
import numpy as np

class Array1D():
    """An alias for a 1D numpy array."""
    def __init__(self,
                 ModelVector: np.typing.NDArray # A 1D numpy array suitable for stacks of scalar parameter values
                ):
        pass

class Array2D():
    """An alias for a 2D numpy array."""
    def __init__(self,
                Model2DArray: np.typing.NDArray, # A 2D numpy array suitable for stacks of state vectors
                ):
        pass

class Array3D():
    """An alias for 3D numpy array, last two dimensions of equal size."""
    def __init__(self,
                Model3DArray: np.typing.NDArray # A 3D numpy array suitable for stacks of payoff or transition matrices 
                ):
        pass

class ModelTypeDSAIR():
    """This is the schema for the inputs to a DSAIR model.
    
    Note: This schema is not enforced and is here purely for documentation
    purposes."""
    def __init__(self, 
                 b: Array1D, # benefit: The size of the per round benefit of leading the AI development race, b>0
                 c: Array1D, # cost: The cost of implementing safety recommendations per round, c>0
                 s: Array1D, # speed: The speed advantage from choosing to ignore safety recommendations, s>1
                 p: Array1D, # avoid_risk: The probability that unsafe firms avoid an AI disaster, p ∈ [0, 1]
                 B: Array1D, # prize: The size of the prize from winning the AI development race, B>>b
                 W: Array1D, # timeline: The anticipated timeline until the development race has a winner if everyone behaves safely, W ∈ [10, 10**6]
                 pfo: Array1D=None, # detection risk: The probability that firms who ignore safety precautions are found out, pfo ∈ [0, 1]
                 α: Array1D=None, # the cost of rewarding/punishing a peer
                 γ: Array1D=None, # the effect of a reward/punishment on a developer's speed
                 epsilon: Array1D=None, # commitment_cost: The cost of setting up and maintaining a voluntary commitment, ϵ > 0
                 ω: Array1D=None, # noise: Noise in arranging an agreement, with some probability they fail to succeed in making an agreement, ω ∈ [0, 1]
                 collective_risk: Array1D=None, # The likelihood that a disaster affects all actors
                ):
        pass
