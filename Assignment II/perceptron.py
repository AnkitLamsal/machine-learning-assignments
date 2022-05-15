from typing import Tuple, List
from numpy.typing import ArrayLike
import numpy as np

def sigmoid(z:ArrayLike)-> ArrayLike:
    """Sigmoid Function Calculates 1/(1 + e^(-x)) for given value x.
       Output value Ranges between 0 to 1.

    Args:
        z (ArrayLike): Numpy array whose Sigmoid value is calculated.

    Returns:
         ArrayLike: Calculated value i.e. 1/(1 + e^(-x))
    """    
    return (1/(1+np.exp(-z)))
