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

def relu(x:ArrayLike)->ArrayLike:
    """Returns 0 if x is less than 0 or equal to 0, else returns 1

    Args:
        x (ArrayLike): numpy array whose relu function is calculated.

    Returns:
        ArrayLike: numpy array with elements 0 or 1.
    """    
    return x * (x > 0)

class Perceptron:
    """Perceptron is the single unit used for neural network.
    In this scenario, we are simulating training of Logical Gates (And, or,X-or, etc.). 
    """    
    def __init__(self, features_size=2, output_size=1, 
               learning_rate=0.01, itreation=100, activation="sigmoid"):
        """Constructor which takes hyperparameters for initialization.

        Args:
            features_size (int, optional): Row Size of the Feature Value(X). Defaults to 2.
            output_size (int, optional): Row Size of the Label Value(y) Defaults to 1.
            learning_rate (float, optional): Rate of Learning by the weights or biases (Ranges from 0 to 1). Defaults to 0.01.
            itreation (int, optional): Number Itreation used for training. Defaults to 100.
            activation (str, optional): Activation function to use for non linearity. Defaults to "sigmoid".
        """        
        self.learning_rate = learning_rate
        self.features = features_size
        self.output = output_size
        self.itreation = itreation
        self.weights, self.bias = self.initialize_params(self.features, self.output)
        self.activation = activation