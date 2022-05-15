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
        
    def initialize_params(self,feature_size:int, output_size:int)->Tuple:
        """This function initailizes weight and bias parameter for training based on
        X(input/feature size) and Y(output_size/ label size)

        Args:
            feature_size (int): Size of X
            output_size (int): Size of labels.

        Returns:
            Tuple:  Initialized weights and biases.
        """                        
        weight = np.random.randn(output_size,feature_size)
        bias = np.zeros((output_size,1))
        return (weight, bias)

    def forward_pass(self,x:ArrayLike)->ArrayLike:
        """This function takes the forward propagation stage of neural network.
        First Weighted Sum is calculated and Activation function is calculated 
        user given activation function. 
        Args:
            x (ArrayLike): =Numpy Array which are either 0 or 1 in form. Input data with multiple features.

        Returns:
            ArrayLike: Numpy array of predicted output. All elements are either 0 or 1.
        """        
        dot_product = np.dot(self.weights,x) + self.bias
        if(self.activation == "sigmoid"):
            predicted_y = sigmoid(dot_product)
        elif(self.activation == "relu"):
            predicted_y = relu(dot_product)
        elif(self.activation == "tanh"):
            predicted_y = np.tanh(dot_product)
        return predicted_y
    
    def calculate_delta(self,x:ArrayLike,y:ArrayLike,predicted_y:ArrayLike):
        """This Function Calculates difference of the weight.Inorder to adjust the parameters.
        It also assigns into the Object's element.
        Args:
            x (ArrayLike): Input Set
            y (ArrayLike): Labels / Outputs
            predicted_y (ArrayLike): Predicted Output from forward pass.
        """        
        m = x.shape[1]
        difference = y -predicted_y
        gradient_bias = np.sum(difference, axis=1,keepdims=True)
        gradient_weights = np.dot(difference,x.T)/m
        self.delta_weights = gradient_weights
        self.delta_bias = gradient_bias
        
    def update_params(self):
        """This function updates the weight and bias parameters.
        Calculates adjusted weight and adjusted bias as : 
                    Weight(new) = Weight(old) + learning_rate*(gradient_weight)
                    Bias(new) = Bias(old) + learning_rate*(gradient_bias)
        """        
        self.weights = self.weights + self.learning_rate*self.delta_weights
        self.bias = self.bias+self.learning_rate*self.delta_bias