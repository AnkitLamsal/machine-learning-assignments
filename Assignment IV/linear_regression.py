from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np


class LinearRegression:
    """
    Class Based Implementation of Linear Regression Machine Model using standalone 
    numpy library. 
    Methods in this Class:
    1. initalize_params
    2. print_params
    3. predict(X)
    4. mse(y_true, y_predicted)
    5. calculate_delta
    6. update_params
    7. predict
    """    
    def __init__(self,alpha,itreation):
        """Constructor of the Class

        Args:
            alpha (_type_): _description_
            itreation (_type_): _description_
        """        
        self.alpha = alpha
        self.itreation = itreation
        self.weight, self.bias = self.initialize_params()

    def initialize_params(self)->Tuple:
        """Initalizes Parameters weights and bias for linear Regression
        Weight is initialized randomly.
        Whereas Bias is inialized with zero.
        Returns:
            Tuple: Two elements weights and bias initalized randomly
        """        
        weight = np.random.randn()
        bias = 0
        return weight, bias

    def print_params(self):
        """Prints the parameters(weights and bias).
        """        
        print(self.weight, self.bias)

    def predict(self,X:ArrayLike)->ArrayLike:
        """Predicts the Output labels based on input features.

        Args:
            X (ArrayLike): Input features 

        Returns:
            ArrayLike: Predicted Output
        """        
        y_pred  = np.dot(self.weight, X) + self.bias
        return y_pred

    def mse(self,y:ArrayLike, y_pred:ArrayLike)->int:
        """Calculates and returns the mean squared error value between y_true and y_predicted.

        Args:
            y (ArrayLike): True y label
            y_pred (ArrayLike): Predicted y

        Returns:
            int: mean squared loss
        """        
        self.m = y_pred.shape[0]
        mse = np.mean(np.sum((y_pred - y)**2))
        return mse

    def calculate_delta(self):
        """Calculates the d(J(bias,weight))/d(bias) and d(J(bias,weight))/d(weight).
        """        
        self.delta_weight = (1/self.m)*(np.sum(y_pred - y))*x
        self.delta_bias = (1/self.m)*(np.sum(y_pred - y))

    def update_params(self):
        """Updates the Weights and bias based on learning rate and delta_weight, delta_bias
        """        
        self.weight -= self.alpha*self.delta_weight
        self.bias -= self.alpha*self.delta_bias

    def fit(self,X:ArrayLike,y:ArrayLike):
        """This function trains the model based on input features and their labels.

        Args:
            X (ArrayLike): Input features
            y (ArrayLike): Output Labels.
        """        
        for i in range(self.itreation):
            y_pred = self.predict(X)
            mse = self.mse(y,y_pred)
            print("Loss of given itreation is :"+mse)
            self.update_params()
            self.print_params()
            print(i+1+"Itreation Completed.")