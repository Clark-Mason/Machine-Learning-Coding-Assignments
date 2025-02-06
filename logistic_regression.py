# logistic_regression.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Nikhilesh Prabhakar (nikhilesh.prabhakar@utdallas.edu),
# Athresh Karanam (athresh.karanam@utdallas.edu),
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing a simple version of the 
# Logistic Regression algorithm. Insert your code into the various functions 
# that have the comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. 


import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle 


class SimpleLogisiticRegression():
    """
    A simple Logisitc Regression Model which uses a fixed learning rate
    and Gradient Ascent to update the model weights
    """
    def __init__(self):
        self.w = []
        pass

        
    def initialize_weights(self, num_features):
        #DO NOT MODIFY THIS FUNCTION
        w = np.zeros((num_features))
        return w

    def compute_loss(self,  X, y):
        """
        Compute binary cross-entropy loss for given model weights, features, and label.
        :param w: model weights
        :param X: features
        :param y: label
        :return: loss   
        """
        # INSERT YOUR CODE HERE

        raise Exception('Function not yet implemented!')

    
    def sigmoid(self, val):

        """
        Implement sigmoid function
        :param val: Input value (float or np.array)
        :return: sigmoid(Input value)
        """
        # INSERT YOUR CODE HERE
        raise Exception('Function not yet implemented!')


    def gradient_ascent(self, w, X, y, lr):
        # slide 14 bottom

        """
        Perform one step of gradient ascent to update current model weights. 
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        Update the model weights
        """
        # INSERT YOUR CODE HERE
        raise Exception('Function not yet implemented!')



    def fit(self,X, y, lr=0.1, iters=100, recompute=True):
        """
        Main training loop that takes initial model weights and updates them using gradient descent
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        :param recompute: Used to reinitialize weights to 0s. If false, it uses the existing weights Default True

        NOTE: Since we are using a single weight vector for gradient ascent and not using 
        a bias term we would need to append a column of 1's to the train set (X)

        """
        # INSERT YOUR CODE HERE
        
        if(recompute):
            #Reinitialize the model weights
            pass

        for _ in range(iters):
            # INSERT YOUR CODE HERE
            pass


    def predict_example(self, w, x):
        """
        Predicts the classification label for a single example x using the sigmoid function and model weights for a binary class example
        :param w: model weights
        :param x: example to predict
        :return: predicted label for x
        """
         # INSERT YOUR CODE HERE
        raise Exception('Function not yet implemented!')



    def compute_error(y_true, y_pred):
        """
        Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
        :param y_true: true label
        :param y_pred: predicted label
        :return: error rate = (1/n) * sum(y_true!=y_pred)
        """
        # INSERT YOUR CODE HERE
        raise Exception('Function not yet implemented!')




if __name__ == '__main__':

    # Load the training data
    M = np.genfromtxt('./data/monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    lr =  SimpleLogisiticRegression()
    
    #Part 1) Compute Train and Test Errors for different number of iterations and learning rates
    for iter in [10, 100,1000,10000]:
        for a in [0.01,0.1, 0.33]:
            #INSERT CODE HERE
            pass

    #Part 2) Retrain Logistic Regression on the best parameters and store the model as a pickle file
    #INSERT CODE HERE

    # Code to store as pickle file
    netid = 'AXE210038'
    file_pi = open('{}_model_1.obj',format(netid), 'wb')  #Use your NETID
    pickle.dump(lr, file_pi)


    #Part 3) Compare your model's performance to scikit-learn's LR model's default parameters 
    #INSERT CODE HERE

    #Part 4) Plot curves on train and test loss for different learning rates. Using recompute=False might help
    for a in [0.01,0.1, 0.33]:
        lr.fit(Xtrn, ytrn, lr=a, iters=1)
        #INSERT CODE HERE
        for i in range(10):
            lr.fit(Xtrn, ytrn, lr=a, iters=100,recompute=False)
            #INSERT CODE HERE


