# Shaikh, Mohammed Furkhan
# 2020-03-01


import numpy as np

class LinearAssociator(object):
    input_dimensions = 1
    number_of_nodes = 1
    transfer_function="Linear"
    weights = np.random.randn(1,1)
    
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function.lower()
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        else:
            np.random.seed()
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions)

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if(W.shape == (self.number_of_nodes,self.input_dimensions)):
            self.weights = np.copy(W)
        else:
            return -1

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is 'not' included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights
    
    def hard_limit(self, net):
        """
        Activation/transfer function
        """
        if net >= 0 :
            return 1
        return 0

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        out = np.zeros((self.number_of_nodes,X.shape[1]))
        if(self.transfer_function == "linear"):
            out = np.dot(self.weights,X)
        elif(self.transfer_function == "hard_limit"):
            for j in range(X.shape[1]):
                for k in range(self.number_of_nodes):
                    out[k,j] = self.hard_limit(np.dot(self.weights[k],X[:,j].reshape((self.input_dimensions,1)))[0])
        return out

    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        P = np.copy(X)
        Targets = np.copy(y)
#         Pplus = np.dot(np.linalg.inv(np.dot(P.T,P)),P.T) # from book but wrong answer
        Pplus = np.linalg.pinv(P)
        self.weights = np.dot(Targets,Pplus)

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        X = np.copy(X)
        learning = learning.lower()
        
        if learning == "filtered":
            for i in range(num_epochs):
                for j in range(0,X.shape[1],batch_size):
                    if((j+batch_size)<=X.shape[1]):
                        self.weights = (1 - gamma) * self.weights + alpha * np.dot(y[:,j:j+batch_size],X[:,j:j+batch_size].T) 
                    
        elif learning == "delta":
            for i in range(num_epochs):
                for j in range(0,X.shape[1],batch_size):
                    if((j+batch_size)<=X.shape[1]):
                        self.weights += alpha * np.dot((y[:,j:j+batch_size] - self.predict(X[:,j:j+batch_size])),X[:,j:j+batch_size].T)
        elif learning == "unsupervised_hebb":
            for i in range(num_epochs):
                for j in range(0,X.shape[1],batch_size):
                    if((j+batch_size)<=X.shape[1]):
                        self.weights += alpha * np.dot(self.predict(X[:,j:j+batch_size]),X[:,j:j+batch_size].T)
        
    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        predicted = self.predict(X)
        mse = 0
        for i in range(X.shape[1]):
            mse += np.sum((predicted[:,i] - y[:,i])**2)/y.shape[0]
        mse = mse/X.shape[1]
        return mse
