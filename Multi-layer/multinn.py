# Shaikh, Mohammed Furkhan
# 2020-03-22

# %tensorflow_version 2.x
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np

class MultiNN(object):

    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimensions = input_dimension
        self.weights = list()
        self.bias = list()
        self.layers_activation = list()
        self.no_of_nodes= list()

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        if(len(self.no_of_nodes)==0):    
#             self.weights.append(tf.Variable(tf.random.normal([self.input_dimensions,num_nodes]),trainable=True))
            self.weights.append(tf.Variable(np.random.randn(self.input_dimensions,num_nodes),name="weights"+str(len(self.no_of_nodes)),trainable=True))
        else:
#             self.weights.append(tf.Variable(tf.random.normal([self.no_of_nodes[-1],num_nodes]),trainable=True))
            self.weights.append(tf.Variable(np.random.randn(self.no_of_nodes[-1],num_nodes),name="weights"+str(len(self.no_of_nodes)),trainable=True))
#         self.bias.append(tf.Variable(tf.random.normal([1,num_nodes]),trainable=True))
        self.bias.append(tf.Variable(np.random.randn(1,num_nodes),name="bias"+str(len(self.no_of_nodes)),trainable=True))
        self.layers_activation.append(transfer_function.lower())
        self.no_of_nodes.append(num_nodes)
        return
    
    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.weights[layer_number].numpy()

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.bias[layer_number].numpy()

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        if(weights.shape == self.weights[layer_number].numpy().shape):
            self.weights[layer_number].assign(weights) 
        else:
            return -1

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        if(biases.shape == self.bias[layer_number].numpy().shape):
            self.bias[layer_number].assign(biases) 
        else:
            return -1

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y,y_hat))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        Y = tf.Variable(X,trainable=True)
        for i in range(len(self.no_of_nodes)):
            if(self.layers_activation[i]=="sigmoid"):
                Y = tf.nn.sigmoid(tf.matmul(Y,self.weights[i]) + self.bias[i])
            elif(self.layers_activation[i]=="relu"):
                Y = tf.nn.relu(tf.matmul(Y,self.weights[i]) + self.bias[i])
            else:
                Y = tf.matmul(Y,self.weights[i]) + self.bias[i]
        return Y

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        for i in range(num_epochs):
            for j in range(0,X_train.shape[0],batch_size):
                if((j+batch_size)<=X_train.shape[0]):
                        with tf.GradientTape(persistent = True) as tape:
                            y_pred = self.predict(X_train[j:j+batch_size,:])
                            loss = self.calculate_loss(y_train[j:j+batch_size],y_pred)
                        for k in range(len(self.no_of_nodes)-1,-1,-1):
                            dloss_dw, dloss_db = tape.gradient(loss, [self.weights[k], self.bias[k]])
                            self.weights[k].assign_sub(alpha * dloss_dw)
                            self.bias[k].assign_sub(alpha * dloss_db)                

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        y_pred_mat = self.predict(X).numpy()
        y_pred = np.zeros((y_pred_mat.shape[0],1),dtype='int32')
        err=0
        for i in range(y_pred_mat.shape[0]):
#             y_pred[i] = np.where(y_pred_mat[i] == np.max(y_pred_mat[i]))
            y_pred[i] = np.argmax(y_pred_mat[i])
            if(y[i]!=y_pred[i][0]):
                err+=1
        return err/X.shape[0]

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        y_pred_mat = self.predict(X).numpy()
        y_pred = np.zeros((y_pred_mat.shape[0],1),dtype='int32')
        for i in range(y_pred_mat.shape[0]):
#             y_pred[i] = np.where(y_pred_mat[i] == np.max(y_pred_mat[i]))
            y_pred[i] = np.argmax(y_pred_mat[i])
        return tf.math.confusion_matrix(y,y_pred,num_classes=self.no_of_nodes[-1])

