import sys
import os
import numpy as np
import pandas as pd

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 90

def ReLu(x):
	return max(0,x)

def signum(x):
	if x < 0:
		return 0
	else:
		return 1

class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.

		Here, we have provided an example structure for the weights and biases.
		It is a list of weight and bias matrices, in which, the
		dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
		weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
		biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

		Please note that this is just an example.
		You are free to modify or entirely ignore this initialization as per your need.
		Also you can add more state-tracking variables that might be useful to compute
		the gradients efficiently.


		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		'''
		self.num_layers = num_layers
		self.num_units = num_units

		self.biases = []
		self.weights = []
		for i in range(num_layers):

			if i==0:
				# Input layer
				self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
			else:
				# Hidden layer
				self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

			self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

		# Output layer
		self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
		self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))
		

	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.

		Note that for a classification task, the output layer should
		be a softmax layer. So perform the computations accordingly

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		'''
		#ReLu activation Function
		relu = np.vectorize(ReLu)

		self.activations = []
		self.hypothesis = []
		#Calculating activations for all layers except output layer
		A = X
		self.activations.append(X)
		self.hypothesis.append(X)
		for i in range(self.num_layers):
			A = np.dot(A,self.weights[i]) + self.biases[i].T
			self.hypothesis.append(A)
			A = relu(A)
			self.activations.append(A)
		

		#output layer
		output = np.dot(A,self.weights[-1]) + self.biases[-1].T

		self.hypothesis.append(output)
		ouptut = relu(output)

		self.activations.append(output)
		return output
		# raise NotImplementedError

	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing backward pass.
		'''
		del_W = []
		del_B = []
		m = y.shape[0]
		signumfun = np.vectorize(signum)
		previous_del = [(self.activations[-1]-y)/(m/2)]

		for i in reversed(range(self.num_layers)):
			dgz = signumfun(self.activations[i+1])
			print(np.outer(self.activations[i],dgz).shape)
			# del_W_current = 
		return self.weights, self.biases
		# rarise NotImplementedError


class Optimizer(object):
	'''
	'''

	def __init__(self, learning_rate):
		'''
		Create a Gradient Descent based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		'''

		self.learning_rate = learning_rate
		# raise NotImplementedError

	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		'''
		updated_weights = []
		updated_biases = []

		for i in range(len(weights)):
			updated_weights.append(weights[i] - np.dot(self.learning_rate,delta_weights[i]))
			updated_biases.append(biases[i] - np.dot(self.learning_rate,delta_biases[i]))

		return updated_weights,updated_biases
		# raise NotImplementedError


def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.
	'''

	m = y.shape[0]

	mse_loss = np.sum(np.square(y-y_hat))/m

	return mse_loss
	# raise NotImplementedError

def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss 
	'''

	regularization_loss = 0.0
	
	for i in range(len(weights)):
		regularization_loss +=np.sum(np.square(weights[i])) + np.sum(np.square(biases[i]))

	return regularization_loss
	# raise NotImplementedError

def loss_fn(y, y_hat, weights, biases, lamda):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss 
	'''

	mse_loss = loss_mse(y,y_hat)
	regularization_loss = loss_regularization(weights,biases)

	loss = mse_loss + lamda*regularization_loss

	return loss
	
	# raise NotImplementedError

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	'''

	m = y.shape[0]

	mse_loss = np.sqrt(np.sum(np.square(y-y_hat)))/m

	return mse_loss
	# raise NotImplementedError

def cross_entropy_loss(y, y_hat):
	'''
	Compute cross entropy loss

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		cross entropy loss
	'''
	raise NotImplementedError

def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
	'''
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each bach of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.

	Here we have added the code to loop over batches and perform backward pass
	for each batch in the loop.
	For this code also, you are free to heavily modify it.
	'''

	m = train_input.shape[0]

	for e in range(max_epochs):
		epoch_loss = 0.
		for i in range(0, m, batch_size):
			batch_input = train_input[i:i+batch_size]
			batch_target = train_target[i:i+batch_size]
			pred = net(batch_input)

			# Compute gradients of loss w.r.t. weights and biases
			dW, db = net.backward(batch_input, batch_target, lamda)

			# Get updated weights based on current weights and gradients
			weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

			# Update model's weights and biases
			net.weights = weights_updated
			net.biases = biases_updated

			# Compute loss for the batch
			batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
			epoch_loss += batch_loss

			#print(e, i, rmse(batch_target, pred), batch_loss)

		print(e, epoch_loss)

		# Write any early stopping conditions required (only for Part 2)
		# Hint: You can also compute dev_rmse here and use it in the early
		# 		stopping condition.

	# After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
	dev_pred = net(dev_input)
	dev_rmse = rmse(dev_target, dev_pred)

	print('RMSE on dev data: {:.5f}'.format(dev_rmse))


def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''


	raise NotImplementedError

def read_data():
	'''
	Read the train, dev, and test datasets
	'''

	train_data_path = "regression\\data\\train.csv"
	dev_data_path = "regression\\data\\dev.csv"
	test_data_path = "regression\\data\\test.csv"

	train_data = pd.read_csv(train_data_path)
	dev_data = pd.read_csv(dev_data_path)
	test_input = pd.read_csv(test_data_path)

	train_input = train_data.drop("1",axis = 1)
	train_target = train_data["1"]

	dev_input = dev_data.drop("1",axis = 1)
	dev_target = dev_data["1"]

	# raise NotImplementedError

	return np.asarray(train_input), np.asarray(train_target), np.asarray(dev_input), np.asarray(dev_target), np.asarray(test_input)


def main():

	# Hyper-parameters 
	max_epochs = 50
	batch_size = 256
	learning_rate = 0.001
	num_layers = 1
	num_units = 64
	lamda = 0.1 # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	# get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()
