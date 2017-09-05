import numpy as np
import scipy as sci

'''
Written by Cameron W.J. Abrams 4/13/2017
'''

# A Neural Network class that holds a trained Neural Network.
class NeuralNet:

	# IMAGES is a matrix of n samples points with 784 features, LABELS
	# is an array of n labels for each sample point and WEIGHT_DECAY
	# is an optional paramter used for regularization. 
	def __init__(self, images, labels, weight_decay=None):
		self.images = np.concatenate((images, np.ones((len(images[:,0]), 1))), axis=1)
		self.labels = labels
		self.weight_decay = weight_decay
		self.input_layer_size = 784
		self.hidden_layer_size = 200
		self.output_layer_size = 26
		mu = 0.0
		sigma_v = np.sqrt(1/self.input_layer_size)
		sigma_w = np.sqrt(1/self.hidden_layer_size)
		self.V = np.random.normal(mu, sigma_v, (self.hidden_layer_size, self.input_layer_size + 1))
		self.W = np.random.normal(mu, sigma_w, (self.output_layer_size, self.hidden_layer_size + 1))

	# The log loss function. Z,Y are predicted lables and lables respectively.
	def costFunction(self, z, y):
		for i in range(len(z)):
			if (z[i] == 0):
				z[i] == 10**-12
			elif (z[i] == 1):
				z[i] = 1 - 10**-12
		return -1*(np.dot(y.T, np.log(z)).ravel() + np.dot((1-y).T, np.log(1-z)).ravel())

	# The forward step of the Neural Network. Takes in a sample point, or batch of
	# sample points X and return a hidden layer H and predicted values Z.
	def forward(self, X):
		h = np.tanh(np.dot(self.V, X))
		h = np.insert(h, len(h), values=1, axis=0)
		z = sci.special.expit((np.dot(self.W, h)))
		return h, z

	# The backwards step of the Neural Network, takes in sample point, or batch
	# of sample points X, labels Y, hidden units H, and predicted values Z and
	# returns dJ/DV and dJdW (the gradients of the cost function w.r.t our weights.
	def backward(self, X, y, h, z):
		dJdV = np.dot(np.delete((np.dot(self.W.T, (z-y)) * (1 - h**2)), 200, axis=0), X.T)
		dJdW = np.dot((z-y), h.T)
		return dJdV, dJdW

	# The learn method is the gradient descent step. Takes in DJDV, DJDW,
	# the gradients of the the loss function w.r.t V and W. V_LEARNING_RATE
	#  and W_LEARNING_RATE are the learning rates for V and W respectively. 
	def learn(self, dJdV, dJdW, v_learning_rate, w_learning_rate):
		self.V = self.V - (v_learning_rate*dJdV)
		self.W = self.W - (w_learning_rate*dJdW)

	# Train method. Takes in V_LEARNING_RATE, W_LEARNING_RATE and trains
	# the data using those learning rates performing one full epoch through
	# the data.
	def train(self, v_learning_rate, w_learning_rate, randomized=False):
		for i in range(len(self.images)):
			if (randomized==True):
				j = np.random.randint(len(self.images))
			else:
				j = i
			x = (self.images[j, :]).reshape((len(self.images[0]), 1))
			y = self.labels[j,:].T.reshape(len(self.labels[0]), 1)
			h, z = self.forward(x)
			dJdV, dJdW = self.backward(x, y, h, z)
			self.learn(dJdV, dJdW, v_learning_rate=v_learning_rate, w_learning_rate=w_learning_rate)

	# TrainMini method. Takes in BATCH_SIZE, V_LEARNING_RATE, and W_LEARNING_RATE 
	# training the data using those learning rates performing one full epoch
	# through the data.
	def trainMini(self, batch_size, v_learning_rate, w_learning_rate):
		splits = len(self.images)//batch_size
		X = (self.images[:batch_size,:]).T
		Y = (self.labels[:batch_size,:]).T
		h,z = self.forward(X)
		dJdV, dJdW = self.backward(X, Y, h, z)
		self.learn((1/batch_size)*dJdV, (1/batch_size)*dJdW, v_learning_rate=v_learning_rate,
			w_learning_rate=w_learning_rate)
		for i in range(2,splits):
			X = (self.images[batch_size*(i-1):batch_size*i,:]).T
			Y = (self.labels[batch_size*(i-1):batch_size*i,:]).T
			h,z = self.forward(X)
			dJdV, dJdW = self.backward(X, Y, h, z)
			self.learn((1/batch_size)*dJdV, (1/batch_size)*dJdW, v_learning_rate=v_learning_rate,
				w_learning_rate=w_learning_rate)			
		X = (self.images[batch_size*splits:,:]).T
		Y = (self.labels[batch_size*splits:,:]).T
		h,z = self.forward(X)
		dJdV, dJdW = self.backward(X, Y, h, z)
		self.learn(dJdV*(1/batch_size), dJdW*(1/batch_size), v_learning_rate=v_learning_rate,
			w_learning_rate=w_learning_rate)

	# Same as the train method but returns two arrays, x_plot and y_plot where
	# the x_plot is the number of iterations and y_plot are the values of the
	# cost function at those iterations using the labels obtained by the Neural
	# Network at those points.
	def trainPlot(self, v_learning_rate, w_learning_rate, randomized=False):
		x_plot = list()
		y_plot = list()
		for i in range(len(self.images)):
			if (randomized==True):
				j = np.random.randint(len(self.images))
			else:
				j = i
			x = (self.images[j, :]).reshape((len(self.images[0]), 1))
			y = self.labels[j,:].T.reshape(len(self.labels[0]), 1)
			h, z = self.forward(x)
			dJdV, dJdW = self.backward(x, y, h, z)
			if (i % 100 == 0 and i != 0):
				x_plot.append(i)
				y_plot.append(self.costFunction(z, y))
			self.learn(dJdV, dJdW, v_learning_rate=v_learning_rate, w_learning_rate=w_learning_rate)
		return x_plot, y_plot

	# The classify method takes in a sample point X and returns a class 1-26.
	def classify(self, x):
		if (len(x) != len(self.V[0,:])):
			x = np.insert(x, len(x), values=1, axis=0)
		h = np.append(np.tanh(np.dot(self.V, x.T)), 1)
		z = sci.special.expit(np.dot(self.W, h))
		return np.argmax(z) + 1

	# The classifyAll method takes a matrix of images _IMAGES and returns
	# an list of classes 1-26 for each images.
	def classifyAll(self, _images):
		ret_arr = list()
		for i in range(len(_images)):
			ret_arr.append(self.classify(_images[i]))
		return ret_arr

	# Updates the Neural Networks images and labels. Used during training to
	# so that the training data can be reshuffled randomly and be fed back
	# to the Neural Network.
	def updateData(self, images, labels):
		self.images = np.concatenate((images, np.ones((len(images[:,0]), 1))), axis=1)
		self.labels = labels

