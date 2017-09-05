from NeuralNetwork import NeuralNet
import scipy.io as sio
import numpy as np
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt

'''
Written by Cameron W.J. Abrams 4/13/2017
'''

images = np.load('data/images.npy')
labels = np.load('data/vec_labels.npy')
data = np.concatenate((images, labels), axis=1)
np.random.shuffle(data)


train_images = data[:round(.8*len(data)), :len(images[0])]
train_labels = data[:round(.8*len(data)), len(images[0]):]
validation_images = data[round(.8*len(data)):, :len(images[0])]
validation_labels = data[round(.8*len(data)):, len(images[0]):]

scaler = skp.StandardScaler()
train_images = scaler.fit_transform(train_images)
validation_images = scaler.transform(validation_images)

# Set target values in our labels matrix to 0.15 and 0.85
for i in range(len(train_labels[:,0])):
	for j in range(len(train_labels[0,:])):
		if train_labels[i,j] < 0.5:
			train_labels[i,j] = 0.1
		else:
			train_labels[i,j] = 0.9

# CREATE NeuralNet Object
NN = NeuralNet(train_images, train_labels)

# TRAIN NeuralNet Object
for i in range(1):
	x_plot, y_plot = NN.trainPlot(v_learning_rate=0.01, w_learning_rate=0.001)
	# Plot of cost function vs number of iterations.
	plt.plot(x_plot, y_plot, 'r-')
	plt.xlabel('Number of Iterations')
	plt.ylabel('J(y, z; x, V, W)')
	plt.title('Cost Function vs. Number of Iterations')
	plt.savefig('Cost_Function.png', bbox_inches='tight')

# TRAINING ACCURACY
y_hat = NN.classifyAll(train_images)
test_correct = 0
test_size = len(NN.images)
for i in range(test_size):
	if (y_hat[i] == np.argmax(NN.labels[i]) + 1):
		test_correct += 1
	else:
		continue
print('\nTest Classfication Complete:')
print('Test Set Error: ', 1 - (test_correct / test_size))

# VALIDATION ACCURACY
total_correct = 0
validation_size = len(validation_images)
z = NN.classifyAll(validation_images)
for i in range(len(z)):
	if (z[i] == np.argmax(validation_labels[i]) + 1):
		total_correct += 1
	else:
		continue
print('\nValidation Classification Complete:')
print('Validation Error Rate: ', 1 - (total_correct/validation_size), '\n')
