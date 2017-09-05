from NeuralNetwork import NeuralNet
import numpy as np
import sklearn.preprocessing as skp

'''
Written by Cameron W.J. Abrams 4/13/2017
'''

def main(batch_size = 25, target_values=[0.1,0.9], epochs=7,
	v_learning_rate=0.1, w_learning_rate=0.01, v_decay_rate=0.9, w_decay_rate=0.6,
	decay_frequency=4):
	_v_learning_rate = v_learning_rate
	_w_learning_rate = w_learning_rate
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
	

	# Set target values in our labels matrix(e.g. to 0.15 and 0.85).
	if target_values is not None:
		for i in range(len(train_labels[:,0])):
			for j in range(len(train_labels[0,:])):
				if train_labels[i,j] < 0.5:
					train_labels[i,j] = target_values[0]
				else:
					train_labels[i,j] = target_values[1]

	print('\n============\n  SETTINGS\n============')
	print('Batch Size: ', batch_size)
	print('Target Values: ', target_values)
	print('Number of Epochs: ', epochs)
	print('V Learning Rate: ', v_learning_rate)
	print('W Learning Rate: ', w_learning_rate)
	print('V Decay Rate: ', v_decay_rate)
	print('W Decay Rate: ', w_decay_rate)
	print('Decay Frequency: ', decay_frequency)

	NN = NeuralNet(train_images, train_labels)

	for i in range(epochs):
		if (i > 0):
			data = np.concatenate((train_images, train_labels), axis=1)
			np.random.shuffle(data)
			train_images = data[:, :len(images[0])]
			train_labels = data[:, len(images[0]):]
			NN.updateData(train_images, train_labels)
			if ((i % decay_frequency) == 0):
				_v_learning_rate = v_decay_rate*_v_learning_rate
				_w_learning_rate = w_decay_rate*_w_learning_rate

		NN.trainMini(batch_size=batch_size,v_learning_rate=_v_learning_rate,
			w_learning_rate=_w_learning_rate)

		print('\n============\n  EPOCH:', i, '\n============')

		# TRAINING ACCURACY
		y_hat = NN.classifyAll(train_images)
		test_correct = 0
		test_size = len(NN.images)
		for j in range(test_size):
			if (y_hat[j] == np.argmax(NN.labels[j]) + 1):
				test_correct += 1
			else:
				continue
		print('\nTest Classfication Complete:')
		print('Test Set Error: ', 1 - (test_correct / test_size))

		# VALIDATION ACCURACY
		total_correct = 0
		validation_size = len(validation_images)
		z = NN.classifyAll(validation_images)
		for j in range(len(z)):
			if (z[j] == np.argmax(validation_labels[j]) + 1):
				total_correct += 1
			else:
				continue
		print('\nValidation Classification Complete:')
		print('Validation Error Rate: ', 1 - (total_correct/validation_size), '\n')

if __name__ == '__main__':
	main()
