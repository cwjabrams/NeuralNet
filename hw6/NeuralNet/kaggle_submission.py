from NeuralNetwork import NeuralNet
import scipy.io as sio
import numpy as np
import sklearn.preprocessing as skp

'''
Written by Cameron W.J. Abrams 4/13/2017
'''

images = np.load('data/images.npy')
labels = np.load('data/vec_labels.npy')
data = np.concatenate((images, labels), axis=1)
np.random.shuffle(data)

train_images = data[:, :len(images[0])]
train_labels = data[:, len(images[0]):]

scaler = skp.StandardScaler()
train_images = scaler.fit_transform(train_images)
test_data = np.load('data/test_data.npy')
test_data = scaler.transform(test_data)

# Set target values in our labels matrix(e.g. to 0.15 and 0.85).
for i in range(len(train_labels[:,0])):
	for j in range(len(train_labels[0,:])):
		if train_labels[i,j] < 0.5:
			train_labels[i,j] = .05
		else:
			train_labels[i,j] = .95

NN = NeuralNet(train_images, train_labels)
_v_learning_rate = 0.1
_w_learning_rate = 0.01
for i in range(12):
	if (i > 0):
		data = np.concatenate((train_images, train_labels), axis=1)
		train_images = data[:, :len(images[0])]
		train_labels = data[:, len(images[0]):]
		NN.updateData(train_images, train_labels)
		if ((i % 6) == 0):
			_v_learning_rate = 0.9*_v_learning_rate
			_w_learning_rate = 0.6*_w_learning_rate
	NN.trainMini(batch_size=25,v_learning_rate=_v_learning_rate,
		w_learning_rate=_w_learning_rate)

f = open('kaggle_submission.csv', 'w')
header = 'Id,Category\n'
f.write(header)

predictions = NN.classifyAll(test_data)
for i in range(len(test_data)):
	predicted_class = predictions[i]
	S = str(i + 1) + ',' + str(predicted_class) + '\n'
	f.write(S)
f.close()
