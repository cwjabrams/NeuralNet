Cameron Abrams
26487031

Included:
	this README
	NeuralNetwork.py
	data
	trainer.py
	practice.py
	batch_practice.py
	batch_trainer.py
	cost_plot.py
	kaggle_submission.py
	kaggle_submission.csv
	visualize.py
	images


Anaconda or having the NumPy and SciPy libraries is sufficient to run all of the following .py files.

=========
ATTENTION: Although these libraries are necessary for this code, the code itself was written using python3 and may not run as expected on an older version of python.
=========

-----------------
NeuralNetwork.py
----------------- 

The source code for the NeuralNet class. Attributes and methods described in code.

-----------------
data
----------------- 

A directory containing the training and test data converted to .npy files for quick loading. 

-----------------
trainer.py
-----------------

A script with a main method which creates a NeuralNet class and trains on 80% of the training data using stochastic gradient descent given certain hyper-parameters, described in code/write-up. The script will print the settings being used, and at each epoch prints the epoch id, the training error rate and the validation error rate. >>>python3 trainer.py will run the main method with preset hyper parameters.

----------------
practice.py
----------------

Imports the main method from trainer.py and can be used to easily test multiple hyper-parameters.

----------------
batch_trainer.py
----------------

A script with a main method which creates a NeuralNet class and trains on 80% of the training data using mini-batch gradient descent given certain hyper-parameters, described in code/write-up. The script will print the settings being used, and at each epoch prints the epoch id, the training error rate and the validation error rate. >>>python3 batch_trainer.py will run the main method with preset hyper parameters.

----------------
batch_practice.py
----------------

Imports the main method from batch_trainer.py and can be used to easily test multiple hyper-parameters.

----------------
cost_plot.py
----------------

A script which creates a NeuralNet class and trains on 80% of the training data using stochastic gradient descent. Outputs an image file ‘images/Cost_Function.png’ that is a graphical representation of the the cost function vs the number of iterations, plotted at every 100th sample point. 

----------------
kaggle_submission.py
----------------

A script which creates a NeuralNet class and trains on all of the training data using mini-batch gradient descent. Outputs a file ‘kaggle_submission.csv’ which contains ID,CATEGORY pairs for all of the test data.

----------------
kaggle_submission.csv
----------------

See above.

----------------
visualize.py
----------------

A script with a main method which creates a NeuralNet class and trains on 80% of the training data using mini-batch gradient descent. Then, by visualizing 10 sample points from the validation set, the script outputs 5 image files from the set of sample points classified correctly and 5 image files from the set of sample points classified incorrectly. The images are places in the images folder described below.

----------------
images
----------------

A directory containing all the images output by visualize.py and cost_plot.py



