# Neural-Networks-with-Pure-NumPy

This repository contains three Jupyter Notebook files: part-1.ipynb, part-2.ipynb, and part-3.ipynb, implementing various models using pure NumPy.

Part-1
Description:
The part-1.ipynb notebook implements an autoencoder using a fully connected neural network. It focuses on unsupervised feature extraction from natural images. The notebook includes preprocessing steps such as converting images to grayscale, normalization, and visualization of sample patches. The autoencoder is trained using backpropagation and gradient descent.

Requirements:
To run the code in part-1.ipynb, the following dependencies are required:

-NumPy

-h5py

-matplotlib

-scipy

Instructions:

Open the part-1.ipynb notebook using Jupyter Notebook or JupyterLab.
Make sure the required dependencies are installed.
Execute the code cells in sequential order to load the data, preprocess it, and train the autoencoder.
The notebook provides detailed explanations and comments to guide you through the code.

Part-2

Description:
The part-2.ipynb notebook  focuses on text classification using a trigram-based approach. It demonstrates how to load a dataset of text samples and their labels, preprocess the text data, and train a neural network for classification. The notebook uses backpropagation and gradient descent for training.

Requirements:
To run the code in part-2.ipynb, the following dependencies are required:

-NumPy

-h5py

-scikit-learn (OneHotEncoder)

Instructions:

Open the part-2.ipynb notebook using Jupyter Notebook or JupyterLab.
Make sure the required dependencies are installed.
Execute the code cells in sequential order to load the data, preprocess it, and train the neural network.
The notebook includes detailed explanations and comments to guide you through the code.

Part 3

Description:
The part-3.ipynb notebook compares the performance of three types of recurrent neural network (RNN) architectures for classifying human activity from movement signals measured with three sensors simultaneously. The three architectures compared are RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU). The notebook utilizes backpropagation through time for training and a multi-layer perceptron with a softmax function for classification. The notebook provides options to adjust the parameters and number of hidden layers for improved performance.

Requirements:
To run the code in part-3.ipynb, the following dependencies are required:

-NumPy

-h5py

-scikit-learn (train_test_split, confusion_matrix, accuracy_score functions)

Instructions:

Open the part-3.ipynb notebook using Jupyter Notebook or JupyterLab.
Make sure the required dependencies are installed.
Execute the code cells to compare the performance of RNN, LSTM, and GRU architectures for human activity classification.
The notebook includes detailed explanations and comments to guide you through the code.
Please note that each notebook provides specific instructions within the code cells to guide you through the implementation and execution of the models.
