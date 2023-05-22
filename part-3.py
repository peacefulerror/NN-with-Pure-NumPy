#!/usr/bin/env python
# coding: utf-8

# Question-3

# To compare the performance of three types of recurrent neural network architectures for classifying human activity from movement signals measured with three sensors simultaneously: Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU). Back propagation through time is used to train the network and a multi-layer perceptron with a softmax function is used for classification. I ran the network with two hidden dimension settings, 128 and 32 neurons in the hidden layer.

# a)

# In[23]:


# 3-a
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Xavier initialization
        self.wh = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.wx = np.random.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
        self.wy = np.random.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))
        # Momentum initializations
        self.m_wx, self.m_wh, self.m_wy = np.zeros_like(self.wx), np.zeros_like(self.wh), np.zeros_like(self.wy)
        self.m_bh, self.m_by = np.zeros_like(self.bh), np.zeros_like(self.by)

    def forward(self, inputs):
        h = np.zeros((self.wh.shape[0], 1)) 
        self.last_inputs = inputs
        self.last_hs = { 0: h }

        # Perform forward pass through time step
        for i, x in enumerate(inputs):
            h = np.tanh(self.wx @ x.reshape(-1, 1) + self.wh @ h + self.bh)
            self.last_hs[i + 1] = h
        y = self.wy @ h + self.by

        return y, h

    def backward(self, d_y, learn_rate=2e-2, momentum=0.85):
        n = self.wy.shape[0]

        # Calculate gradient of output wrt Wy and by
        d_wy = d_y @ self.last_hs[len(self.last_inputs)].T
        d_by = d_y

        # Initialize dh_next and gradients for Wx, Wh, bh
        dh_next = self.wy.T @ d_y
        d_wx = np.zeros_like(self.wx)
        d_wh = np.zeros_like(self.wh)
        d_bh = np.zeros_like(self.bh)

        # Backpropagation through time
        for t in reversed(range(len(self.last_inputs))):
            temp = ((1 - self.last_hs[t + 1] ** 2) * dh_next)

            # Accumulate gradients for Wx, Wh, and bh
            d_wx += temp @ self.last_inputs[t].reshape(1,-1)
            d_wh += temp @ self.last_hs[t].T
            d_bh += temp

            # Next dh_next
            dh_next = self.wh @ temp

        # Update weights and biases using SGD update with momentum
        self.m_wx = momentum * self.m_wx + learn_rate * d_wx
        self.wx -= self.m_wx
        self.m_wh = momentum * self.m_wh + learn_rate * d_wh
        self.wh -= self.m_wh
        self.m_wy = momentum * self.m_wy + learn_rate * d_wy
        self.wy -= self.m_wy
        self.m_bh = momentum * self.m_bh + learn_rate * d_bh
        self.bh -= self.m_bh
        self.m_by = momentum * self.m_by + learn_rate * d_by
        self.by -= self.m_by

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Loss function
def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred))

# Load dataset
def load_data(filename):
    with h5py.File(filename, 'r') as hf:
        trainX = np.array(hf.get('trX'))
        trainY = np.array(hf.get('trY'))
        testX = np.array(hf.get('tstX'))
        testY = np.array(hf.get('tstY'))
    return trainX, trainY, testX, testY

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

def process_data(trainX, trainY, testX, testY):
    assert trainX.shape[0] == trainY.shape[0], "Mismatch in number of training samples"
    assert testX.shape[0] == testY.shape[0], "Mismatch in number of test samples"

    # Normalize the data
    mean = np.mean(trainX, axis=0)
    std = np.std(trainX, axis=0)

    trainX = (trainX - mean) / std
    testX = (testX - mean) / std

    # Convert labels to integers
    trainY = trainY.astype(int)
    testY = testY.astype(int)

    # One-hot encoding
    #trainY = convert_to_one_hot(trainY, 6).T
    #testY = convert_to_one_hot(testY, 6).T

    return trainX, trainY, testX, testY



def run_model(trainX, trainY, testX, testY, hidden_dim=128, epochs=50, mini_batch_size=32, early_stop_epochs=5):
    rnn = RNN(trainX.shape[2], hidden_dim, trainY.shape[1])

    # Split training data into training and validation set
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.1)

    best_val_loss = float('inf')
    stop_counter = 0
    
    for epoch in range(epochs):
        # Mini-batch gradient descent
        mini_batches = [(trainX[k:k+mini_batch_size], trainY[k:k+mini_batch_size]) 
                        for k in range(0, trainX.shape[0], mini_batch_size)]
        
        for mini_batch in mini_batches:
            X_mini, Y_mini = mini_batch
            for x, y_true in zip(X_mini, Y_mini):
                y, _ = rnn.forward(x)
                y = softmax(y)
                error = y - y_true.reshape(-1,1)
                rnn.backward(error)

        # Compute validation loss and accuracy
        val_loss = 0
        pred_val = []
        true_val = []
        for x, y_true in zip(valX, valY):
            y, _ = rnn.forward(x)
            y = softmax(y)
            val_loss += cross_entropy_loss(y, y_true.reshape(-1,1))
            pred_val.append(np.argmax(y))
            true_val.append(np.argmax(y_true))

        # Calculate overall accuracy for each epoch
        val_accuracy = accuracy_score(true_val, pred_val)
        print('Epoch: %d, Validation Loss: %.4f, Validation Accuracy: %.4f' % 
              (epoch, val_loss / valX.shape[0], val_accuracy))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
        else:
            stop_counter += 1

        if stop_counter == early_stop_epochs:
            print('Early stopping at epoch: %d' % epoch)
            break

    # Train set
    train_loss = 0
    pred_train = []
    true_train = []
    for x, y_true in zip(trainX, trainY):
        y, _ = rnn.forward(x)
        y = softmax(y)
        train_loss += cross_entropy_loss(y, y_true.reshape(-1,1))
        pred_train.append(np.argmax(y))
        true_train.append(np.argmax(y_true))
    print('Train Loss: %.4f' % (train_loss / trainX.shape[0]))
    print("Confusion Matrix - Train Set:\n", confusion_matrix(true_train, pred_train))


trainX, trainY, testX, testY = load_data('data3.h5')
trainX, trainY, testX, testY = process_data(trainX, trainY, testX, testY)
run_model(trainX, trainY, testX, testY)


# a) (for hidden_dim = 32)

# In[14]:


# 3-a
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Xavier initialization
        self.wh = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.wx = np.random.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
        self.wy = np.random.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))
        # Momentum initializations
        self.m_wx, self.m_wh, self.m_wy = np.zeros_like(self.wx), np.zeros_like(self.wh), np.zeros_like(self.wy)
        self.m_bh, self.m_by = np.zeros_like(self.bh), np.zeros_like(self.by)

    def forward(self, inputs):
        h = np.zeros((self.wh.shape[0], 1)) 
        self.last_inputs = inputs
        self.last_hs = { 0: h }

        # Perform forward pass through time step
        for i, x in enumerate(inputs):
            h = np.tanh(self.wx @ x.reshape(-1, 1) + self.wh @ h + self.bh)
            self.last_hs[i + 1] = h
        y = self.wy @ h + self.by

        return y, h

    def backward(self, d_y, learn_rate=2e-2, momentum=0.85):
        n = self.wy.shape[0]

        # Calculate gradient of output wrt Wy and by
        d_wy = d_y @ self.last_hs[len(self.last_inputs)].T
        d_by = d_y

        # Initialize dh_next and gradients for Wx, Wh, bh
        dh_next = self.wy.T @ d_y
        d_wx = np.zeros_like(self.wx)
        d_wh = np.zeros_like(self.wh)
        d_bh = np.zeros_like(self.bh)

        # Backpropagation through time
        for t in reversed(range(len(self.last_inputs))):
            temp = ((1 - self.last_hs[t + 1] ** 2) * dh_next)

            # Accumulate gradients for Wx, Wh, and bh
            d_wx += temp @ self.last_inputs[t].reshape(1,-1)
            d_wh += temp @ self.last_hs[t].T
            d_bh += temp

            # Next dh_next
            dh_next = self.wh @ temp

        # Update weights and biases using SGD update with momentum
        self.m_wx = momentum * self.m_wx + learn_rate * d_wx
        self.wx -= self.m_wx
        self.m_wh = momentum * self.m_wh + learn_rate * d_wh
        self.wh -= self.m_wh
        self.m_wy = momentum * self.m_wy + learn_rate * d_wy
        self.wy -= self.m_wy
        self.m_bh = momentum * self.m_bh + learn_rate * d_bh
        self.bh -= self.m_bh
        self.m_by = momentum * self.m_by + learn_rate * d_by
        self.by -= self.m_by

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Loss function
def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred))

# Load dataset
def load_data(filename):
    with h5py.File(filename, 'r') as hf:
        trainX = np.array(hf.get('trX'))
        trainY = np.array(hf.get('trY'))
        testX = np.array(hf.get('tstX'))
        testY = np.array(hf.get('tstY'))
    return trainX, trainY, testX, testY

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

def process_data(trainX, trainY, testX, testY):
    assert trainX.shape[0] == trainY.shape[0], "Mismatch in number of training samples"
    assert testX.shape[0] == testY.shape[0], "Mismatch in number of test samples"

    # Normalize the data
    mean = np.mean(trainX, axis=0)
    std = np.std(trainX, axis=0)

    trainX = (trainX - mean) / std
    testX = (testX - mean) / std

    # Convert labels to integers
    trainY = trainY.astype(int)
    testY = testY.astype(int)

    # One-hot encoding
    #trainY = convert_to_one_hot(trainY, 6).T
    #testY = convert_to_one_hot(testY, 6).T

    return trainX, trainY, testX, testY



def run_model(trainX, trainY, testX, testY, hidden_dim=32, epochs=50, mini_batch_size=32, early_stop_epochs=5):
    rnn = RNN(trainX.shape[2], hidden_dim, trainY.shape[1])

    # Split training data into training and validation set
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.1)

    best_val_loss = float('inf')
    stop_counter = 0
    
    for epoch in range(epochs):
        # Mini-batch gradient descent
        mini_batches = [(trainX[k:k+mini_batch_size], trainY[k:k+mini_batch_size]) 
                        for k in range(0, trainX.shape[0], mini_batch_size)]
        
        for mini_batch in mini_batches:
            X_mini, Y_mini = mini_batch
            for x, y_true in zip(X_mini, Y_mini):
                y, _ = rnn.forward(x)
                y = softmax(y)
                error = y - y_true.reshape(-1,1)
                rnn.backward(error)

        # Compute validation loss and accuracy
        val_loss = 0
        pred_val = []
        true_val = []
        for x, y_true in zip(valX, valY):
            y, _ = rnn.forward(x)
            y = softmax(y)
            val_loss += cross_entropy_loss(y, y_true.reshape(-1,1))
            pred_val.append(np.argmax(y))
            true_val.append(np.argmax(y_true))

        # Calculate overall accuracy for each epoch
        val_accuracy = accuracy_score(true_val, pred_val)
        print('Epoch: %d, Validation Loss: %.4f, Validation Accuracy: %.4f' % 
              (epoch, val_loss / valX.shape[0], val_accuracy))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
        else:
            stop_counter += 1

        if stop_counter == early_stop_epochs:
            print('Early stopping at epoch: %d' % epoch)
            break

    # Train set
    train_loss = 0
    pred_train = []
    true_train = []
    for x, y_true in zip(trainX, trainY):
        y, _ = rnn.forward(x)
        y = softmax(y)
        train_loss += cross_entropy_loss(y, y_true.reshape(-1,1))
        pred_train.append(np.argmax(y))
        true_train.append(np.argmax(y_true))
    print('Train Loss: %.4f' % (train_loss / trainX.shape[0]))
    print("Confusion Matrix - Train Set:\n", confusion_matrix(true_train, pred_train))


trainX, trainY, testX, testY = load_data('data3.h5')
trainX, trainY, testX, testY = process_data(trainX, trainY, testX, testY)
run_model(trainX, trainY, testX, testY)


# Recurrent Neural Network (RNN)
# Hidden Dimension = 128
# With a validation accuracy of 0.2767, the RNN with 128 neurons in the hidden layer stopped early at the 5th epoch. The train loss was found to be high, indicating that the model did not fit the training data well. The confusion matrix also revealed a relatively random distribution of classifications, indicating that the model was unable to differentiate between different activities.
# 
# Hidden Dimension = 32
# The RNN's performance was improved by reducing the dimensionality of the hidden layer to 32. The model was terminated at the 15th epoch with a validation accuracy of 0.2800. The confusion matrix, however, revealed that the model had difficulty distinguishing between different types of activities.

# b)

# In[13]:


# 3-b
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_dim = hidden_dim
        # Xavier initialization
        self.Wf = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wi = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wc = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wo = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wy = np.random.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.bf = np.zeros((hidden_dim, 1))
        self.bi = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))
        self.last_os = {}
        self.last_c_bars = {}
        self.last_is = {}
        self.last_fs = {}

    def forward(self, inputs):
        h_prev = np.zeros((self.hidden_dim, 1))
        c_prev = np.zeros((self.hidden_dim, 1))
        self.last_inputs = inputs
        self.last_hs = { 0: h_prev }
        self.last_cs = { 0: c_prev }

        # Perform forward pass through time step
        for i, x in enumerate(inputs):
            z = np.row_stack((h_prev, x.reshape(-1, 1)))
            f_gate = sigmoid(self.Wf @ z + self.bf)
            self.last_fs[i] = f_gate
            i_gate = sigmoid(self.Wi @ z + self.bi)
            self.last_is[i] = i_gate
            c_bar = np.tanh(self.Wc @ z + self.bc)
            self.last_c_bars[i] = c_bar
            c = f_gate * c_prev + i_gate * c_bar
            o_gate = sigmoid(self.Wo @ z + self.bo)
            h = o_gate * np.tanh(c)
            self.last_hs[i + 1] = h
            self.last_cs[i + 1] = c
            self.last_os[i + 1] = o_gate  # Store the output gate activation at each time step
            h_prev = h
            c_prev = c

        y = self.Wy @ h + self.by

        return y, h

    def backward(self, d_y, learn_rate=0.1, momentum=0.85):
        # Initialize gradients
        d_Wf, d_Wi, d_Wc, d_Wo, d_Wy = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wc), np.zeros_like(self.Wo), np.zeros_like(self.Wy)
        d_bf, d_bi, d_bc, d_bo, d_by = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bc), np.zeros_like(self.bo), np.zeros_like(self.by)

        dh_next = np.zeros_like(self.last_hs[0])
        dc_next = np.zeros_like(self.last_cs[0])

        d_Wy += d_y @ self.last_hs[len(self.last_inputs)].T
        d_by += d_y
        dh_next += self.Wy.T @ d_y

        # Backpropagate through time
        for t in reversed(range(1, len(self.last_inputs + 1))):
            z = np.row_stack((self.last_hs[t], self.last_inputs[t].reshape(-1, 1)))

            dc = dc_next + (dh_next * self.last_os[t] * (1 - np.tanh(self.last_cs[t+1])**2))
            do = dh_next * np.tanh(self.last_cs[t+1])
            do_input = sigmoid_derivative(self.last_os[t]) * do

            di = dc * self.last_c_bars[t]
            di_input = sigmoid_derivative(self.last_is[t]) * di
            df = dc * self.last_cs[t]
            df_input = sigmoid_derivative(self.last_fs[t]) * df
            dc_bar = dc * self.last_is[t]
            dc_bar_input = (1 - (self.last_c_bars[t])**2) * dc_bar

            # Update gradients
            dz = (self.Wf.T @ df_input
                + self.Wi.T @ di_input
                + self.Wc.T @ dc_bar_input
                + self.Wo.T @ do_input)
            dh_prev = dz[:self.hidden_dim, :]
            d_Wf += df_input @ z.T
            d_bf += df_input
            d_Wi += di_input @ z.T
            d_bi += di_input
            d_Wc += dc_bar_input @ z.T
            d_bc += dc_bar_input
            d_Wo += do_input @ z.T
            d_bo += do_input

            # Prepare for next iteration
            dc_next = self.last_fs[t] * dc
            dh_next = dh_prev

        # Clip to prevent exploding gradients
        for d in [d_Wf, d_Wi, d_Wc, d_Wo, d_Wy, d_bf, d_bi, d_bc, d_bo, d_by]:
            np.clip(d, -1, 1, out=d)

        # Update weights and biases using SGD with momentum
        self.Wf -= learn_rate * d_Wf
        self.Wi -= learn_rate * d_Wi
        self.Wc -= learn_rate * d_Wc
        self.Wo -= learn_rate * d_Wo
        self.Wy -= learn_rate * d_Wy
        self.bf -= learn_rate * d_bf
        self.bi -= learn_rate * d_bi
        self.bc -= learn_rate * d_bc
        self.bo -= learn_rate * d_bo
        self.by -= learn_rate * d_by

        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Loss function
def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred))

# Load dataset
def load_data(filename):
    with h5py.File(filename, 'r') as hf:
        trainX = np.array(hf.get('trX'))
        trainY = np.array(hf.get('trY'))
        testX = np.array(hf.get('tstX'))
        testY = np.array(hf.get('tstY'))
    return trainX, trainY, testX, testY

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

def process_data(trainX, trainY, testX, testY):
    assert trainX.shape[0] == trainY.shape[0], "Mismatch in number of training samples"
    assert testX.shape[0] == testY.shape[0], "Mismatch in number of test samples"

    # Normalize
    trainX /= np.max(trainX)
    testX /= np.max(testX)

    # Convert labels to integers
    trainY = trainY.astype(int)
    testY = testY.astype(int)

    # One-hot encoding
    #trainY = convert_to_one_hot(trainY, 6).T
    #testY = convert_to_one_hot(testY, 6).T

    return trainX, trainY, testX, testY



def run_model(trainX, trainY, testX, testY, hidden_dim=128, epochs=50, mini_batch_size=32, early_stop_epochs=5):
    lstm = LSTM(trainX.shape[2], hidden_dim, trainY.shape[1])

    # Split training data into training and validation set
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.1)

    best_val_loss = float('inf')
    stop_counter = 0
    
    for epoch in range(epochs):
        # Mini-batch gradient descent
        mini_batches = [(trainX[k:k+mini_batch_size], trainY[k:k+mini_batch_size]) 
                        for k in range(0, trainX.shape[0], mini_batch_size)]
        
        for mini_batch in mini_batches:
            X_mini, Y_mini = mini_batch
            for x, y_true in zip(X_mini, Y_mini):
                y, _ = lstm.forward(x)
                y = softmax(y)
                error = y - y_true.reshape(-1,1)
                lstm.backward(error)

        # Compute validation loss and accuracy
        val_loss = 0
        pred_val = []
        true_val = []
        for x, y_true in zip(valX, valY):
            y, _ = lstm.forward(x)
            y = softmax(y)
            val_loss += cross_entropy_loss(y, y_true.reshape(-1,1))
            pred_val.append(np.argmax(y))
            true_val.append(np.argmax(y_true))

        # Calculate overall accuracy for each epoch
        val_accuracy = accuracy_score(true_val, pred_val)
        print('Epoch: %d, Validation Loss: %.4f, Validation Accuracy: %.4f' % 
              (epoch, val_loss / valX.shape[0], val_accuracy))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
        else:
            stop_counter += 1

        if stop_counter == early_stop_epochs:
            print('Early stopping at epoch: %d' % epoch)
            break

    # Test model
    test_loss = 0
    pred_test = []
    true_test = []
    for x, y_true in zip(testX, testY):
        y, _ = lstm.forward(x)
        y = softmax(y)
        test_loss += cross_entropy_loss(y, y_true.reshape(-1,1))
        pred_test.append(np.argmax(y))
        true_test.append(np.argmax(y_true))
    print('Test Loss: %.4f' % (test_loss / testX.shape[0]))
    print("Confusion Matrix - Test Set:\n", confusion_matrix(true_test, pred_test))


trainX, trainY, testX, testY = load_data('data3.h5')
trainX, trainY, testX, testY = process_data(trainX, trainY, testX, testY)
run_model(trainX, trainY, testX, testY)


# b) (for hidden_dim = 32)

# In[9]:


# 3-b (for hidden_dim = 32)
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_dim = hidden_dim
        # Xavier initialization
        self.Wf = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wi = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wc = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wo = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wy = np.random.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.bf = np.zeros((hidden_dim, 1))
        self.bi = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))
        self.last_os = {}
        self.last_c_bars = {}
        self.last_is = {}
        self.last_fs = {}

    def forward(self, inputs):
        h_prev = np.zeros((self.hidden_dim, 1))
        c_prev = np.zeros((self.hidden_dim, 1))
        self.last_inputs = inputs
        self.last_hs = { 0: h_prev }
        self.last_cs = { 0: c_prev }

        # Perform forward pass through time step
        for i, x in enumerate(inputs):
            z = np.row_stack((h_prev, x.reshape(-1, 1)))
            f_gate = sigmoid(self.Wf @ z + self.bf)
            self.last_fs[i] = f_gate # Store the forget gate activation at each time step
            i_gate = sigmoid(self.Wi @ z + self.bi)
            self.last_is[i] = i_gate  # Store the input gate activation at each time step
            c_bar = np.tanh(self.Wc @ z + self.bc)
            self.last_c_bars[i] = c_bar
            c = f_gate * c_prev + i_gate * c_bar
            o_gate = sigmoid(self.Wo @ z + self.bo)
            h = o_gate * np.tanh(c)
            self.last_hs[i + 1] = h
            self.last_cs[i + 1] = c
            self.last_os[i + 1] = o_gate  # Store the output gate activation at each time step
            h_prev = h
            c_prev = c

        y = self.Wy @ h + self.by

        return y, h

    def backward(self, d_y, learn_rate=0.1, momentum=0.85):
        # Initialize gradients
        d_Wf, d_Wi, d_Wc, d_Wo, d_Wy = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wc), np.zeros_like(self.Wo), np.zeros_like(self.Wy)
        d_bf, d_bi, d_bc, d_bo, d_by = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bc), np.zeros_like(self.bo), np.zeros_like(self.by)

        dh_next = np.zeros_like(self.last_hs[0])
        dc_next = np.zeros_like(self.last_cs[0])

        d_Wy += d_y @ self.last_hs[len(self.last_inputs)].T
        d_by += d_y
        dh_next += self.Wy.T @ d_y

        # Backpropagate through time
        for t in reversed(range(1, len(self.last_inputs + 1))):
            z = np.row_stack((self.last_hs[t], self.last_inputs[t].reshape(-1, 1)))

            dc = dc_next + (dh_next * self.last_os[t] * (1 - np.tanh(self.last_cs[t+1])**2))
            do = dh_next * np.tanh(self.last_cs[t+1])
            do_input = sigmoid_derivative(self.last_os[t]) * do

            di = dc * self.last_c_bars[t]
            di_input = sigmoid_derivative(self.last_is[t]) * di
            df = dc * self.last_cs[t]
            df_input = sigmoid_derivative(self.last_fs[t]) * df
            dc_bar = dc * self.last_is[t]
            dc_bar_input = (1 - (self.last_c_bars[t])**2) * dc_bar

            # Update gradients
            dz = (self.Wf.T @ df_input
                + self.Wi.T @ di_input
                + self.Wc.T @ dc_bar_input
                + self.Wo.T @ do_input)
            dh_prev = dz[:self.hidden_dim, :]
            d_Wf += df_input @ z.T
            d_bf += df_input
            d_Wi += di_input @ z.T
            d_bi += di_input
            d_Wc += dc_bar_input @ z.T
            d_bc += dc_bar_input
            d_Wo += do_input @ z.T
            d_bo += do_input

            # Prepare for next iteration
            dc_next = self.last_fs[t] * dc
            dh_next = dh_prev

        # Clip to prevent exploding gradients
        for d in [d_Wf, d_Wi, d_Wc, d_Wo, d_Wy, d_bf, d_bi, d_bc, d_bo, d_by]:
            np.clip(d, -1, 1, out=d)

        # Update weights and biases using SGD with momentum
        self.Wf -= learn_rate * d_Wf
        self.Wi -= learn_rate * d_Wi
        self.Wc -= learn_rate * d_Wc
        self.Wo -= learn_rate * d_Wo
        self.Wy -= learn_rate * d_Wy
        self.bf -= learn_rate * d_bf
        self.bi -= learn_rate * d_bi
        self.bc -= learn_rate * d_bc
        self.bo -= learn_rate * d_bo
        self.by -= learn_rate * d_by

        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Loss function
def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred))

# Load dataset
def load_data(filename):
    with h5py.File(filename, 'r') as hf:
        trainX = np.array(hf.get('trX'))
        trainY = np.array(hf.get('trY'))
        testX = np.array(hf.get('tstX'))
        testY = np.array(hf.get('tstY'))
    return trainX, trainY, testX, testY

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

def process_data(trainX, trainY, testX, testY):
    assert trainX.shape[0] == trainY.shape[0], "Mismatch in number of training samples"
    assert testX.shape[0] == testY.shape[0], "Mismatch in number of test samples"

    # Normalize
    trainX /= np.max(trainX)
    testX /= np.max(testX)

    # Convert labels to integers
    trainY = trainY.astype(int)
    testY = testY.astype(int)

    # One-hot encoding
    #trainY = convert_to_one_hot(trainY, 6).T
    #testY = convert_to_one_hot(testY, 6).T

    return trainX, trainY, testX, testY



def run_model(trainX, trainY, testX, testY, hidden_dim=32, epochs=50, mini_batch_size=32, early_stop_epochs=5):
    lstm = LSTM(trainX.shape[2], hidden_dim, trainY.shape[1])

    # Split training data into training and validation set
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.1)

    best_val_loss = float('inf')
    stop_counter = 0
    
    for epoch in range(epochs):      
        # Mini-batch gradient descent
        mini_batches = [(trainX[k:k+mini_batch_size], trainY[k:k+mini_batch_size]) 
                        for k in range(0, trainX.shape[0], mini_batch_size)]
        
        for mini_batch in mini_batches:
            X_mini, Y_mini = mini_batch
            for x, y_true in zip(X_mini, Y_mini):
                y, _ = lstm.forward(x)
                y = softmax(y)
                error = y - y_true.reshape(-1,1)
                lstm.backward(error)

        # Compute validation loss and accuracy
        val_loss = 0
        pred_val = []
        true_val = []
        for x, y_true in zip(valX, valY):
            y, _ = lstm.forward(x)
            y = softmax(y)
            val_loss += cross_entropy_loss(y, y_true.reshape(-1,1))
            pred_val.append(np.argmax(y))
            true_val.append(np.argmax(y_true))

        # Calculate overall accuracy for each epoch
        val_accuracy = accuracy_score(true_val, pred_val)
        print('Epoch: %d, Validation Loss: %.4f, Validation Accuracy: %.4f' % 
              (epoch, val_loss / valX.shape[0], val_accuracy))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
        else:
            stop_counter += 1

        if stop_counter == early_stop_epochs:
            print('Early stopping at epoch: %d' % epoch)
            break

    # Test model
    test_loss = 0
    pred_test = []
    true_test = []
    for x, y_true in zip(testX, testY):
        y, _ = lstm.forward(x)
        y = softmax(y)
        test_loss += cross_entropy_loss(y, y_true.reshape(-1,1))
        pred_test.append(np.argmax(y))
        true_test.append(np.argmax(y_true))
    print('Test Loss: %.4f' % (test_loss / testX.shape[0]))
    print("Confusion Matrix - Test Set:\n", confusion_matrix(true_test, pred_test))


trainX, trainY, testX, testY = load_data('data3.h5')
trainX, trainY, testX, testY = process_data(trainX, trainY, testX, testY)
run_model(trainX, trainY, testX, testY)


# Long Short-Term Memory (LSTM)
# Hidden Dimension = 128
# When compared to the RNN, the LSTM with 128 neurons stopped training at the 24th epoch and achieved a higher validation accuracy of 0.5767. The confusion matrix revealed a more accurate distribution of classifications, implying that LSTM was better able to learn the sequence dependencies within the data.
# 
# Hidden Dimension = 32
# The LSTM model with 32 hidden units performed even better, terminating training at the 9th epoch and achieving a higher validation accuracy of 0.6900. The confusion matrix revealed a more accurate classification distribution, implying that reducing the number of hidden neurons improved the model's performance on this problem.

# c)

# In[11]:


# 3-c
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class GRU:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_dim = hidden_dim
        # Xavier initialization
        self.Wz = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wr = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wh = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wy = np.random.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.bz = np.zeros((hidden_dim, 1))
        self.br = np.zeros((hidden_dim, 1))
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))

    def forward(self, inputs):
        h_prev = np.zeros((self.hidden_dim, 1))
        self.last_inputs = inputs
        self.last_hs = { 0: h_prev }
        
        # Perform forward pass through time step
        for i, x in enumerate(inputs):
            z = np.row_stack((h_prev, x.reshape(-1, 1)))
            z_gate = sigmoid(self.Wz @ z + self.bz)
            r_gate = sigmoid(self.Wr @ z + self.br)
            h_bar = np.tanh(self.Wh @ np.row_stack((r_gate * h_prev, x.reshape(-1, 1))) + self.bh)
            h = ((1 - z_gate) * h_prev) + (z_gate * h_bar)
            self.last_hs[i + 1] = h
            h_prev = h

        y = self.Wy @ h + self.by

        return y, h

    def backward(self, d_y, learn_rate=0.1, momentum=0.85):
        d_Wz, d_Wr, d_Wh, d_Wy = np.zeros_like(self.Wz), np.zeros_like(self.Wr), np.zeros_like(self.Wh), np.zeros_like(self.Wy)
        d_bz, d_br, d_bh, d_by = np.zeros_like(self.bz), np.zeros_like(self.br), np.zeros_like(self.bh), np.zeros_like(self.by)

        dh_next = np.zeros_like(self.last_hs[0])

        d_Wy += d_y @ self.last_hs[len(self.last_inputs)].T
        d_by += d_y
        dh_next += self.Wy.T @ d_y

        for t in reversed(range(1, len(self.last_inputs + 1))):
            z = np.row_stack((self.last_hs[t-1], self.last_inputs[t-1].reshape(-1, 1)))

            dh = dh_next

            z_gate = sigmoid(self.Wz @ z + self.bz)
            r_gate = sigmoid(self.Wr @ z + self.br)
            h_bar = np.tanh(self.Wh @ np.row_stack((r_gate * self.last_hs[t-1], self.last_inputs[t-1].reshape(-1, 1))) + self.bh)

            dh_bar = dh * z_gate
            dz_gate = dh * (h_bar - self.last_hs[t-1])
            dh_prev = dh * (1 - z_gate)

            d_Wh += (1 - np.tanh(h_bar) ** 2) * dh_bar @ np.row_stack((r_gate * self.last_hs[t-1], self.last_inputs[t-1].reshape(-1, 1))).T
            d_bh += (1 - np.tanh(h_bar) ** 2) * dh_bar

            dr_gate = self.Wh.T @ (1 - np.tanh(h_bar) ** 2) * dh_bar * self.last_hs[t-1]
            d_Wr += sigmoid_derivative(r_gate) * dr_gate @ z.T
            d_br += sigmoid_derivative(r_gate) * dr_gate

            dz_gate = self.Wh.T @ (1 - np.tanh(h_bar) ** 2) * dh_bar * self.last_hs[t-1] + dh * (h_bar - self.last_hs[t-1])
            d_Wz += sigmoid_derivative(z_gate) * dz_gate @ z.T
            d_bz += sigmoid_derivative(z_gate) * dz_gate

            dh_next = self.Wz.T @ sigmoid_derivative(z_gate) * dz_gate + self.Wr.T @ sigmoid_derivative(r_gate) * dr_gate + self.Wh.T @ (1 - np.tanh(h_bar) ** 2) * dh_bar * r_gate + dh * (1 - z_gate)

        for d in [d_Wz, d_Wr, d_Wh, d_Wy, d_bz, d_br, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Update weights and biases using SGD
        self.Wz -= learn_rate * d_Wz
        self.Wr -= learn_rate * d_Wr
        self.Wh -= learn_rate * d_Wh
        self.Wy -= learn_rate * d_Wy
        self.bz -= learn_rate * d_bz
        self.br -= learn_rate * d_br
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by


        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Loss function
def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred))

# Load dataset
def load_data(filename):
    with h5py.File(filename, 'r') as hf:
        trainX = np.array(hf.get('trX'))
        trainY = np.array(hf.get('trY'))
        testX = np.array(hf.get('tstX'))
        testY = np.array(hf.get('tstY'))
    return trainX, trainY, testX, testY

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

def process_data(trainX, trainY, testX, testY):
    assert trainX.shape[0] == trainY.shape[0], "Mismatch in number of training samples"
    assert testX.shape[0] == testY.shape[0], "Mismatch in number of test samples"

    # Normalize
    trainX /= np.max(trainX)
    testX /= np.max(testX)

    # Convert labels to integers
    trainY = trainY.astype(int)
    testY = testY.astype(int)

    # One-hot encoding
    #trainY = convert_to_one_hot(trainY, 6).T
    #testY = convert_to_one_hot(testY, 6).T

    return trainX, trainY, testX, testY



def run_model(trainX, trainY, testX, testY, hidden_dim=128, epochs=50, mini_batch_size=32, early_stop_epochs=5):
    lstm = LSTM(trainX.shape[2], hidden_dim, trainY.shape[1])

    # Split training data into training and validation set
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.1)

    best_val_loss = float('inf')
    stop_counter = 0
    
    for epoch in range(epochs):
        # Mini-batch gradient descent
        mini_batches = [(trainX[k:k+mini_batch_size], trainY[k:k+mini_batch_size]) 
                        for k in range(0, trainX.shape[0], mini_batch_size)]
        
        for mini_batch in mini_batches:
            X_mini, Y_mini = mini_batch
            for x, y_true in zip(X_mini, Y_mini):
                y, _ = lstm.forward(x)
                y = softmax(y)
                error = y - y_true.reshape(-1,1)
                lstm.backward(error)

        # Compute validation loss and accuracy
        val_loss = 0
        pred_val = []
        true_val = []
        for x, y_true in zip(valX, valY):
            y, _ = lstm.forward(x)
            y = softmax(y)
            val_loss += cross_entropy_loss(y, y_true.reshape(-1,1))
            pred_val.append(np.argmax(y))
            true_val.append(np.argmax(y_true))

        # Calculate overall accuracy for each epoch
        val_accuracy = accuracy_score(true_val, pred_val)
        print('Epoch: %d, Validation Loss: %.4f, Validation Accuracy: %.4f' % 
              (epoch, val_loss / valX.shape[0], val_accuracy))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
        else:
            stop_counter += 1

        if stop_counter == early_stop_epochs:
            print('Early stopping at epoch: %d' % epoch)
            break

    # Test model
    test_loss = 0
    pred_test = []
    true_test = []
    for x, y_true in zip(testX, testY):
        y, _ = lstm.forward(x)
        y = softmax(y)
        test_loss += cross_entropy_loss(y, y_true.reshape(-1,1))
        pred_test.append(np.argmax(y))
        true_test.append(np.argmax(y_true))
    print('Test Loss: %.4f' % (test_loss / testX.shape[0]))
    print("Confusion Matrix - Test Set:\n", confusion_matrix(true_test, pred_test))


trainX, trainY, testX, testY = load_data('data3.h5')
trainX, trainY, testX, testY = process_data(trainX, trainY, testX, testY)
run_model(trainX, trainY, testX, testY)


# c) (for hidden_dim = 32)

# In[12]:


# 3-c (for hidden_dim = 32)
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class GRU:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_dim = hidden_dim
        # Xavier initialization
        self.Wz = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wr = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wh = np.random.randn(hidden_dim, hidden_dim + input_dim) / np.sqrt(hidden_dim + input_dim)
        self.Wy = np.random.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.bz = np.zeros((hidden_dim, 1))
        self.br = np.zeros((hidden_dim, 1))
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))

    def forward(self, inputs):
        h_prev = np.zeros((self.hidden_dim, 1))
        self.last_inputs = inputs
        self.last_hs = { 0: h_prev }
        
        # Perform forward pass through time step
        for i, x in enumerate(inputs):
            z = np.row_stack((h_prev, x.reshape(-1, 1)))
            z_gate = sigmoid(self.Wz @ z + self.bz)
            r_gate = sigmoid(self.Wr @ z + self.br)
            h_bar = np.tanh(self.Wh @ np.row_stack((r_gate * h_prev, x.reshape(-1, 1))) + self.bh)
            h = ((1 - z_gate) * h_prev) + (z_gate * h_bar)
            self.last_hs[i + 1] = h
            h_prev = h

        y = self.Wy @ h + self.by

        return y, h

    def backward(self, d_y, learn_rate=0.1, momentum=0.85):
        d_Wz, d_Wr, d_Wh, d_Wy = np.zeros_like(self.Wz), np.zeros_like(self.Wr), np.zeros_like(self.Wh), np.zeros_like(self.Wy)
        d_bz, d_br, d_bh, d_by = np.zeros_like(self.bz), np.zeros_like(self.br), np.zeros_like(self.bh), np.zeros_like(self.by)

        dh_next = np.zeros_like(self.last_hs[0])

        d_Wy += d_y @ self.last_hs[len(self.last_inputs)].T
        d_by += d_y
        dh_next += self.Wy.T @ d_y

        for t in reversed(range(1, len(self.last_inputs + 1))):
            z = np.row_stack((self.last_hs[t-1], self.last_inputs[t-1].reshape(-1, 1)))

            dh = dh_next

            z_gate = sigmoid(self.Wz @ z + self.bz)
            r_gate = sigmoid(self.Wr @ z + self.br)
            h_bar = np.tanh(self.Wh @ np.row_stack((r_gate * self.last_hs[t-1], self.last_inputs[t-1].reshape(-1, 1))) + self.bh)

            dh_bar = dh * z_gate
            dz_gate = dh * (h_bar - self.last_hs[t-1])
            dh_prev = dh * (1 - z_gate)

            d_Wh += (1 - np.tanh(h_bar) ** 2) * dh_bar @ np.row_stack((r_gate * self.last_hs[t-1], self.last_inputs[t-1].reshape(-1, 1))).T
            d_bh += (1 - np.tanh(h_bar) ** 2) * dh_bar

            dr_gate = self.Wh.T @ (1 - np.tanh(h_bar) ** 2) * dh_bar * self.last_hs[t-1]
            d_Wr += sigmoid_derivative(r_gate) * dr_gate @ z.T
            d_br += sigmoid_derivative(r_gate) * dr_gate

            dz_gate = self.Wh.T @ (1 - np.tanh(h_bar) ** 2) * dh_bar * self.last_hs[t-1] + dh * (h_bar - self.last_hs[t-1])
            d_Wz += sigmoid_derivative(z_gate) * dz_gate @ z.T
            d_bz += sigmoid_derivative(z_gate) * dz_gate

            dh_next = self.Wz.T @ sigmoid_derivative(z_gate) * dz_gate + self.Wr.T @ sigmoid_derivative(r_gate) * dr_gate + self.Wh.T @ (1 - np.tanh(h_bar) ** 2) * dh_bar * r_gate + dh * (1 - z_gate)

        for d in [d_Wz, d_Wr, d_Wh, d_Wy, d_bz, d_br, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Update weights and biases using SGD
        self.Wz -= learn_rate * d_Wz
        self.Wr -= learn_rate * d_Wr
        self.Wh -= learn_rate * d_Wh
        self.Wy -= learn_rate * d_Wy
        self.bz -= learn_rate * d_bz
        self.br -= learn_rate * d_br
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by


        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Loss function
def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred))

# Load dataset
def load_data(filename):
    with h5py.File(filename, 'r') as hf:
        trainX = np.array(hf.get('trX'))
        trainY = np.array(hf.get('trY'))
        testX = np.array(hf.get('tstX'))
        testY = np.array(hf.get('tstY'))
    return trainX, trainY, testX, testY

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

def process_data(trainX, trainY, testX, testY):
    assert trainX.shape[0] == trainY.shape[0], "Mismatch in number of training samples"
    assert testX.shape[0] == testY.shape[0], "Mismatch in number of test samples"

    # Normalize
    trainX /= np.max(trainX)
    testX /= np.max(testX)

    # Convert labels to integers
    trainY = trainY.astype(int)
    testY = testY.astype(int)

    # One-hot encoding
    #trainY = convert_to_one_hot(trainY, 6).T
    #testY = convert_to_one_hot(testY, 6).T

    return trainX, trainY, testX, testY



def run_model(trainX, trainY, testX, testY, hidden_dim=32, epochs=50, mini_batch_size=32, early_stop_epochs=5):
    lstm = LSTM(trainX.shape[2], hidden_dim, trainY.shape[1])

    # Split training data into training and validation set
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.1)

    best_val_loss = float('inf')
    stop_counter = 0
    
    for epoch in range(epochs):
        # Mini-batch gradient descent
        mini_batches = [(trainX[k:k+mini_batch_size], trainY[k:k+mini_batch_size]) 
                        for k in range(0, trainX.shape[0], mini_batch_size)]
        
        for mini_batch in mini_batches:
            X_mini, Y_mini = mini_batch
            for x, y_true in zip(X_mini, Y_mini):
                y, _ = lstm.forward(x)
                y = softmax(y)
                error = y - y_true.reshape(-1,1)
                lstm.backward(error)

        # Compute validation loss and accuracy
        val_loss = 0
        pred_val = []
        true_val = []
        for x, y_true in zip(valX, valY):
            y, _ = lstm.forward(x)
            y = softmax(y)
            val_loss += cross_entropy_loss(y, y_true.reshape(-1,1))
            pred_val.append(np.argmax(y))
            true_val.append(np.argmax(y_true))

        # Calculate overall accuracy for each epoch
        val_accuracy = accuracy_score(true_val, pred_val)
        print('Epoch: %d, Validation Loss: %.4f, Validation Accuracy: %.4f' % 
              (epoch, val_loss / valX.shape[0], val_accuracy))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
        else:
            stop_counter += 1

        if stop_counter == early_stop_epochs:
            print('Early stopping at epoch: %d' % epoch)
            break

    # Test model
    test_loss = 0
    pred_test = []
    true_test = []
    for x, y_true in zip(testX, testY):
        y, _ = lstm.forward(x)
        y = softmax(y)
        test_loss += cross_entropy_loss(y, y_true.reshape(-1,1))
        pred_test.append(np.argmax(y))
        true_test.append(np.argmax(y_true))
    print('Test Loss: %.4f' % (test_loss / testX.shape[0]))
    print("Confusion Matrix - Test Set:\n", confusion_matrix(true_test, pred_test))


trainX, trainY, testX, testY = load_data('data3.h5')
trainX, trainY, testX, testY = process_data(trainX, trainY, testX, testY)
run_model(trainX, trainY, testX, testY)


# Gated Recurrent Unit (GRU)
# Hidden Dimension = 128
# The GRU model with 128 hidden units performed poorly, with training stopping at the sixth epoch and a validation accuracy of 0.1967. According to the confusion matrix, the model struggled with the classification task, most likely due to overfitting or an inability to learn the dependencies within the sequence data.
# 
# Hidden Dimension = 32
# The GRU model with 32 hidden neurons outperformed the others, with training stopping at the 13th epoch and a validation accuracy of 0.6033. The confusion matrix revealed a better classification distribution, implying that reducing the number of hidden neurons improved the model's performance for this problem.

# Out of all configurations, the LSTM and GRU models with 32 hidden units performed the best. Both models had higher validation accuracy and better confusion matrices, indicating that they were better at learning the dependencies within the sequence data and distinguishing between different activities.
# 
# Although the LSTM and GRU models with 128 hidden units did not perform as well as the RNN models, they outperformed them. This suggests that for this type of time-series classification problem, the ability of LSTM and GRU to control the flow of information through time and retain important historical information from the sequence data provides significant advantages over a simple RNN model.
# 
# The best performance was achieved by the LSTM model with 32 hidden neurons, which had the highest validation accuracy and a confusion matrix indicating a good classification distribution. The LSTM's mechanism for controlling and retaining information over time may make it particularly well suited to this problem.
# 
# Each model's confusion matrix provides information about the model's specific strengths and weaknesses. The RNN models struggled to distinguish between different activities, resulting in poor performance. The LSTM and GRU models, on the other hand, performed better in distinguishing between activities, as evidenced by a greater concentration of values along the diagonal of the confusion matrices.
# 
# In conclusion, due to their ability to handle time-series data, LSTM and GRU appear to be better choices than a simple RNN for this human activity classification task. The choice between LSTM and GRU may be influenced by the number of hidden dimensions and computational resources available, with LSTM requiring more computational resources but potentially providing better performance with larger hidden dimensions.
# 
# 
