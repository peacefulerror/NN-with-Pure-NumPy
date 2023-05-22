#!/usr/bin/env python
# coding: utf-8

# Question-2

# In[3]:


# Question 2-a
import h5py
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def load_data(filename):
    with h5py.File(filename, 'r') as hf:
        trainx = np.array(hf['trainx'])
        traind = np.array(hf['traind'])
        valx = np.array(hf['valx'])
        vald = np.array(hf['vald'])
        testx = np.array(hf['testx'])
        testd = np.array(hf['testd'])
    return trainx, traind, valx, vald, testx, testd

def one_hot_encode(data, vocab_size):
    encoder = OneHotEncoder(sparse=False, categories=[range(vocab_size)])
    return encoder.fit_transform(data.reshape(-1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]

def init_params(input_size, hidden_size, output_size, vocab_size, embedding_dim):
    params = {}
    params['E'] = np.random.normal(0, 0.01, (vocab_size + 1, embedding_dim))
    params['W1'] = np.random.normal(0, 0.01, (embedding_dim, hidden_size))
    params['b1'] = np.zeros((1, hidden_size))
    params['W2'] = np.random.normal(0, 0.01, (hidden_size, output_size))
    params['b2'] = np.zeros((1, output_size))
    return params

def forward_backward(params, x_batch, y_batch, vocab_size, momentum=0.85, learning_rate=0.15):
    # Forward pass
    E, W1, b1, W2, b2 = params['E'], params['W1'], params['b1'], params['W2'], params['b2']
    x_embedded = np.sum(E[x_batch], axis=1)
    h = sigmoid(x_embedded.dot(W1) + b1)
    y_pred = softmax(h.dot(W2) + b2)
    grads_prev = {key: np.zeros_like(val) for key, val in params.items()}

    # Backward pass
    grads = {}
    d_y = (y_pred - y_batch) / y_batch.shape[0]
    grads['W2'] = h.T.dot(d_y) + momentum * W2
    grads['b2'] = np.sum(d_y, axis=0, keepdims=True)

    d_h = d_y.dot(W2.T) * (h * (1 - h))
    grads['W1'] = x_embedded.T.dot(d_h) + momentum * W1
    grads['b1'] = np.sum(d_h, axis=0, keepdims=True)

    d_E = np.zeros_like(E)
    for i, x in enumerate(x_batch):
        d_embedded = d_h[i].dot(W1.T)
        d_E[x] += d_embedded

    grads['E'] = d_E + momentum * E

    # Update params
    params['W2'] -= learning_rate * grads['W2']
    params['b2'] -= learning_rate * grads['b2']
    params['W1'] -= learning_rate * grads['W1']
    params['b1'] -= learning_rate * grads['b1']
    params['E'] -= learning_rate * grads['E']

    for key in params:
        params[key] -= learning_rate * (grads[key] + momentum * grads_prev[key])
        grads_prev[key] = grads[key]
        
    return params, y_pred

def train_network(trainx, traind, valx, vald, testx, testd, vocab_size, embedding_dim, hidden_size, epochs=50, batch_size=200):
    trainy = one_hot_encode(traind, vocab_size + 1)
    valy = one_hot_encode(vald, vocab_size + 1)
    testy = one_hot_encode(testd, vocab_size + 1)
    params = init_params(embedding_dim, hidden_size, vocab_size + 1, vocab_size, embedding_dim)
    prev_val_loss = float('inf')
    tolerance = 1e-4
    for epoch in range(epochs):

        for batch_start in range(0, trainx.shape[0], batch_size):
            x_batch = trainx[batch_start:batch_start + batch_size]
            y_batch = trainy[batch_start:batch_start + batch_size]
            params, _ = forward_backward(params, x_batch, y_batch, vocab_size)

        _, val_pred = forward_backward(params, valx, valy, vocab_size)
        val_loss = cross_entropy(val_pred, valy)
        print(f'Epoch {epoch + 1}/{epochs}, validation loss: {val_loss}')
        if prev_val_loss - val_loss < tolerance:
            print('Training stopped due to early stopping')
            break
        prev_val_loss = val_loss
    _, test_pred = forward_backward(params, testx, testy, vocab_size)
    test_loss = cross_entropy(test_pred, testy)
    print(f'Test loss: {test_loss}')

    return params

def predict(params, trigrams, vocab_size, top_k=10):
    E, W1, b1, W2, b2 = params['E'], params['W1'], params['b1'], params['W2'], params['b2']
    x_embedded = np.sum(E[trigrams], axis=1)
    h = sigmoid(x_embedded.dot(W1) + b1)
    y_pred = softmax(h.dot(W2) + b2)

    top_k_indices = np.argsort(y_pred, axis=1)[:, -top_k:]
    return top_k_indices, y_pred

if __name__ == '__main__':
    trainx, traind, valx, vald, testx, testd = load_data('data2.h5')
    vocab_size = 250
    embedding_dim = 32
    hidden_size = 256
    params = train_network(trainx, traind, valx, vald, testx, testd, vocab_size, embedding_dim, hidden_size)

    # Sample trigrams
    sample_trigrams = testx[:5]
    top_k_indices, _ = predict(params, sample_trigrams, vocab_size)
    for i, trigram in enumerate(sample_trigrams):
        print(f'Trigram: {trigram}, top 10 predictions: {top_k_indices[i]}')


# In[29]:


if __name__ == '__main__':
    trainx, traind, valx, vald, testx, testd = load_data('data2.h5')
    vocab_size = 250
    embedding_dim = 16
    hidden_size = 128
    params = train_network(trainx, traind, valx, vald, testx, testd, vocab_size, embedding_dim, hidden_size)
    
    # Sample trigrams
    sample_trigrams = testx[:5]
    top_k_indices, _ = predict(params, sample_trigrams, vocab_size)
    for i, trigram in enumerate(sample_trigrams):
        print(f'Trigram: {trigram}, top 10 predictions: {top_k_indices[i]}')


# In[3]:


if __name__ == '__main__':
    trainx, traind, valx, vald, testx, testd = load_data('data2.h5')
    vocab_size = 250
    embedding_dim = 8
    hidden_size = 64
    params = train_network(trainx, traind, valx, vald, testx, testd, vocab_size, embedding_dim, hidden_size)

    # Sample trigrams
    sample_trigrams = testx[:5]
    top_k_indices, _ = predict(params, sample_trigrams, vocab_size)
    for i, trigram in enumerate(sample_trigrams):
        print(f'Trigram: {trigram}, top 10 predictions: {top_k_indices[i]}')


# Models trained with (D, P) = (32,256), (16,128) and (8,64) value configurations all show promise as they reduce validation loss over epochs, indicating learning from training data.
# 
# (32,256): The model consistently improves its performance over 31 epochs, demonstrating a strong ability to learn from data.
# 
# 
# (16,128): This model achieves comparable validation loss levels in 31 epochs, indicating potentially faster learning and efficient training time.
# 
# (8,64): Despite being the smallest model, it performs well, achieving a similar validation loss level in only 20 epochs due to early stopping.
# 
# These findings are encouraging. To improve further, we can test larger models and fine-tune learning rates too.

# In[28]:


# # Question 2-b
import numpy as np

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def predict(params, x_batch, vocab_size):
    # Forward pass
    E, W1, b1, W2, b2 = params['E'], params['W1'], params['b1'], params['W2'], params['b2']
    x_embedded = np.sum(E[x_batch], axis=1)
    h = sigmoid(x_embedded.dot(W1) + b1)
    y_pred = softmax(h.dot(W2) + b2)
    return y_pred

def top_k_predictions(pred_probs, k=10):
    return np.argsort(pred_probs, axis=1)[:, -k:]

# load the data
with h5py.File('data2.h5', 'r') as hf:
    words = hf['words'][()]
    testx = hf['testx'][()]

# decode bytes to string
words = [word.decode('utf-8') for word in words]
word2index = {word: index for index, word in enumerate(words)}
index2word = {index: word for index, word in enumerate(words)}

# pick some sample trigrams from the test data
sample_indices = np.random.choice(len(testx), size=1, replace=False)
sample_trigrams = testx[sample_indices]

# generate predictions for the fourth word
pred_probs = predict(params, sample_trigrams, len(words))
top_10_preds = top_k_predictions(pred_probs, k=10)

# print the top 10 candidates for each sample trigram
for i, trigram in enumerate(sample_trigrams):
    print(f'Trigram: {" ".join(index2word[idx] for idx in trigram)}')
    print('Top 10 predictions for the fourth word:')
    for pred in reversed(top_10_preds[i]):
        if pred in index2word:
            print(f'  {index2word[pred]}')
        else:
            print(f'  Index {pred} not found in vocabulary')
    print()


# The model's predictions for "only them said" seem reasonable. Depending on the situation, phrases like "only them said its," "only them said after," or "only them said we" might make sense. 
