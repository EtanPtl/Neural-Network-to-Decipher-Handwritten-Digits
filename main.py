import cv2 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./mnist_train.csv')
data.head()

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

# Neural Network

def init_params():
    W1 = np.random.rand(128, 784) - 0.5
    b1 = np.random.rand(128, 1) - 0.5
    W2 = np.random.rand(64, 128) - 0.5
    b2 = np.random.rand(64, 1) - 0.5
    W3 = np.random.rand(10, 64) - 0.5
    b3 = np.random.rand(10,1)

    return W1, b1, W2, b2, W3, b3 

def LeakyRelu(Z):
    return np.where(Z>0, Z, 0.01*Z)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))


def forward_propagation(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = LeakyRelu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = LeakyRelu(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def derivative_Relu(Z):
    return Z > 0

def back_propagation(Z1, A1, Z2, A2, Z3, A3, W3, W2, X, Y):
    # m is batch size
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1/m * dZ3.dot(A2.T)
    db3 = 1/m * np.sum(dZ3)

    dZ2 = W3.T.dot(dZ3) * derivative_Relu(Z2)
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * derivative_Relu(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1    
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, epochs, alpha, batch_size):
    W1, b1, W2, b2, W3, b3 = init_params()
    m = X.shape[1]
    for i in range(1, epochs+ 1):
        indices = np.random.permutation(m)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[indices]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[:, j:j+batch_size]
            Y_batch = Y_shuffled[j:j+batch_size]
            Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X_batch)
            dW1, db1, dW2, db2, dW3, db3 = back_propagation(Z1, A1, Z2, A2, Z3, A3, W3, W2, X_batch, Y_batch)
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        _, _, _, _, _, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
        print("epoch: ", i)
        print("Accuracy: ", get_accuracy(get_predictions(A3), Y))
    
    return W1, b1, W2, b2, W3, b3

W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 5, 0.1, 32)


# TESTING ON THE TEST DATA

data_test = pd.read_csv('./mnist_test.csv')
data_test = np.array(data_test)
m_test, n_test = data_test.shape

Y_test = data_test[:,0]
X_test = data_test[:, 1:n_test]
X_test = X_test.T / 255


_, _, _, _, _, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X_test)
print("TEST DATA")
print("Accuracy: ", get_accuracy(get_predictions(A3), Y_test))

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def test_predictions(index, W1, b1, W2, b2, W3, b3):
    current_img = X_test[:, index, None]
    prediction = make_predictions(X_test[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_test[index]

    print("Prediciton: ", prediction)
    print("Label: ", label)
    current_img = current_img.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_img, interpolation='nearest')
    plt.show()
    

for i in range(5):
    n = np.random.rand()
    test_predictions(i, W1, b1, W2, b2, W3, b3)
