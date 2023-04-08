
# coding: utf-8

# In[4]:


import numpy as np
from mnist import load_mnist

(train_images, train_labels), (test_images, test_labels) = load_mnist()


# In[ ]:


def normalize_images(images):
    return images / 255.0

def one_hot_labels(labels, num_classes):
    return np.eye(num_classes)[labels]

num_classes = 10
train_images = normalize_images(train_images)
test_images = normalize_images(test_images)
train_labels = one_hot_labels(train_labels, num_classes)
test_labels = one_hot_labels(test_labels, num_classes)


# In[ ]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(0)
    W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
    b2 = np.zeros((1, output_size))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


# In[ ]:


def forward_propagation(X, parameters):
    W1, b1, W2, b2 = parameters
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}


# In[ ]:


def compute_loss(X, Y, parameters, lambd):
    m = X.shape[0]
    A2 = forward_propagation(X, parameters)['A2']
    logprobs = np.multiply(-np.log(A2), Y)
    data_loss = np.sum(logprobs) / m
    reg_loss = (lambd / (2 * m)) * (np.sum(np.square(parameters['W1'])) + np.sum(np.square(parameters['W2'])))
    return data_loss + reg_loss

def compute_gradients(X, Y, parameters, lambd):
    m = X.shape[0]
    cache = forward_propagation(X, parameters)
    Z1, A1, Z2, A2 = cache['Z1'], cache['A1'], cache['Z2'], cache['A2']
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(A1.T, dZ2) + (lambd / m) * parameters['W2']
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = np.dot(dZ2, parameters['W2'].T) * sigmoid_derivative(Z1)
    dW1 = (1 / m) * np.dot(X.T, dZ1) + (lambd / m) * parameters['W1']
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


# In[ ]:


def train_model(X, Y, num_iterations, learning_rate, lambd, hidden_size):
    input_size = X.shape[1]
    output_size = Y.shape[1]
    parameters = initialize_parameters(input_size, hidden_size, output_size)
    for i in range(num_iterations):
        gradients = compute_gradients(X, Y, parameters, lambd)
        parameters['W1'] -= learning_rate * gradients['dW1']
        parameters['b1'] -= learning_rate * gradients['db1']
        parameters['W2'] -= learning_rate * gradients['dW2']
        parameters['b2'] -= learning_rate * gradients['db2']
        if i % 100 == 0:
            loss = compute_loss(X, Y, parameters, lambd)
            print(f'Iteration {i}, loss: {loss}')
    return parameters
num_iterations = 5000
learning_rate = 0.1
lambd = 0.01
hidden_size = 256

parameters = train_model(X_train, Y_train, num_iterations, learning_rate, lambd, hidden_size)


# In[ ]:


def predict(X, parameters):
    A2 = forward_propagation(X, parameters)['A2']
    return np.argmax(A2, axis=1)

predictions = predict(X_test, parameters)
accuracy = np.mean(predictions == np.argmax(Y_test, axis=1))
print(f'Test accuracy: {accuracy:.2%}')


# In[ ]:


def plot_curve(data, title):
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(title)
    plt.show()

train_losses = [compute_loss(X_train, Y_train, parameters, lambd) for i in range(num_iterations)]
test_losses = [compute_loss(X_test, Y_test, parameters, lambd) for i in range(num_iterations)]
train_accuracies = []
test_accuracies = []

for i in range(num_iterations):
    parameters = train_model(X_train, Y_train, 1, learning_rate, lambd, hidden_size)
    train_predictions = predict(X_train, parameters)
    train_accuracy = np.mean(train_predictions == np.argmax(Y_train, axis=1))
    train_accuracies.append(train_accuracy)
    test_predictions = predict(X_test, parameters)
    test_accuracy = np.mean(test_predictions == np.argmax(Y_test, axis=1))
    test_accuracies.append(test_accuracy)

plot_curve(train_losses, 'Train Loss')
plot_curve(test_losses, 'Test Loss')
plot_curve(train_accuracies, 'Train Accuracy')
plot_curve(test_accuracies, 'Test Accuracy')


# In[ ]:


def plot_weights(W):
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow(W[:, i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()

plot_weights(parameters['W1'])

