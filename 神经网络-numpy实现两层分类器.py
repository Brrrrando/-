
# coding: utf-8

# In[1]:


def load_mnist():
    # 设置MNIST数据集文件所在的路径
    mnist_dir = r'C:\Users\86138\mnist_data'

    # 加载训练集数据
    with gzip.open(r'C:\Users\86138\mnist_data\train-images-idx3-ubyte.gz', 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    with gzip.open(r'C:\Users\86138\mnist_data\train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    # 加载测试集数据
    with gzip.open(r'C:\Users\86138\mnist_data\t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    with gzip.open(r'C:\Users\86138\mnist_data\t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    # 返回数据集
    return (train_images, train_labels), (test_images, test_labels)


# In[2]:


import gzip
import numpy as np


# In[3]:


(train_images, train_labels), (test_images, test_labels) = load_mnist()
train_images = np.array(train_images)
test_images = np.array(test_images)
X_train = train_images
Y_train = train_labels


# In[4]:


def normalize_images(images):
    return images / 255.0

def one_hot_labels(labels, num_classes):
    return np.eye(num_classes)[labels]

num_classes = 10
train_images = normalize_images(train_images)
test_images = normalize_images(test_images)
train_labels = one_hot_labels(np.array(train_labels, dtype=int), num_classes)
test_labels = one_hot_labels(np.array(test_labels, dtype=int), num_classes)


# In[12]:


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameters


# In[13]:


def forward_propagation(X, parameters):
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return cache


# In[14]:


def compute_cost(X, Y, parameters, lambd):
    m = X.shape[0]
    W1, W2 = parameters['W1'], parameters['W2']
    cache = forward_propagation(X, parameters)
    A2 = cache['A2']
    cross_entropy_loss = -np.sum(Y * np.log(A2 + 1e-10) + (1 - Y) * np.log(1 - A2 + 1e-10)) / m
    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2))) / (2 * m)
    cost = cross_entropy_loss + L2_regularization_cost
    return cost

def backward_propagation(X, Y, parameters, cache, lambd):
    m = X.shape[0]
    W1, W2 = parameters['W1'], parameters['W2']
    Z1, A1, Z2, A2 = cache['Z1'], cache['A1'], cache['Z2'], cache['A2']
    dZ2 = A2 - Y
    dW2 = (1 / m) * (np.dot(A1.T, dZ2) + lambd * W2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = np.dot(dZ2, W2.T) * (Z1 > 0)
    dW1 = (1 / m) * (np.dot(X.T, dZ1) + lambd * W1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
    gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return gradients


# In[15]:


(train_images, train_labels), (test_images, test_labels) = load_mnist()
train_images = np.array(train_images)
test_images = np.array(test_images)
X_train = train_images
Y_train = train_labels
X_train = X_train.astype('float64')
Y_train = Y_train.astype('float64')
X_test = test_images
Y_test = test_labels
X_test = X_test.astype('float64')
Y_test = Y_test.astype('float64')


# In[20]:


def train_model_sgd(X, Y, num_iterations, learning_rate, lambd, hidden_size, batch_size):
    X = X.astype('float64')
    Y = Y.astype('float64').reshape((-1, 1))  # 将Y还原为列向量
    input_size = X.shape[1]
    output_size = Y.shape[1]  # Y的列数为1
    parameters = initialize_parameters(input_size, hidden_size, output_size)
    m = X.shape[0]
    for i in range(num_iterations):
        # 随机打乱数据集
        permutation = np.random.permutation(m)
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :]
        # 分成若干个batch
        num_batches = m // batch_size
        for j in range(num_batches):
            start = j * batch_size
            end = (j + 1) * batch_size
            batch_X = shuffled_X[start:end, :]
            batch_Y = shuffled_Y[start:end, :]
            # 计算梯度并更新模型参数
            cache = forward_propagation(batch_X, parameters)
            gradients = backward_propagation(batch_X, batch_Y, parameters,cache,lambd)
            parameters['W1'] -= learning_rate * gradients['dW1']
            parameters['b1'] -= learning_rate * gradients['db1']
            parameters['W2'] -= learning_rate * gradients['dW2']
            parameters['b2'] -= learning_rate * gradients['db2']
        # 计算整个训练集上的损失函数值
        loss = compute_cost(X, Y, parameters, lambd)
        if i % 10 == 0:
            print(f'Iteration {i}, loss: {loss}')
    return parameters

num_iterations = 1000
learning_rate = 0.1
lambd = 0.01
hidden_size = 1
batch_size = 64

parameters = train_model_sgd(X_train, Y_train, num_iterations, learning_rate, lambd, hidden_size, batch_size)


# In[21]:


def predict(X, parameters):
    A2 = forward_propagation(X, parameters)['A2']
    return np.argmax(A2, axis=1)

predictions = predict(X_test, parameters)
accuracy = np.mean(predictions == np.argmax(Y_test, axis=1))
print(f'Test accuracy: {accuracy:.2%}')


# In[23]:


Y_test = Y_test.reshape((-1, 1))


# In[24]:


predictions = predict(X_test, parameters)
Y_test_labels = np.argmax(Y_test, axis=1)
accuracy = np.mean(predictions == Y_test_labels)
print(f'Test accuracy: {accuracy:.2%}')


# In[26]:


def plot_curve(data, title):
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(title)
    plt.show()

train_losses = [compute_cost(X_train, Y_train, parameters, lambd) for i in range(num_iterations)]
test_losses = [compute_cost(X_test, Y_test, parameters, lambd) for i in range(num_iterations)]
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


# In[27]:


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


# In[28]:


print(parameters['W1'])

