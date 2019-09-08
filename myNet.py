from torchvision import datasets, transforms
import torch
batch_size = 15
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
import random
import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1)for y in sizes[1:]]
        self.weights = [np.random.randn(x, y)for x, y in zip(sizes[:-1], sizes[1:])]
    def sigmoid(self,z):
        return 1/(1.0+np.exp(-z))
    def feedforward(self, a):
        a = a.reshape(-1)
        # print(a)
        for b, w in zip(self.biases, self.weights):
            a = np.dot(a,w)
            b = b.reshape(a.shape)
            a = self.sigmoid(a+b)
        return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        return sum(int(x == np.argmax(y)) for (x, y) in test_results)
    def SGD(self,training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[begin:begin+mini_batch_size]\
                for begin in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
    def cost_derivative(self, output_activation, y):
        return (output_activation - y)
    def sigmoid_prime(self,z):
        return self.sigmoid(z) * (1-self.sigmoid(z))
    def backprop(self, x, y):
        """
        x0 z0=x0w0+b x1=sigmoid(z0) x1 z1=x1w1+b x2=sigmoid(z1)...loss=.5(xn-y)^2
        loss'xn = (xn-y)
        xn'zn-1 = sig'
        zn-1'wn-1 = xn-1
        zn-1'xn-1 = xn-1
        """
        nabla_b = [np.zeros(b.shape)for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.biases]
        activation = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            # 收集原料
            z = np.dot(activation[-1],w).reshape(1,-1)+b.reshape(1,-1)
            zs.append(z)
            # print('x{0}w{1}+b='.format(activation[-1].shape,w.shape))
            # print(z.shape)
            activation.append(self.sigmoid(z))
        delta = self.cost_derivative(activation[-1], y.numpy())[0] * self.sigmoid_prime(zs[-1])[0]
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(nabla_b[-1].reshape(-1,1), activation[-2]).transpose()

        for l in range(len(nabla_b)-2,-1,-1):
            nabla_b[l] = np.dot(nabla_b[l+1].transpose(), self.weights[l+1].transpose())
            nabla_b[l] = nabla_b[l]*self.sigmoid_prime(zs[l])[0]

            nabla_w[l] = np.dot(activation[l].reshape(-1,1), nabla_b[l].reshape(1,-1))

            # nabla_w[l] = nabla_b[l] * zs[l]
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # print(len(mini_batch))
        for x, y in mini_batch:
            x = x.reshape(-1)
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            delta_nabla_b = np.array(delta_nabla_b)
            nabla_b = np.array(nabla_b)
            nabla_b = [nb.reshape(-1,1) + dnb.reshape(-1,1) for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

net = Network([784,30,10])
def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]
for x, y in train_loader:

    # y = torch.zeros(len(y), 10).scatter_(1, y, 1)
    y = one_hot(y, 10)
    train_data = zip(x, y)
    train_data = list(train_data)
    net.SGD(train_data, 5, batch_size, 3.0, test_data=train_data)