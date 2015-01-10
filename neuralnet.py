from __future__ import division
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from scipy.signal import convolve2d, correlate2d
from layers import InputLayer, FullyConnectedLayer, ReLuLayer, DropoutLayer, \
                   ConvolutionLayer, PoolingLayer, SquaredLossLayer, SoftmaxLossLayer

import numpy as np

class NeuralNet:
    def __init__(self, layers, l2_decay=0.001, debug=False, learning_rate=0.001):
        mapping = {"input": lambda x: InputLayer(x),
                   "fc": lambda x: FullyConnectedLayer(x),
                   "convolution": lambda x: ConvolutionLayer(x),
                   "pool": lambda x: PoolingLayer(x),
                   "squaredloss": lambda x: SquaredLossLayer(x),
                   "softmax": lambda x: SoftmaxLossLayer(x),
                   "relu": lambda x: ReLuLayer(x),
                   "dropout": lambda x: DropoutLayer(x)}
        self.layers = []
        self.l2_decay = l2_decay
        self.debug = debug
        self.learning_rate = learning_rate
        prev = None

        np.seterr(all="warn")

        for layer in layers:
            layer["input_shape"] = layer.get("input_shape", None) or prev.output_shape
            layer["l2_decay"] = layer.get("l2_decay", None) or self.l2_decay
            layer["debug"] = self.debug
            layer = mapping[layer["type"]](layer)
            self.layers.append(layer)
            prev = layer

    def forward(self, input):
        inputs = [input]

        for layer in self.layers:
            assert(layer.input_shape == inputs[-1].shape)
            inputs.append(layer.forward(inputs[-1]))

        return inputs

    def backward(self, inputs, parent_gradient):
        gradients = [parent_gradient]

        for input, layer in zip(inputs[:-1][::-1], self.layers[::-1]):
            gradients.append(layer.backward(input, gradients[-1]))

        return gradients

    def update(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def loss(self, input, expected):
        prediction = self.predict(input)
        loss = self.layers[-1].loss(prediction, expected)

        for layer in self.layers[:-1][::-1]:
            loss += layer.loss() # regularization terms

        return loss

    def predict(self, buffer):
        inputs = [buffer]

        for layer in self.layers:
            assert(layer.input_shape == inputs[-1].shape)
            inputs.append(layer.predict(inputs[-1]))

        return inputs[-1]

    def train(self, X, y, n_epochs=10, n_samples=None):
        for epoch in range(0, n_epochs):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            for i in indices[:n_samples]:
                self._train(X[i], y[i])

    def _train(self, x, y):
        inputs = self.forward(x)
        gradients = self.backward(inputs, y)

        if self.debug:
            numerical = self.numerical_gradient(x, y)
            if not np.all(abs(numerical - gradients[-1]) < 0.00001):
                print "Numerical gradient:\n {}\nAnalytical gradient:\n {}".format(numerical, gradients[-1])
                print "loss: {}\n".format(self.loss(x, y))
                assert(False)

        self.update(self.learning_rate)

    def numerical_gradient(self, input, expected):
        eps = 0.000001
        pert = input.copy()
        res = np.zeros(shape=input.shape)

        for index, x in np.ndenumerate(input):
            pert[index] = input[index] + eps
            res[index] = (self.loss(pert, expected) - self.loss(input, expected))/eps
            pert[index] = input[index]

        return res

def _error(X, y):
    assert(len(X) == len(y))
    mispred = 0
    for i in range(0, len(X)):
        mispred += np.argmax(net.predict(X[i])) != y[i]
    return 100*mispred/len(X)

if __name__ == "__main__":
    n_classes = 10
    net = NeuralNet([{"type": "input", "input_shape": (8, 8)},
                     {"type": "convolution", "filters": 5, "size": 3},
                     {"type": "dropout"},
                     {"type": "relu"},
                     {"type": "pool", "size": 2},
                     {"type": "fc", "neurons": 100},
                     {"type": "dropout"},
                     {"type": "relu"},
                     {"type": "fc", "neurons": n_classes},
                     {"type": "relu"},
                     {"type": "softmax", "categories": n_classes}])

    digits = load_digits(n_class=n_classes)
    data = digits.images.reshape(len(digits.images), 64)
    target = digits.target

    train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=0.9)

    train_data = train_data.reshape((len(train_data), 8, 8))
    test_data = test_data.reshape((len(test_data), 8, 8))

    net.train(train_data, train_target, n_epochs=60)
    print "Train Error: {:.2f}%".format(_error(train_data, train_target))
    print "Test Error: {:.2f}%".format(_error(test_data, test_target))
