from __future__ import division
from scipy.signal import convolve2d, correlate2d

import numpy as np
import numpy.random as random

def _random_weights(shape):
    size = np.prod(shape)
    #random.seed(42) # use only for debugging
    return random.uniform(-np.sqrt(1/size), np.sqrt(1/size), shape)

class Layer:
    def __init__(self, config):
        assert(config["input_shape"] and config["l2_decay"] is not None and config["debug"] is not None)
        self.input_shape = config["input_shape"]
        self.l2_decay = config["l2_decay"]
        self.debug = config["debug"]

    def forward(self, buffer):
        return buffer

    def backward(self, input, buffer):
        return buffer

    def update(self, rate):
        pass

    def predict(self, input):
        return self.forward(input)

    def loss(self):
        return 0

class InputLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)
        self.output_shape = self.input_shape

class FullyConnectedLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)

        assert(config["neurons"])
        neurons = config["neurons"]
        self.neurons = neurons
        self.output_shape = (neurons, )
        self.weights = np.c_[_random_weights((neurons, np.prod(self.input_shape))), np.zeros(neurons)]

    def forward(self, input, weights=None):
        weights = weights if weights is not None else self.weights # allow to overwrite weights for testing purposes
        return np.dot(weights, np.append(input.reshape(-1), 1))

    def backward(self, input, parent_gradient):
        self.dweights = np.c_[np.tile(input.reshape(-1), (self.neurons, 1)), np.ones(self.neurons)]

        if self.debug:
            numerical =  self.numerical_gradient(input, self.weights)
            if not np.all(np.abs(numerical - self.dweights).reshape(-1) <= 0.00001):
                print "Numerical Gradient:\n{}\nAnalytical Gradient:\n{}".format(numerical, self.dweights)
                assert(False)

        decay = self.l2_decay*self.weights
        decay[:, -1] = 0 # don't apply decay to bias

        self.dweights = self.dweights*parent_gradient[:, None] + decay # loss gradient wrt. weights
        return parent_gradient.dot(self.weights)[:-1].reshape(self.input_shape) # loss gradient wrt. input

    def update(self, rate):
        self.weights = self.weights - self.dweights*rate

    def loss(self):
        return self.l2_decay*(np.square(self.weights.reshape(-1)[:-1])).sum()/2

    def numerical_gradient(self, input, params):
        eps = 0.000001
        pert = params.copy()
        res = np.zeros(shape=params.shape)

        for index, x in np.ndenumerate(params):
            neuron = index[0]
            pert[index] = params[index] + eps
            res[index] = (self.forward(input, pert)[neuron] - self.forward(input, params)[neuron])/eps
            pert[index] = params[index]

        return res

class ReLuLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)
        self.output_shape = self.input_shape

    def forward(self, buffer):
        # See http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
        # about caveats of ReLu, i.e. "it could lead to cases where a unit never activates as a gradient-based
        # optimization algorithm will not adjust the weights of a unit that never activates initially"
        return np.where(buffer < 0, 0.01*buffer, buffer)

    def backward(self, input, buffer):
        return np.where(input < 0, 0.01, 1)*buffer

class DropoutLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)
        self.output_shape = self.input_shape
        self.prob = 0.5

    def forward(self, input):
        self.rnd = random.binomial(1, self.prob, input.size)
        self.rnd = self.rnd.reshape(input.shape)
        return input*self.rnd

    def backward(self, input, parent_gradient):
        return parent_gradient*self.rnd

    def predict(self, input):
        assert(not self.debug) # TODO: make it work in debug mode
        return input*self.prob # approximates the geometric mean

class ConvolutionLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)
        assert(config["filters"] > 0 and config["size"] > 0)
        assert(len(self.input_shape) >= 2) # Assume 2D Matrices as input

        size = config["size"]

        self.n_filters = config["filters"]
        self.n_input_maps = 1 if len(self.input_shape) == 2 else self.input_shape[0]
        self.output_shape = (self.n_filters, self.input_shape[-2] - size + 1, self.input_shape[-1] - size + 1)
        self.filter = _random_weights((self.n_filters, size, size))

    def forward(self, imgs):
        imgs = imgs.reshape(self.n_input_maps, imgs.shape[-2], imgs.shape[1])
        return self._convolve(imgs, self.filter)

    def _convolve(self, imgs, filters):
        assert(imgs.ndim == 3 and filters.ndim == 3)
        assert(imgs.shape[-2] >= filters.shape[-2] and imgs.shape[-1] >= filters.shape[-1])
        assert(filters.shape[-2] == filters.shape[-1] and filters.shape[-1] % 2 != 0)

        lx = filters.shape[-1]//2
        rx = imgs.shape[-1] - lx - 1
        ly = lx
        ry = imgs.shape[-2] - ly - 1
        output = np.zeros((filters.shape[0], rx - lx + 1, ry - ly + 1))

        for f in range(0, filters.shape[0]):
            filter = filters[f]
            filter_map = np.zeros((rx - lx + 1, ry - ly + 1))

            for i in range(0, imgs.shape[0]):
                img = imgs[i]
                convolved = np.zeros((rx - lx + 1, ry - ly + 1))

                for x in range(lx, rx + 1):
                    for y in range(ly, ry + 1):
                        subimg = img[y - ly:y + ly + 1:,x - lx:x + lx + 1]
                        convolved[y - ly, x - lx] = (subimg * filter).sum()

                if self.debug:
                    lib_convolved = correlate2d(img, filter, "valid")
                    if not np.all(np.abs(convolved - lib_convolved) < 0.000001):
                        print "Convolved:\n{}\nLib Convolved:\n{}\nFilter:\n{}".format(convolved, lib_convolved, filter)
                        assert(False)

                filter_map += convolved
            output[f]=filter_map

        return output

    def backward(self, imgs, parents_gradient):
        imgs = imgs.reshape(self.n_input_maps, imgs.shape[-2], imgs.shape[1])
        input_gradient, dfilter = self._gradient(imgs, self.filter, parents_gradient)
        self.dfilter = dfilter
        return input_gradient

    def _gradient(self, imgs, filters, parents_gradient):
        assert(imgs.ndim == 3 and filters.ndim == 3)
        assert(imgs.shape[-2] >= filters.shape[-2] and imgs.shape[-1] >= filters.shape[-1])
        assert(filters.shape[-2] == filters.shape[-1] and filters.shape[-1] % 2 != 0)

        lx = filters.shape[-1]//2
        rx = imgs.shape[-1] - lx - 1
        ly = lx
        ry = imgs.shape[-2] - ly - 1
        imgs_gradient = np.zeros(imgs.shape)
        filters_gradient = np.zeros(filters.shape)

        for f in range(0, filters.shape[0]):
            filter = filters[f]
            filter_gradient = filters_gradient[f]
            parent_gradient = parents_gradient[f]

            for i in range(0, imgs.shape[0]):
                img = imgs[i]
                img_gradient = imgs_gradient[i]

                for x in range(lx, rx + 1):
                    for y in range(ly, ry + 1):
                        img_gradient[y - ly:y + ly + 1:,x - lx:x + lx + 1] += filter*parent_gradient[y - ly, x - lx]
                        filter_gradient += img[y - ly:y + ly + 1:,x - lx:x + lx + 1]*parent_gradient[y - ly, x - lx]

            filter_gradient += self.l2_decay*filter

        return (imgs_gradient, filters_gradient)

    def update(self, rate):
        self.filter = self.filter - self.dfilter*rate

    def loss(self):
        return self.l2_decay*(np.square(self.filter.reshape(-1))).sum()/2

class PoolingLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)
        assert(config["size"] > 0)
        assert(len(self.input_shape) == 3)

        self.size = config["size"]
        self.output_shape = (self.input_shape[0],
                             (self.input_shape[1] - self.size)//self.size + 1,
                             (self.input_shape[2] - self.size)//self.size + 1)

    def forward(self, imgs):
        assert(imgs.ndim == 3)
        maps = np.zeros(self.output_shape)

        for i in range(0, imgs.shape[0]):
            img = imgs[i]
            map = maps[i]

            for x in range(0, self.output_shape[1]):
                x_img = x*self.size

                for y in range(0, self.output_shape[2]):
                    y_img = y*self.size
                    map[y][x] = img[y_img:y_img+self.size, x_img:x_img+self.size].max()

        return maps

    def backward(self, imgs, parents_gradient):
        imgs_gradient = np.zeros(self.input_shape)

        for i in range(0, imgs.shape[0]):
            img = imgs[i]
            img_gradient = imgs_gradient[i]
            parent_gradient = parents_gradient[i]

            for x in range(0, self.output_shape[1]):
                x_img = x*self.size

                for y in range(0, self.output_shape[2]):
                    y_img = y*self.size

                    sub = img[y_img:y_img+self.size, x_img:x_img+self.size]
                    sub_max_index = np.unravel_index(sub.argmax(), sub.shape)
                    max_index = np.add(sub_max_index, (y_img, x_img))
                    img_gradient[tuple(max_index)] = parent_gradient[y, x]

        return imgs_gradient

class SquaredLossLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)

    def forward(self, buffer):
        return buffer

    def backward(self, input, expected):
        if np.isscalar(expected):
            expected = np.array([expected])

        assert(input.shape == expected.shape)
        return input - expected

    def loss(self, predicted, expected):
        if np.isscalar(expected):
            expected = np.array([expected])
        assert(predicted.shape == expected.shape)
        return np.square(predicted - expected).sum()*0.5

class SoftmaxLossLayer(Layer):
    def __init__(self, config):
        assert(config["categories"] > 0)
        assert(config["categories"] == config["input_shape"][0] and len(config["input_shape"]) == 1)
        Layer.__init__(self, config)
        self.categories = config["categories"]

    def forward(self, buffer):
        max = np.max(buffer)
        exp = np.exp(buffer - max) # numerically stable
        total = exp.sum()
        return exp/total

    def backward(self, input, expected):
        assert(expected.dtype.kind == 'i')
        output = self.forward(input)
        mask = np.zeros(shape=input.shape)
        mask[expected] = 1
        return output - mask

    def loss(self, predicted, expected):
        output = predicted[expected]
        return -np.log(output)
