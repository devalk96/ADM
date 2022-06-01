from collections import Counter
from copy import deepcopy
import random
import math
import numpy as np


# Activation

def pre_activation(x_n, w_o, b_o):
    w_adj_xs = [w_oi * x_ni for w_oi, x_ni in zip(w_o, x_n)]
    return sum(w_adj_xs) + b_o


def linear(activation):
    return activation


def hinge(y_hat, y):
    return max(0, 1 - y * y_hat)


def sign(activation):
    return 1 if activation > 0 else -1


def tanh(a):
    return (math.exp(a) - math.exp(-a)) / (math.exp(a) + math.exp(-a))


def mean_squared_error(yhat, y):
    return (yhat - y) ** 2


def mean_absolute_error(yhat, y):
    return math.fabs(yhat - y)


# Derivative
def derivative(function, delta=0.01):
    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    return wrapper_derivative


class LinearRegression():

    def __init__(self, dim):
        self.dim = dim
        self.bias = 0
        self.weights = [0, 0]

    def __repr__(self):
        text = f'LinearRegression(dim={self.dim})'
        return text

    def predict(self, xs):
        result = []
        for char in xs:
            x1 = char[0] * self.weights[0]
            x2 = char[1] * self.weights[1]
            result.append(self.bias + x1 + x2)

        return result

    def partial_fit(self, xs, ys, *, alpha=0.001):
        index = 0
        predicted_value = self.predict(xs)
        for x, y in zip(xs, ys):
            self.bias = self.bias - alpha * (predicted_value[index] - y)
            self.weights[0] = self.weights[0] - alpha * (predicted_value[index] - y) * x[0]
            self.weights[1] = self.weights[1] - alpha * (predicted_value[index] - y) * x[1]
            index += 1

    def check(self, actual, predict):
        wrong = 0
        for x, y in zip(actual, predict):
            if x - y != 0:
                wrong += 1
        if wrong == 0:
            return False
        else:
            return True

    def fit(self, xs, ys, *, alpha=0.001, epochs=1000):
        for i in range(0, epochs):
            predicted_value = self.predict(xs)
            self.partial_fit(xs, ys)
            if self.check(ys, predicted_value) == False:
                break


class Neuron:
    bias = 0

    def __init__(self, dim, activation=None, loss=None):
        self.activation = linear if activation is None else activation
        self.loss = mean_squared_error if loss is None else loss
        self.dim = dim
        self.weights = np.random.uniform(0, 1, dim)

    def __repr__(self):
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text

    def predict(self, xs):
        w_adj_xs = [[wi * xi for wi, xi in zip(self.weights, x)] for x in xs]
        return [self.activation(sum(wx) + self.bias) for wx in w_adj_xs]

    def single_predict(self, x):
        return self.predict([x])[0]

    def partial_fit(self, xs, ys, alpha=0.001):
        for x, y in zip(xs, ys):
            d_loss = derivative(self.loss)
            d_activation = derivative(self.activation)

            d_y_hat = d_activation(self.single_predict(x))
            d_loss = d_loss(self.single_predict(x), y)

            self.bias = self.bias - (alpha * d_loss * d_y_hat)
            self.weights = [wi - (alpha * d_loss * d_y_hat * xi) for wi, xi in zip(self.weights, x)]

    def fit(self, xs, ys, *, epochs=0, alpha=0.001, max_iter=2000):
        if epochs != 0:
            for epoch in range(epochs):
                self.partial_fit(xs, ys, alpha)

        training = True
        epoch_nr = 0

        while training:
            if not training or epoch_nr == max_iter:
                break

            weights = list(self.weights)
            bias = self.bias
            self.partial_fit(xs, ys, alpha)
            if weights == self.weights and bias == self.bias:
                training = False
            epoch_nr += 1
        print(f"Total epochs: {epoch_nr}")


class Layer():
    classcounter = Counter()

    def __init__(self, outputs, name=None, *, next=None):
        Layer.classcounter[type(self)] += 1
        if name is None:
            name = f'{type(self).__name__}_{Layer.classcounter[type(self)]}'
        self.inputs = 0
        self.outputs = outputs
        self.name = name
        self.next = next

    def __call__(self, xs):
        raise NotImplementedError('Abstract __call__ method')

    def __repr__(self):
        text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def add(self, next):
        if self.next is None:
            self.next = next
            next.set_inputs(self.outputs)
        else:
            self.next.add(next)

    def set_inputs(self, inputs):
        self.inputs = inputs

        # Formula initialize weights
        limit = np.sqrt(6 / float(self.inputs + self.outputs))
        weights = np.random.uniform(low=-limit, high=limit, size=(self.outputs, self.inputs))
        self.weights = weights

    def __add__(self, next):
        result = deepcopy(self)
        result.add(deepcopy(next))
        return result

    def __getitem__(self, index):
        if index == 0 or index == self.name:
            return self
        if isinstance(index, int):
            if self.next is None:
                raise IndexError('Layer index out of range')
            return self.next[index - 1]
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]
        raise TypeError(f'Layer indices must be integers or strings, not {type(index).__name__}')


class InputLayer(Layer):

    def __repr__(self):
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def set_inputs(self, inputs):
        return NotImplementedError

    def __call__(self, xs, ys=None):
        return self.next(xs, ys)

    def predict(self, xs, ys=None):
        yhats, ls = self(xs, ys)
        if ys is not None:
            return yhats, ls
        return yhats


class DenseLayer(Layer):

    def __init__(self, outputs, name=None):
        super().__init__(outputs, name)
        self.bias = [random.randint(0, 0) for x in range(self.outputs)]
        self.weights = []

    def __repr__(self):
        text = f'DenseLayer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += '+' + repr(self.next)
        return text

    def __call__(self, xs, ys=None):
        aa = [[pre_activation(x, w_o, b_o) for w_o, b_o in zip(self.weights, self.bias)] for x in xs]
        yhats, ls = self.next(aa, ys)
        return yhats, ls


class ActivationLayer(Layer):

    def __init__(self, outputs, name=None, activation=linear):
        super().__init__(outputs, name)
        self.activation = activation

    def __repr__(self):
        text = (f'ActivationLayer(inputs={self.inputs}, outputs={self.outputs}, '
                f'activation= {self.activation.__name__}, name={repr(self.name)})'
                )
        if self.next is not None:
            text += '+' + repr(self.next)
        return text

    def __call__(self, aa, ys=None):
        hs = [[self.activation(a_ni) for a_ni in a_n] for a_n in aa]
        yhats, ls = self.next(hs, ys)
        return yhats, ls


class LossLayer(Layer):


    def __init__(self, loss=mean_squared_error, name=None):
        super().__init__(self, name)
        self.loss = loss

    def add(self):
        return NotImplementedError

    def set_inputs(self, inputs):
        return NotImplementedError

    def __repr__(self):
        text = f'LossLayer(inputs={self.inputs}, loss={self.loss.__name__}, name={repr(self.name)})'
        return text


    def __call__(self, xs, ys=None):
        yhats = xs
        ls = None
        if ys is not None:
            print("YS is not none")
            ls = [self.loss(y_cap, y) for y_cap, y in zip(xs, ys)]
        return yhats, ls