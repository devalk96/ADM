import math
import random
import numpy as np


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
            # print(f"Epoch round: {epoch_nr}\tOld: {weights}\tNew: {self.weights}")
        print(f"Total epochs: {epoch_nr}")


# Activation
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


if __name__ == '__main__':
    neuron = Neuron(2, linear, mean_squared_error)
