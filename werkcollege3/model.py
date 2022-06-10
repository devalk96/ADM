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


def tanh(activation):
    tan = math.tanh(activation)
    return tan


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

    def partial_fit(self, xs, ys, alpha=0.0001):
        for x, y in zip(xs, ys):
            d_loss = derivative(self.loss)
            d_activation = derivative(self.activation)

            d_y_hat = d_activation(self.single_predict(x))
            d_loss = d_loss(self.single_predict(x), y)

            self.bias = self.bias - (alpha * d_loss * d_y_hat)
            self.weights = [wi - (alpha * d_loss * d_y_hat * xi) for wi, xi in zip(self.weights, x)]

    def fit(self, xs, ys, *, epochs=0, alpha=0.1, max_iter=2000):
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
        self.prev_layer = None
        self.curr_layer = self

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
            next.set_prev_layer(self.curr_layer)
        else:
            self.next.add(next)

    def set_inputs(self, inputs):
        self.inputs = inputs

        # Formula initialize weights
        limit = np.sqrt(6 / float(self.inputs + self.outputs))
        weights = np.random.uniform(low=-limit, high=limit, size=(self.outputs, self.inputs))
        self.weights = weights

    def set_prev_layer(self, prev_layer):
        self.prev_layer = prev_layer

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

    def __call__(self, xs, *, ys=None, alpha=None):
        return self.next(xs, ys=ys, alpha=alpha)

    def predict(self, xs):
        hs, _, _ = self(xs)
        return hs

    def evaluate(self, xs, ys):
        _, ls, _ = self(xs, ys=ys)
        return sum(ls) / len(ls)

    def partial_fit(self, xs, ys, alpha=0.1):
        return self(xs, ys=ys, alpha=alpha)

    def fit(self, xs, ys, alpha=0.1, epochs=None):
        if not epochs:
            for _ in range(100):
                self.partial_fit(xs, ys, alpha=alpha)
        else:
            for _ in range(epochs):
                self.partial_fit(xs, ys, alpha=alpha)


class DenseLayer(Layer):
    def __init__(self, outputs, name=None):
        super().__init__(outputs, name)
        self.bias = [random.randint(0, 0) for _ in range(self.outputs)]
        self.weights = []

    def __repr__(self):
        text = f'DenseLayer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += '+' + repr(self.next)
        return text

    def __call__(self, xs, *, ys=None, alpha=None):

        # Set initial weight and bias values
        if self.bias is None and self.weights is None:
            low = math.sqrt(6 / (self.inputs + self.outputs)) * -1
            high = math.sqrt(6 / (self.inputs + self.outputs))

            self.bias = [np.random.uniform(low, high)
                         for _ in range(self.outputs)]

            self.weights = [np.random.uniform(low, high,
                                              self.inputs).tolist()
                            for _ in range(self.outputs)]

        # Calculate pre-activation values
        aa = [[pre_activation(x_n, w_o, b_o)
               for w_o, b_o in zip(self.weights, self.bias)]
              for x_n in xs]

        # Loss over bias gradients per output per instance
        hs, ls, df_la = self.next(aa, ys=ys, alpha=alpha)

        # Predicting/Evaluating
        if alpha is None:
            return hs, ls, None

        # Learning
        alpha_n = alpha / len(xs)

        # Loss over input gradients per output per instance
        df_lx = [[sum([df_la_n[o] * self.weights[o][i]
                       for o in range(self.outputs)]) for i in range(self.inputs)] for df_la_n in
                 df_la]

        # Adjust weights and bias
        for df_la_n, x_n in zip(df_la, xs):
            self.weights = [[w_oi - (alpha_n * df_la_no * x_ni) for w_oi, x_ni in zip(w_o, x_n)]
                            for df_la_no, w_o in zip(df_la_n, self.weights)]

            self.bias = [b_o - (alpha_n * df_la_no)
                         for b_o, df_la_no in zip(self.bias, df_la_n)]

        return hs, ls, df_lx


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

    def __call__(self, aa, *, ys=None, alpha=None):
        # Calculate predictions
        hs = [[self.activation(a_ni) for a_ni in a_n] for a_n in aa]

        # Average loss over prediction gradients per instance
        hs, ls, df_lh = self.next(hs, ys=ys, alpha=alpha)

        # Predicting/Evaluating
        if alpha is None:
            return hs, ls, None

        # Average prediction over pre-activation gradients per instance
        df_activation = derivative(self.activation)
        df_ha = [[df_activation(a_ni)
                  for a_ni in a_n] for a_n in aa]

        # Average loss over pre-activation gradients per instance
        df_la = [[df_lh_ni * df_ha_ni
                  for df_lh_ni, df_ha_ni in zip(df_lh_n, df_ha_n)]
                 for df_lh_n, df_ha_n in zip(df_lh, df_ha)]

        # Learning
        return hs, ls, df_la


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

    def __call__(self, hs, *, ys=None, alpha=None):

        # Predicting
        if ys is None:
            return hs, None, None

        # Average loss per instance
        ls = [sum([self.loss(h_ni, y_ni)
                   for h_ni, y_ni in zip(h_n, y_n)])
              for h_n, y_n in zip(hs, ys)]

        # Evaluating
        if alpha is None:
            return hs, ls, None

        # Loss over prediction gradients per instance
        df_loss = derivative(self.loss)
        df_lh = [[df_loss(h_ni, y_ni)
                  for h_ni, y_ni in zip(h_n, y_n)]
                 for h_n, y_n in zip(hs, ys)]

        # Learning
        return hs, ls, df_lh
