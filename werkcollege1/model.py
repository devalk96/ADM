import data

xs, ys = data.linear('nominal')


class Perceptron:
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0
        self.weights = [0 for _ in range(dim)]

    def __repr__(self):
        return f"Perceptron dim = {self.dim}"

    def predict(self, xs):
        y = []
        for instance in xs:
            a = 0
            for ind in range(self.dim):
                w = self.weights[ind]
                x = instance[ind]
                a += w * x
            y.append(1 if a + self.bias > 0 else - 1)
        return y

    def partial_fit(self, xs, ys):
        for x, y in zip(xs, ys):
            activation = self.bias + sum([wi * xi for wi, xi in zip(self.weights, x)])
            y_hat = 1 if activation > 0 else -1

            self.weights = [wi-xi * (y_hat - y) for wi, xi in zip(self.weights, x)]
            self.bias = self.bias - (y_hat - y)

    def fit(self, xs, ys, *, epochs=0):
        if epochs != 0:
            for epoch in range(epochs):
                self.partial_fit(xs, ys)

        training = True
        epoch_nr = 0

        while training:
            if not training:
                break
            weights = list(self.weights)
            bias = self.bias
            self.partial_fit(xs, ys)
            if weights == self.weights and bias == self.bias:
                training = False
            epoch_nr += 1
            # print(f"Epoch round: {epoch_nr}\tOld: {weights}\tNew: {self.weights}")
        print(f"Total epochs: {epoch_nr}")

class LinearRegression():
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0
        self.weights = [0 for _ in range(dim)]

    def __repr__(self):
        return f"Perceptron dim = {self.dim}"

    def predict(self, xs):
        y = []
        for instance in xs:
            a = 0
            for ind in range(self.dim):
                w = self.weights[ind]
                x = instance[ind]
                a += w * x
            y.append(a + self.bias)
        return y

    def partial_fit(self, xs, ys, alpha=0.1):
        for x, y in zip(xs, ys):
            activation = self.bias + sum([wi * xi for wi, xi in zip(self.weights, x)])
            y_hat = activation

            self.weights = [wi - xi * (alpha * (y_hat - y)) for wi, xi in zip(self.weights, x)]
            self.bias = self.bias - (alpha * (y_hat - y))

    def fit(self, xs, ys, *, epochs=0, alpha=0.1, max_iter=1000):
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

if __name__ == '__main__':
    pass
