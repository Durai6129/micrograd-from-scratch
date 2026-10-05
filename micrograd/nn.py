import random
from micrograd.engine import Value

class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.random()) for _ in range(n_in)]
        self.b = Value(random.random())

    def __call__(self, x):
        weighted_sum = sum((wi * xi for wi, xi in zip(self.w, x)), self.b).tanh()
        return weighted_sum

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        output = [neuron(x) for neuron in self.neurons]
        return output

    def parameters(self):
        parameters = []
        for neuron in self.neurons:
            parameters.extend(neuron.parameters())
        return parameters


class MLP:
    def __init__(self, n_in, layers):
        self.layers = []
        sizes = [n_in] + layers
        for i in range(len(layers)):
            self.layers.append(Layer(sizes[i],sizes[i+1]))

    def __call__(self,x):
        output = self.layers[0](x)
        for i in range(1,len(self.layers)):
            data = self.layers[i](output)
            output = data
        return output

    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters
