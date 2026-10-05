import math

class Value:
    def __init__(self, data, _op = ''):
        self.data = data
        self.grad = 0
        self._prev = set() # we only care about unique nodes, same parent might appear mutiple times.
        self._op = _op
        self._backward = lambda : None

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        child = Value(self.data + other.data, _op = '+')
        child._prev = {self, other}
        def _backward():
            self.grad += child.grad * 1
            other.grad += child.grad * 1
        child._backward = _backward
        return child

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        child = Value(self.data * other.data, _op = '*')
        child._prev = {self, other}
        def _backward():
            self.grad += child.grad * other.data
            other.grad += child.grad * self.data
        child._backward = _backward
        return child

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            return 
        child = Value(self.data ** other)
        child._prev = {self}
        child._op = "**"
        def _backward():
            self.grad += other * self.data ** (other-1) * child.grad
        child._backward = _backward
        return child

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def backward(self):
        self.grad = 1
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)
        for node in topo[::-1]:
            node._backward()

    def tanh(self):
        child = Value(math.tanh(self.data))
        child._prev = {self}
        child._op = 'tanh'
        def _backward():
            self.grad += (1 - child.data ** 2) * child.grad
        child._backward = _backward
        return child

    def __repr__(self):
        return f"Value : {self.data}"

    