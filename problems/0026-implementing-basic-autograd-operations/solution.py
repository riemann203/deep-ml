class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        def fmt(x):
            return int(x) if float(x).is_integer() else round(x, 4)
        return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(data = self.data + other.data, _children = (self, other), _op = '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(data = self.data * other.data, _children = (self, other), _op = '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(data = max(0, self.data), _children=(self,), _op = 'relu')

        def _backward():
            if self.data > 0:
                self.grad += out.grad

        out._backward = _backward

        return out

    def backward(self):
        visited = set()
        topo: list[Value] = []

        def dfs(node: Value):
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                dfs(child)
            topo.append(node)

        dfs(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()
