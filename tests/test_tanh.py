from micrograd.engine import Value

a = Value(1)
b = a.tanh()
b.backward()
print(a.grad)
print(b.grad)