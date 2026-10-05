from micrograd.engine import Value

a = Value(6)
b = Value(2)
c = a / b
c.backward()
print(a.grad)
print(b.grad)