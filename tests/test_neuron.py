from micrograd.nn import MLP
from micrograd.engine import Value

X = [
    [Value(0), Value(0)],
    [Value(0), Value(1)],
    [Value(1), Value(0)],
    [Value(1), Value(1)],
]

Y = [0, 1, 1, 0]

nn = MLP(2,[4,4,1])


for i in range(1000):
    predict = []
    for x in X:
        predict.append(nn(x)[0])

    loss = 0
    for yp,y in zip(predict,Y):
        loss += (yp - y) ** 2
    if i % 100 == 0:
        print(f"Before Updation Loss: {loss}")

    loss.backward()
    learning_rate = 0.1
    for p in nn.parameters():
        p.data = p.data - learning_rate * p.grad
        p.grad = 0

for x in X:
    print(nn(x))