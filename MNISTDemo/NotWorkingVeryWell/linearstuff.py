import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

print("Finished loading!")


# want to learn this
def f(x, noise=0):
    return 3*x+5+noise

data_set_size = 100
inputs = torch.tensor(np.arange(data_set_size), dtype = torch.float32)
noise = torch.tensor(np.random.normal(0,1,data_set_size), dtype=torch.float32)
# X = torch.zeros((data_set_size,2), dtype=torch.float32)
X = torch.zeros((data_set_size), dtype=torch.float32)
X = inputs

Y = f(inputs, noise)


Ym = Y.mean()
Ysd = Y.std()
Xm = X.mean()
Xsd = X.std()

X = (X-Xm)/Xsd
Y = (Y-Ym)/Ysd




A = torch.tensor(np.random.normal(0, 1))
b = torch.tensor(np.random.normal(0, 10))

lr = 1e-4
oldloss = torch.tensor(np.inf)

A.requires_grad_()
b.requires_grad_()

print(A, b)

for epoch in range(20000):
    # yhat = torch.dot(X, A) + b
    yhat = A*X + b
    loss = ((yhat - Y)**2).mean()
    if epoch % 1000 == 0:
        print("Loss:", loss)
    if loss>oldloss+1e-8:
        print("Converged!")
        break
    if epoch % 5000 == 0:
        lr /= 1.1
    loss.backward()
    with torch.no_grad():
        A -= lr * A.grad
        b -= lr * b.grad
    A.grad.zero_()
    b.grad.zero_()
    if torch.isnan(A) or torch.isnan(b):
        print("Oh no! Overflow!")
        break
    oldloss=loss

print(A, b)
