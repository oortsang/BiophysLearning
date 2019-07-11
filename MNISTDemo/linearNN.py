print("Loading...")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
print("... finished l0ad1ng!")

# torch.random.manual_seed(17)

###  Initializing data set  ###
print("Initializing data set...")
N = 1000 # training examples
coeffs = torch.tensor([1, -3, 0, 0, 15, 0.34, 0], dtype=torch.double)
fdim = coeffs.shape[0]
def f(x, coefficients = coeffs, noise = 0):
    """ Computes the linear combination weighted by the secret coeffs coefficients """
    return (x * coefficients).sum(x.dim() - 1) + noise # sum over last dimension

X = torch.tensor(np.random.normal(0, 10, (N, fdim)), dtype=torch.double)
noise = torch.tensor(np.random.normal(0, 1, N))
Y = f(X, noise=noise)

###  Setting up neural network  ###
print("Setting up Neural Network...")
class MyNN(nn.Module):
    def __init__(self, insize=fdim):
        super().__init__()
        self.linear1 = nn.Linear(in_features = insize, out_features = 1, bias=None)
    def forward(self, xs):
        if xs.dim == 2:
            res = torch.zeros(xs.shape[1], dtype=torch.double)
            for i in range(xs.shape[0]):
                res[i] = self.linear1(xs)
            return res
        else:
            return self.linear1(xs)

modd = MyNN(fdim)
modd.double()

###  Train time!  ###
print("Training Neural Network...")
n_epochs = 20
batch_size = 100
ll = 0.0005
loss_fn = lambda yy, ytrue: torch.mean((yy-ytrue)**2) + ll*(next(modd.parameters())[0]**2).sum()
# loss_fn = nn.MSELoss()
lr = 3e-4
optimizer = optim.SGD(modd.parameters(), lr=lr, momentum=0.1)

for epoch in range(n_epochs):
    perm = torch.randperm(N)
    for i in range(0, N, batch_size):
        idcs = perm[i: i+batch_size] #indices within the selected minibatch
        optimizer.zero_grad()
        Xbatch = X[idcs]
        Ybatch = Y[idcs]
        # import pdb; pdb.set_trace()
        yy = modd(Xbatch).reshape(batch_size)
        loss = loss_fn(yy, Ybatch)
        print(loss)
        loss.backward()
        optimizer.step()

###  Testing time!  ###
print("Parameters:")
for p in modd.parameters():
    for i in range(p.shape[1]):
        print(p[0,i].detach().numpy(), "\t vs \t", coeffs[i].detach().numpy())


testsize = 100
TestSet = torch.tensor(np.random.normal(0, 10, (testsize, fdim)), dtype=torch.double)
noise = torch.tensor(np.random.normal(0, 1, testsize))
TestAnswers = f(TestSet, noise=noise)

MyGuesses = modd(TestSet).reshape(testsize)
print("RMSE: ", loss_fn(MyGuesses, TestAnswers)/testsize)
