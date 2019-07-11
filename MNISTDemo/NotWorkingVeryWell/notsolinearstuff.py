# Non-linear problem using a simple neural network

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

print("Finished loading!")

torch.manual_seed(42)
def f(x, noise=0):
#    return 0.02 * x*x + 0.3* x*np.log(x+1) + 1.3*x - 70 + noise
#    return 0.3* x*np.log(x+1) + 1.3*x - 70 + noise
    return 1.3*x + 3

inputs = torch.tensor(np.arange(100), dtype = torch.float32)
noise = torch.tensor(np.random.normal(0,5,100), dtype=torch.float32)
X = torch.zeros((100,3), dtype=torch.float32)

# X[:,0] = inputs
# X[:,1] = 3 * inputs + 7 + noise
# X[:,2] = -14 * inputs
X[:,0] = inputs
X[:,1] = inputs * np.log(inputs)
X[:,2] = inputs * inputs
Y = f(inputs, noise)

idx = np.arange(20)
np.random.shuffle(idx)
train_size = 10
trainees = idx[:train_size]
#testeees = idx[train_size:]



class myNN(nn.Module):
    def __init__(self, isize=3, hsize=3, osize=1):
        super().__init__()
        modules = [ # nn.Linear(isize, hsize),
                    # nn.ReLU(),
                    nn.Linear(hsize, osize)
                  ]
        self.net = nn.Sequential(*modules)
    def forward(self, input):
        return self.net(input)

# doesn't work at the moment
fcnn = myNN()
fcnn = fcnn.float()
optimizer = optim.SGD(fcnn.parameters(), 1e-5)
print("Before step:")
for p in fcnn.parameters():
    print(p)
print("\n\n")
for epoch in range(15):
    optimizer.zero_grad()
    yy = fcnn(X[trainees]).reshape(train_size)
    loss = ((yy - Y[trainees])**2).mean()
    loss.backward()
    optimizer.step()

print("After step:")

for p in fcnn.parameters():
    print(p)


# plt.plot(np.array(inputs), np.array(Y))
# plt.plot(np.array(inputs), np.array(f(inputs)))
# plt.show()
