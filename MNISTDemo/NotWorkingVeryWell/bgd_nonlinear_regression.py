import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

print("Finished loading!")


# want to learn this
coeffs = np.array([0.0113, 0.57, 3, 5])
xdim = coeffs.shape[0]

def f(x, noise=0):
    return coeffs[0]*x*x + coeffs[1]*x*torch.log(x+1)+ coeffs[2]*x+coeffs[3]+noise

data_set_size = 100
inputs = torch.tensor(np.arange(data_set_size), dtype = torch.double)
noise = torch.tensor(np.random.normal(0,1,data_set_size), dtype = torch.double)
X = torch.zeros((data_set_size, xdim), dtype = torch.double)
X[:,0] = inputs * inputs
X[:,1] = inputs * torch.log(inputs+1)
X[:,2] = inputs
X[:,3] = 1

# Y = f(inputs, noise)
Y = f(inputs)
# A = torch.tensor(np.random.normal(0,0.5,xdim), dtype = torch.double, requires_grad = True)
A = torch.tensor(coeffs, dtype=torch.double, requires_grad = True)

lr = 5e-8
print(A)




n_epochs = 20
batch_size = 20
# import pdb; pdb.set_trace()

for epoch in range(n_epochs):
    perm = torch.randperm(X.shape[0])
    
    yhat = torch.mv(X, A)
    loss = torch.mean((yhat - Y)**2)
    loss.backward()

    if (epoch+1) % 1 == 0:
        print("Epoch", epoch)
        print("\tCoef:", A)
        print("\tPred:", yhat)
        print("\tTrue:", Y)
        print("\tGrad:", A.grad)
        print("\tLoss:", loss)
        print("\n\n")

    with torch.no_grad():
        A -= lr * A.grad

    A.grad.zero_()

    if (epoch+1) % 1000 == 0:
        lr /= 1.1



from sklearn.linear_model import LinearRegression
rm = LinearRegression(fit_intercept=False)
Xx = X.detach().numpy()
Yy = Y.detach().numpy()
rm.fit(Xx, Yy)


print("Ours:", A)
print("sklearn:", rm.coef_)
print("Truth:", coeffs)

myPreds = torch.mv(X, A)
print("RMSE is: ", ((Y-myPreds)**2).mean().detach().numpy())


plt.plot(inputs.detach().numpy(), (torch.mv(X, A)).detach().numpy())
plt.plot(inputs.detach().numpy(), np.dot(X.detach().numpy(), coeffs))
