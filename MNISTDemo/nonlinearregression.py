import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

print("Finished loading!")


# want to learn this
# coeffs = np.array([0.0113, 0.37, -0.2, 1])
coeffs = np.array([1, 2, 3, 5])
xdim = coeffs.shape[0]

def f(x, noise=0):
    return coeffs[0]*x*x + coeffs[1]*x*torch.log(x+1)+ coeffs[2]*x+coeffs[3]+noise

data_set_size = 100
inputs = torch.tensor(np.arange(data_set_size), dtype = torch.double)
noise = torch.tensor(np.random.normal(0,1,data_set_size), dtype = torch.double)
X = torch.zeros((data_set_size, xdim), dtype = torch.double)
X[:,0] = inputs * inputs
X[:,1] = inputs * torch.log(inputs+1)
# X[:,1] = 0
X[:,2] = inputs
X[:,3] = 1

# Y = f(inputs, noise)
Y = f(inputs)
A = torch.tensor(np.random.normal(0,1,xdim), dtype = torch.double, requires_grad = True)
# A = torch.tensor(coeffs, dtype=torch.double, requires_grad = True)
# A = torch.tensor([-1, 3, 4, 7], dtype=torch.double, requires_grad = True)

lr = 7e-7
# lr = torch.tensor([1e-8, 1e-7, 1e-6, 1e-5], dtype=torch.double)
print(A)

n_epochs = 2000
batch_size = 4
scale = (batch_size/data_set_size)
# import pdb; pdb.set_trace()


besta = A
bestL = torch.tensor(10000000, dtype=torch.double)
for epoch in range(n_epochs):
    perm = torch.randperm(data_set_size)
    # print("Epoch", epoch)
    for i in range(0, data_set_size, batch_size):
        idcs = perm[i:i+batch_size]
        xb, yb = X[idcs], Y[idcs]
        
        yhat = torch.mv(xb, A)
        loss = torch.sum((yhat - yb)**2)/data_set_size + 0.1*(A*A).sum()
        # loss = torch.sum((yhat - yb)**2)
        loss.backward()
        # summ = 0
        # for j in range(batch_size):
        #     summ += (torch.dot(xb[j], A) - yb[j]) * xb[j]
        # mygrad = 2/batch_size * summ
            
        
        if False and (epoch) % 10 == 0:
            print("\tCoeffs:  ", A)
            print("\tGradient:", A.grad)
            # print("\tMy grad: ", mygrad)
            print("\tLoss:    ", loss)
            print("\tPred:", torch.mv(xb,A))
            print("\tTrue:", yb)
            print("\n\n")

        with torch.no_grad():
            A -= lr * A.grad * scale
            # A -= lr*mygrad
            
            if (loss < bestL):
                bestL = loss
                besta = A

        A.grad.zero_()

        if (epoch+1) % 500 == 0:
            lr *= 0.94
    if epoch % 100 == 1:
        print("Loss:", loss)



from sklearn.linear_model import LinearRegression
rm = LinearRegression(fit_intercept=False)
Xx = X.detach().numpy()
Yy = Y.detach().numpy()
rm.fit(Xx, Yy)


print("Ours:", A)
print("Our best:", besta, "(loss=",bestL,")")
print("sklearn:", rm.coef_)
print("Truth:", coeffs)

myPreds = torch.mv(X, A)
print("RMSE is: ", ((Y-myPreds)**2).mean().detach().numpy())


plt.plot(inputs.detach().numpy(), (torch.mv(X, A)).detach().numpy())
plt.plot(inputs.detach().numpy(), np.dot(X.detach().numpy(), coeffs))
plt.show()
