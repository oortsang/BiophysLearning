import numpy as np
import sklearn
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
# import pandas as pd
import h5py as h5


import loader # loader.py

from loader import sim_data as h5_sim_data # normalized
# from loader import raw_sim_data as h5_sim_data # unnormalized
sim_data = h5_sim_data[:] # numpy array

print("Done Loading!")

n_z = 2

pca = PCA()
X = pca.fit_transform(sim_data)
X_clipped = np.zeros(X.shape)
X_clipped[:, :n_z] = X[:, :n_z] # only copy the first n_z rows

reconstd = pca.inverse_transform(X_clipped)


# this loss value is not right
# pca_guesses = pca.inverse_transform(reconstd)
# pca_loss = ((pca_guesses - sim_data.data[:])**2).sum(1).mean()
# print("PCA gets a loss of %f" % pca_loss)
# plt.plot(sim_data[:,:2])
# plt.plot(reconstd[:,:2])
# plt.show()


dimensions = 8
axes = (3,3)


plt.subplot(*axes, 1)
plt.title("Latent variable(s)")
plt.plot(X_clipped[:,:n_z])

for i in range(dimensions):
    plt.subplot(*axes, 2+i)
    plt.title("Particle %d" % (i+1))
    #plt.plot(X_clipped[:,i], label = ("Input %d"% (i+1)))
    plt.plot(sim_data[:,i], label = ("Input %d"% (i+1)))
    plt.plot(reconstd[:,i], label = ("Reconstruction %d" % (i+1)))
    plt.legend()
plt.show()
