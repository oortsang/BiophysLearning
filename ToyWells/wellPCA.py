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

n_z = 1

pca = PCA()
X = pca.fit_transform(sim_data)
X_clipped = np.zeros(X.shape)
X_clipped[:, :n_z] = X[:, :n_z] # only copy the first n_z rows

reconstd = pca.inverse_transform(X_clipped)

# plt.plot(sim_data[:,:2])
# plt.plot(reconstd[:,:2])
# plt.show()


dimensions = 3
axes = (2,2)


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
