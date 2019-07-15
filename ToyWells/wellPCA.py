# wellPCA.py - Oliver Tsang, July 2019
# File to demonstrate perfotmance of PCA (Principal Component Analysis)
# on the same data set as the variational autoencoder.
# The n_z parameter controls the dimensionality of the latent variables,
# i.e., the number of axes that are not erased before we attempt to
# reconstruct the original data set.

# Package dependencies: NumPy, Scikit-Learn, matplotlib, and h5py (for loader.py)

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
# import h5py as h5


import loader # loader.py is in charge of loading from the *.h5 files (binary file type)

from loader import sim_data as h5_sim_data # normalized
# from loader import raw_sim_data as h5_sim_data # unnormalized
sim_data = h5_sim_data[:] # numpy array

print("Done Loading!")

dimensions = 5 # number of dimensions to expect from the simulation data -- make sure it matches the output dimension from multisim.py
n_z = 2 # can play with this to see the performance change
plot_axes = (3,3) # choose how to display the plots - there should be (1+dimensions) plots (first is to show the latent/saved variables)

pca = PCA()
latent_vars = pca.fit_transform(sim_data)
reduced_latent_vars = np.zeros(latent_vars.shape)
reduced_latent_vars[:, :n_z] = latent_vars[:, :n_z] # only copy the first n_z rows

reconstructed = pca.inverse_transform(reduced_latent_vars)

plt.subplot(*axes, 1)
plt.title("Latent variable(s)")
plt.plot(reduced_latent_vars[:,:n_z])

for i in range(dimensions):
    plt.subplot(*plot_axes, 2+i)
    plt.title("Particle %d" % (i+1))
    plt.plot(sim_data[:,i], label = ("Input %d"% (i+1)))
    plt.plot(reconstructed[:,i], label = ("Reconstruction %d" % (i+1)))
    plt.legend()
plt.show()
