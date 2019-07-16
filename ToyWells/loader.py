# loader.py - Oliver Tsang, July 2019
# This file is a helper file that takes care of loading
# and cleaning the outputs of multisim.py

# Dependencies: NumPy, h5py, PyTorch, and Scikit-Learn

import os
import numpy as np
import h5py as h5
from torch.utils import data
from sklearn.decomposition import PCA

# Class to hold the data we load
class MyData(data.TensorDataset):
    def __init__(self, fname, transform = None):
        self.archive = h5.File(fname, 'r')
        self.data = self.archive['particle_tracks']
        self.transform = transform

    def __getitem__(self, index):
        datum = self.data[index]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum

    def __len__(self):
        return len(self.data)

    def close(self):
        self.archive.close()

def normalize(dataset):
    """Set each dimension to have mean 0, variance 1; also clips the first 500 datapoints because they """
    norm_data = np.zeros(dataset.data.shape, dtype = np.float32)[500:]
    clipped_data = dataset[500:, :] # smooth out the beginning
    for i in range(dataset.data.shape[1]):
        mu  = (clipped_data[:,i]).mean()
        sig = (clipped_data[:,i]).std() # standard deviation 1 --> variance 1
        norm_data[:,i] = (clipped_data[:,i]-mu)/sig
    return norm_data

def time_lag(dataset, dt = 10000):
    """Put the current coordinates followed by the coordinates dt steps in the future"""
    start_cutoff = 500
    mean_free_data = dataset.data[start_cutoff:]
    data_shape = mean_free_data.shape
    # get rid of the means
    for i in range(data_shape[1]):
        mu = mean_free_data[:,i].mean()
        mean_free_data[:i] -= mu

    lag_data = np.zeros((data_shape[0] - dt, 2 * data_shape[1]), dtype = np.float32)
    lag_data[:, : data_shape[1]] = mean_free_data[: -dt, :]
    lag_data[:, data_shape[1] :] = mean_free_data[ dt :, :]
    return lag_data

def pcaify(dataset):
    """Runs PCA on the dataset, but only from the first 500 datapoints"""
    pca = PCA()
    pca_latent = pca.fit_transform(dataset.data[500:])
    return pca_latent

rawfile = "data/SimOutput.h5"
norfile = "data/NormalizedSimOutput.h5"
pcafile = "data/PCANormdSimOutput.h5"
tlafile = "data/TimeLaggedSimOutput.h5"

raw_sim_data = MyData(rawfile)
files = [(norfile, normalize),
         (pcafile, pcaify),
         (tlafile, time_lag)]

for file_name, fxn in files:
    needs_recompute = False
    if os.path.isfile(file_name):
        if os.path.getmtime(rawfile) > os.path.getmtime(file_name):
            needs_recompute = True
    else:
        needs_recompute = True
    if needs_recompute:
        print("Creating/updating", file_name)
        new_data = fxn(raw_sim_data)
        # save the data
        h5file = h5.File(file_name, 'w')
        h5file.create_dataset('particle_tracks', data = new_data)
        h5file.close()

# if os.path.getmtime(rawfile) > os.path.getmtime(norfile):
#     # if the raw file is more recent than the normalized file, redo it
#     print("Normalizing the simulation data...")
#     raw_sim_data = MyData(rawfile)
#     normalized_data = normalize(raw_sim_data)

#     h5file = h5.File(norfile, 'w')
#     h5file.create_dataset('', data=normalized_data)
#     h5file.close()

# if os.path.getmtime(rawfile) > os.path.getmtime(pcafile):
#     # if the pca file is more recent than the normalized file, redo it
#     print("Normalizing the simulation data...")
#     raw_sim_data = MyData(rawfile)
#     pca_data = pcaify(raw_sim_data)

#     h5file = h5.File(pcafile, 'w')
#     h5file.create_dataset('particle_tracks', data=pca_data)
#     h5file.close()

# Accessible from other files by "from loader import *_data"

# raw_sim_data = MyData(rawfile) # already loaded
sim_data = MyData(norfile)
pca_sim_data = MyData(pcafile)
tla_sim_data = MyData(tlafile)
