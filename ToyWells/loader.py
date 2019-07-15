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

def pcaify(dataset):
    """Runs PCA on the dataset, but only from the first 500 datapoints"""
    pca = PCA()
    pca_latent = pca.fit_transform(dataset.data[500:])
    return pca_latent

rawfile = "data/SimOutput.h5"
norfile = "data/NormalizedSimOutput.h5"
pcafile = "data/PCANormdSimOutput.h5"

if os.path.getmtime(rawfile) > os.path.getmtime(norfile):
    # if the raw file is more recent than the normalized file, redo it
    print("Normalizing the simulation data...")
    raw_sim_data = MyData(rawfile)
    normalized_data = normalize(raw_sim_data)

    h5file = h5.File(norfile, 'w')
    h5file.create_dataset('particle_tracks', data=normalized_data)
    h5file.close()

if os.path.getmtime(rawfile) > os.path.getmtime(pcafile):
    # if the pca file is more recent than the normalized file, redo it
    print("Normalizing the simulation data...")
    raw_sim_data = MyData(rawfile)
    pca_data = pcaify(raw_sim_data)

    h5file = h5.File(pcafile, 'w')
    h5file.create_dataset('particle_tracks', data=pca_data)
    h5file.close()

# Accessible from other files by "from loader import *_data"

raw_sim_data = MyData(rawfile)
sim_data = MyData(norfile)
pca_sim_data = MyData(pcafile)
