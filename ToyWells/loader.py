# loader.py - Oliver Tsang, July 2019
# This file is a helper file that takes care of loading
# and cleaning the outputs of sim.py

# Dependencies: NumPy, h5py, PyTorch, and Scikit-Learn

import os
import numpy as np
import h5py as h5
from torch.utils import data
from sklearn.decomposition import PCA

start_cutoff = 0 # where the data begin
dt = 50 # time lag in frames
force_recompute = True # recompute the transformations even if there isn't a new simulation?

# Class to hold the data we load
class MyData(data.TensorDataset):
    def __init__(self, fname=None, transform = None, data = None):
        if fname is not None:
            self.archive = h5.File(fname, 'r')
            self.data = self.archive['particle_tracks']
        elif data is not None:
            self.data = data
        self.transform = transform

    def __getitem__(self, index):
        datum = self.data[index]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum

    def __len__(self):
        return len(self.data)

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def get_slice(self, start, stop, skip=1):
        new_set = MyData(data = self.data[:][np.arange(start, stop, skip)])
        return new_set


def bspln3(x, support):
    """Interpolating kernel """
    ret = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        xx = np.abs(x[i])/(2*support)
        if xx >= 2:
            ret[i] = 0
        elif xx >= 1:
            ret[i] = -1/6*(xx-2)**3
        else:
            ret[i] = 2/3 - xx**2 + 1/2*xx**3
    return ret

def convolve(dataset):
    """Set each dimension to have mean 0, variance 1 then convolves over the whole thing using a spline"""
    norm_data = np.zeros(dataset.data.shape, dtype = np.float32)[start_cutoff:]
    clipped_data = dataset[start_cutoff:, :] # smooth out the beginning
    for i in range(dataset.data.shape[1]):
        mu  = (clipped_data[:,i]).mean()
        norm_data[:,i] = clipped_data[:,i] - mu
        # sig = (clipped_data[:,i]).std() # standard deviation 1 --> variance 1
        # norm_data[:,i] = (clipped_data[:,i] - mu) / sig
    support = dt//2

    kernel = bspln3(np.arange(-2, 2, 4/support), support)
    # kernel = np.array([-0.125, 0, 1.25, 2, 1.25, 0, -0.125], dtype=np.float32)


    conv_data = np.zeros((norm_data.shape[0] - kernel.shape[0] + 1, norm_data.shape[1]), dtype=np.float32)
    conv_data[:,0] = np.convolve(norm_data[:,0], kernel, mode = 'valid')
    conv_data[:,1] = np.convolve(norm_data[:,1], kernel, mode = 'valid')

    covar = np.cov(conv_data[:-dt, :].T)
    u, s, vh = np.linalg.svd(covar)
    covar_sqinv = u @ np.diag(np.sqrt(1/s)) @ vh
    conv_data = conv_data @ covar_sqinv.T
    conv_data = conv_data.astype(np.float32)

    return conv_data

def normalize(dataset):
    """Set each dimension to have mean 0, variance 1; also clips the first 500 datapoints because they """
    norm_data = np.zeros(dataset.data.shape, dtype = np.float32)[start_cutoff:]
    clipped_data = dataset[start_cutoff:, :] # smooth out the beginning
    for i in range(dataset.data.shape[1]):
        mu  = (clipped_data[:,i]).mean()
        norm_data[:,i] = clipped_data[:,i] - mu
        # sig = (clipped_data[:,i]).std() # standard deviation 1 --> variance 1
        # norm_data[:,i] = (clipped_data[:,i] - mu) / sig

    covar = np.cov(norm_data[:-dt,:].T)
    u, s, vh = np.linalg.svd(covar)
    covar_sqinv = u @ np.diag(np.sqrt(1/s)) @ vh
    norm_data = norm_data @ covar_sqinv.T
    norm_data = norm_data.astype(np.float32)

    return norm_data


def time_lag(dataset):
    """Put the current coordinates followed by the coordinates dt steps in the future"""
    mean_free_data = dataset.data[start_cutoff:] # data that is mean-free, i.e., has mean 0
    data_shape = mean_free_data.shape

    # get rid of the means
    for i in range(data_shape[1]):
        mu = mean_free_data[:, i].mean()
        mean_free_data[:, i] -= mu

    # # Trying out convolution
    # support = dt//2
    # kernel = bspln3(np.arange(-2, 2, 4/support), support)
    # conv_data = np.zeros((mean_free_data.shape[0] - kernel.shape[0] + 1, mean_free_data.shape[1]), dtype=np.float32)
    # conv_data[:,0] = np.convolve(mean_free_data[:,0], kernel, mode = 'valid')
    # conv_data[:,1] = np.convolve(mean_free_data[:,1], kernel, mode = 'valid')
    # mean_free_data = conv_data

    # Whiten the data
    covar = np.cov(mean_free_data[:-dt,:].T)
    u, s, vh = np.linalg.svd(covar)
    covar_sqinv = u @ np.diag(np.sqrt(1/s)) @ vh

    mean_free_data = mean_free_data @ covar_sqinv.T

    # # optionally scramble the data
    # # rotate = 0.5*np.array([[np.sqrt(2), np.sqrt(2)], [-np.sqrt(2), np.sqrt(2)]], dtype=np.float32)
    # scramble = np.random.rand(2,2)
    # mean_free_data = mean_free_data @ scramble.T

    # copy over data from dt timesteps later
    lag_data = np.zeros((mean_free_data.shape[0] - dt, 2 * mean_free_data.shape[1]), dtype = np.float32)
    lag_data[:, : data_shape[1]] = mean_free_data[: -dt, :]
    lag_data[:, data_shape[1] :] = mean_free_data[ dt :, :]
    return lag_data

def pcaify(dataset):
    """Runs PCA on the dataset, but only after the first start_cutoff datapoints"""
    pca = PCA()
    pca_latent = pca.fit_transform(dataset.data[start_cutoff:])
    return pca_latent

rawfile = "data/SimOutput.h5"
norfile = "data/NormalizedSimOutput.h5"
pcafile = "data/PCANormdSimOutput.h5"
tlafile = "data/TimeLaggedSimOutput.h5"
confile = "data/ConvolvedSimOutput.h5"

raw_sim_data = MyData(rawfile)
files = [(norfile, normalize),
         (pcafile, pcaify),
         (tlafile, time_lag),
         (confile, convolve)]

for file_name, fxn in files:
    needs_recompute = False
    if os.path.isfile(file_name):
        if os.path.getmtime(rawfile) > os.path.getmtime(file_name):
            needs_recompute = True
    else:
        needs_recompute = True
    if needs_recompute or force_recompute:
        print("Creating/updating", file_name)
        new_data = fxn(raw_sim_data)
        # save the data
        h5file = h5.File(file_name, 'w')
        h5file.create_dataset('particle_tracks', data = new_data)
        h5file.close()

# Accessible from other files by "from loader import *_data"

# raw_sim_data = MyData(rawfile) # already loaded
nor_sim_data = MyData(norfile)
pca_sim_data = MyData(pcafile)
tla_sim_data = MyData(tlafile)
con_sim_data = MyData(confile)
