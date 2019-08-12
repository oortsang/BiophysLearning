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
force_recompute = False # recompute the transformations even if there isn't a new simulation or any updates to this file?

########## Class to hold the data we load ##########
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

##########  Helper functions  ##########

def remove_means(x, norm = False):
    """Subtract out the dimension-wise means"""
    for i in range(x.shape[1]):
        mu       = x[:, i].mean()
        x[:, i] -= mu
    if norm:
        for i in range(x.shape[1]):
            sig      = x[:, i].std()
            x[:, i] /= sig
    return x


def gaussian_blur(x, support):
    ret = np.zeros(x.shape[0])
    xx = np.abs(x)
    # assume stdev 1
    factors = np.exp(-xx*xx/2)
    return factors/factors.sum()

def bspln3(x, support):
    """Interpolating kernel """
    # But an interpolating kernel is not desired here...
    # want it to blur and lose some of the noise
    ret = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        xx = np.abs(x[i]) # rescale to fit the equation
        if xx >= 2:
            ret[i] = 0
        elif xx >= 1:
            ret[i] = -1/6*(xx-2)**3
        else:
            ret[i] = 2/3 - xx**2 + 1/2*xx**3
    return ret

def whiten(x):
    """Whitens using svd from np.linalg"""
    covar = np.cov(x[:-dt, :].T)
    u, s, vh = np.linalg.svd(covar)
    covar_sqinv = u @ np.diag(np.sqrt(1/s)) @ vh
    xw = x @ covar_sqinv.T
    # xw = u @ vh
    xw.astype(x.dtype)
    return xw

def conver(x, kfunc = gaussian_blur, support = dt//5):
    """Performs the convolution itself on the data"""
    kernel = kfunc(np.arange(-2, 2, 4/support), support)
    conv_data = np.zeros((x.shape[0] - kernel.shape[0] + 1, x.shape[1]), dtype = x.dtype)
    for j in range(conv_data.shape[-1]):
        conv_data[:,j] = np.convolve(x[:,j], kernel, mode = 'valid')
    return conv_data

def scrambler(x):
    """Performs some scrambling operation"""
    # # A few possible ways to scramble the data
    # x = 0.5*np.array([[np.sqrt(2), np.sqrt(2)], [-np.sqrt(2), np.sqrt(2)]], dtype=np.float32)
    # scramble = np.random.rand(2,2)
    # x = x @ scramble.T
    # x[:,1] += 3*np.sqrt(np.abs(x[:,0]))
    x[:,1] += 3 * np.cos(x[:,0])
    return x



##########  The actual production of new dataset objects  ##########

def normalize(dataset):
    """Set each dimension to have mean 0, variance 1; also clips the first 500 datapoints because they """
    norm_data = np.zeros(dataset.data.shape, dtype = np.float32)[start_cutoff:]
    clipped_data = dataset[start_cutoff:, :] # smooth out the beginning
    norm_data = remove_means(clipped_data, norm = False)

    # Whiten data for best results
    norm_data = whiten(norm_data)
    return norm_data


def convolve(dataset):
    """Set each dimension to have mean 0, variance 1 then convolves over the whole thing using a spline"""
    norm_data = np.zeros(dataset.data.shape, dtype = np.float32)[start_cutoff:]
    clipped_data = dataset[start_cutoff:, :] # smooth out the beginning
    norm_data = remove_means(clipped_data, norm = False)

    # Convolve
    conv_data = conver(norm_data)
    # conv_data = conver(norm_data, support = dt)

    # Whiten data for best results
    conv_data = whiten(conv_data)
    return conv_data


def time_lag(dataset):
    """Put the current coordinates followed by the coordinates dt steps in the future"""
    mean_free_data = dataset.data[start_cutoff:] # data that is mean-free, i.e., has mean 0
    data_shape = mean_free_data.shape

    mean_free_data = remove_means(mean_free_data, norm = False)

    # Convolve
    mean_free_data = conver(mean_free_data)

    # Whiten the data
    mean_free_data = whiten(mean_free_data)

    # mean_free_data = scrambler(mean_free_data)
    # mean_free_data = whiten(mean_free_data)

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


########## Setting up the files ##########

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

# Update files when necessary
for file_name, fxn in files:
    needs_recompute = False
    if os.path.getmtime(os.path.basename(__file__)) > os.path.getmtime(file_name):
        # if this python file has been modified since the last simulation
        needs_recompute = True
    elif os.path.isfile(file_name):
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

# load the files we just saved to disk
# raw_sim_data = MyData(rawfile) # already loaded
nor_sim_data = MyData(norfile)
pca_sim_data = MyData(pcafile)
tla_sim_data = MyData(tlafile)
con_sim_data = MyData(confile)
