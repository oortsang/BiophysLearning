# loader.py - Oliver Tsang, July 2019
# This file is a helper file that takes care of loading
# and cleaning the outputs of sim.py

# Dependencies: NumPy, h5py, PyTorch, and Scikit-Learn

import os
import numpy as np
import h5py as h5
from torch.utils import data
from sklearn.decomposition import PCA

from tica import TICA

start_cutoff = 0
dt = 10 # time lag in frames
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
    print("Whitening...", end='')
    covar = np.cov(x[:-dt, :].T)
    u, s, vh = np.linalg.svd(covar)
    covar_sqinv = u @ np.diag(np.sqrt(1/s)) @ vh
    xw = x @ covar_sqinv.T
    # xw = u @ vh
    xw.astype(x.dtype)
    print(" done!")
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
    scramble = 2*np.random.rand(x.shape[1],x.shape[1])-1
    x = x @ scramble.T
    # x[:,1] += 3*np.sqrt(np.abs(x[:,0]))
    # x[:,1] += 3 * np.cos(x[:,0])
    return x

def shared_preprocessing(x):
    """Runs things all the process would like a part of"""
    # pca = PCA()
    # x = pca.fit_transform(x[start_cutoff:])

    norm_data = np.zeros(x.shape, dtype = np.float32)[start_cutoff:]
    clipped_data = x[start_cutoff:, :]
    norm_data = remove_means(clipped_data)
    # pca = PCA(whiten = True)
    # whitened = pca.fit_transform(norm_data)
    whitened = whiten(norm_data)
    print("Running TICA...", end = '')
    from sys import stdout
    stdout.flush()
    tica = TICA(whitened, dt, kinetic_map = True)
    # tica = TICA(whitened, dt, kinetic_map = False)
    tic = tica.transform(whitened)
    print(" done!")

    # tic = whiten(scrambler(tic[:, : 100]))
    return tic[:, : 50]

##########  The actual production of new dataset objects  ##########

def normalize(data):
    """Set each dimension to have mean 0, variance 1"""
    return data

def convolve(data):
    """Set each dimension to have mean 0, variance 1 then convolves over the whole thing using a spline"""
    conv_data = conver(data)

    # # run tica
    # tica = TICA(conv_data, dt, kinetic_map = True)
    # conv_data = tica.transform(conv_data)

    return conv_data


def time_lag(data):
    """Put the current coordinates followed by the coordinates dt steps in the future"""
    # # Convolve
    data = conver(data)
    # # # Whiten the data
    # # data = whiten(data)

    # # run tica
    # tica = TICA(data, dt, kinetic_map = True)
    # data = tica.transform(data)

    # data = scrambler(data)
    # data = whiten(data)

    # copy over data from dt timesteps later
    lag_data = np.zeros((data.shape[0] - dt, 2 * data.shape[1]), dtype = np.float32)
    lag_data[:, : data.shape[1]] = data[: -dt, :]
    lag_data[:, data.shape[1] :] = data[ dt :, :]
    return lag_data

def pcaify(data):
    """Runs PCA on the dataset, but only after the first start_cutoff datapoints"""
    pca = PCA()
    pca_latent = pca.fit_transform(data[start_cutoff:])
    return pca_latent


########## Setting up the files ##########

rawfile = "data/SimFeatures.h5"
norfile = "data/NormalizedSimFeatures.h5"
pcafile = "data/PCANormdSimFeatures.h5"
tlafile = "data/TimeLaggedSimFeatures.h5"
confile = "data/ConvolvedSimFeatures.h5"

raw_sim_data = MyData(rawfile)
files = [(norfile, normalize),
         (pcafile, pcaify),
         (tlafile, time_lag),
         (confile, convolve)]

shared_whitening = None

# Update files when necessary
for file_name, fxn in files:
    needs_recompute = force_recompute
    if os.path.getmtime(os.path.basename(__file__)) > os.path.getmtime(file_name):
        # if this python file has been modified since the last simulation
        needs_recompute = True
    elif os.path.isfile(file_name):
        if os.path.getmtime(rawfile) > os.path.getmtime(file_name):
            needs_recompute = True
    else:
        needs_recompute = True

    if needs_recompute and shared_whitening is None:
        shared_whitening = shared_preprocessing(raw_sim_data.data[:])

    if needs_recompute:
        print("Creating/updating", file_name)
        new_data = fxn(shared_whitening).astype(np.float32)
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
